# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decide which params to fp8-quantize by reading the checkpoint, not by name.

SGLang's ``update_weights_from_tensor`` does not re-quantize, so the trainer has
to ship pre-quantized fp8 codes + scales, which means the *sender* must know
which params are fp8. The sender has no model object -- that lives in the rollout
process -- so ``fp8_utils.should_quantize_param`` guesses from an allowlist of
Llama-style names. DeepSeek-V4 calls its linears wq_a / wq_b / wkv / wo_b /
w1..w3, matched nothing, and every weight shipped as BF16 into an fp8 slot.

vLLM does not have this problem because it quantizes on the *receiver*, where it
can just ask ``module.weight.dtype``.

The sender does have an authoritative source it was not using: the checkpoint
itself. safetensors stores a JSON header per shard with every tensor's dtype, and
reading only the headers is cheap -- measured 0.36 s for all 46 shards / 69,143
tensors of DeepSeek-V4-Flash-FP8, with no tensor data touched.

This is exact whenever the rollout serves the checkpoint's own quantization
(the usual case, and ours). If an engine ever decides its own layout at load
time -- serving a BF16 checkpoint as fp8, say -- the checkpoint no longer
describes the destination and a real receiver-side handshake would be needed.
``build_ckpt_fp8_predicate`` returns None when it cannot answer, so callers fall
back rather than silently quantizing nothing.
"""

import functools
import glob
import json
import logging
import os
import struct

logger = logging.getLogger(__name__)

_FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


def _tail(n: str) -> str:
    """Longest-common-tail key for matching checkpoint names to export names.

    The export renames as it converts (model.layers.N.self_attn.* vs
    layers.N.attn.*), so match on the longest common tail rather than on the
    full string. Normalise the one rename we know about first: SGLang/HF spell
    the block ``self_attn`` while Megatron-Bridge's DSv4 mapping writes
    ``attn`` -- without this the tails differ in the very segment we keep.
    Three components, not two: two would reduce both ``attn.wkv.weight`` (fp8)
    and ``compressor.wkv.weight`` (bf16) to ``wkv.weight`` -- a collision
    between exactly the pair whose dtypes disagree.
    """
    n = n.replace(".self_attn.", ".attn.")
    parts = n.split(".")
    return ".".join(parts[-3:]) if len(parts) >= 3 else n


# Both entry points are memoised per model path. They walk EVERY shard of the
# checkpoint and read its safetensors header -- 46 shards for DSv4-FP8 -- and
# nothing upstream cached the result: _quant_predicate rebuilt the predicate on
# every _fp8_spec call, which happens on the seed, on EVERY steady sync, and in
# the verify path, on every trainer rank. That is 80 ranks x 46 shards = 3680
# NFS opens per sync, landing on the same nodes that host the SGLang servers.
# The route-B run took 451 ActorDiedErrors and stalled with 16 requests never
# finishing; the name-allowlist route reads no files at all and took zero.

@functools.lru_cache(maxsize=4)
def read_checkpoint_dtypes(model_path: str) -> dict[str, str]:
    """``{tensor_name: safetensors_dtype_string}`` from the shard headers alone."""
    out: dict[str, str] = {}
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    for shard in shards:
        try:
            with open(shard, "rb") as fh:
                (hdr_len,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(hdr_len))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning("fp8 dtype map: cannot read header of %s (%s)", shard, e)
            continue
        for name, meta in header.items():
            if name != "__metadata__" and isinstance(meta, dict) and "dtype" in meta:
                out[name] = meta["dtype"]
    return out


@functools.lru_cache(maxsize=4)
def build_ckpt_fp8_predicate(model_path: str):
    """A ``name -> bool`` predicate, or None if the checkpoint cannot answer.

    Returning None (rather than a predicate that says False for everything) is
    deliberate: "no fp8 params" and "I could not read the checkpoint" must not
    look the same to the caller. The former is a legitimate answer, the latter is
    the failure that shipped BF16 into fp8 slots for three runs without a word.
    """
    dtypes = read_checkpoint_dtypes(model_path)
    if not dtypes:
        logger.warning("fp8 dtype map: no safetensors headers under %s; falling back", model_path)
        return None
    fp8_names = {n for n, d in dtypes.items() if d in _FP8_DTYPES}
    if not fp8_names:
        logger.info("fp8 dtype map: %s has no fp8 tensors; falling back", model_path)
        return None

    # Two tensors never share a 3-component tail in practice, and a collision
    # would have to be between an fp8 and a non-fp8 tensor to matter.
    fp8_tails = {_tail(n) for n in fp8_names}
    all_tails = {_tail(n) for n in dtypes}
    logger.info(
        "fp8 dtype map: %d tensors read from %d shards, %d fp8 (%d distinct tails)",
        len(dtypes),
        len(glob.glob(os.path.join(model_path, "*.safetensors"))),
        len(fp8_names),
        len(fp8_tails),
    )

    def predicate(param_name: str) -> bool:
        if param_name in fp8_names:
            return True
        t = _tail(param_name)
        if t in fp8_tails:
            return True
        # Known to the checkpoint and not fp8 -> a confident False.
        if param_name in dtypes or t in all_tails:
            return False
        # Unknown name (scales the export adds, fused params, ...) -> not fp8.
        return False

    return predicate


@functools.lru_cache(maxsize=4)
def build_ckpt_fp32_predicate(model_path: str):
    """A ``name -> bool`` predicate for params the checkpoint stores in FP32,
    or None if the checkpoint cannot answer.

    DSv4 keeps a handful of sensitive families in fp32 on disk and in the
    serving engine (hyper-connection coefficients, ape compressor position
    embeddings, attention sinks, router e_score_correction_bias -- ~68 MB
    total). The wire used to fold every non-fp8 float to the rollout dtype,
    silently costing these params 16 mantissa bits per sync (measured rel err
    p50 1.35e-3, max 3.9e-3), invisibly to the verify sweep because the replay
    folds identically. The sender must know which params to keep fp32, and the
    routing has to be identical on every rank -- including ranks that do not
    own the param and see no tensor -- so the decision comes from the
    checkpoint's own headers, exactly like the fp8 predicate above.

    Quantization scale grids (``<stem>.scale`` / ``*_scale_inv``) are also F32
    in the headers but are NOT this predicate's business: they ride the wire's
    dedicated scale group and are excluded here.
    """
    dtypes = read_checkpoint_dtypes(model_path)
    if not dtypes:
        logger.warning("fp32 dtype map: no safetensors headers under %s; falling back", model_path)
        return None

    def _is_scale(n: str) -> bool:
        return n.endswith(".scale") or n.endswith("_scale_inv")

    fp32_names = {n for n, d in dtypes.items() if d == "F32" and not _is_scale(n)}
    if not fp32_names:
        logger.info("fp32 dtype map: %s stores no non-scale fp32 tensors", model_path)
        return None
    fp32_tails = {_tail(n) for n in fp32_names}
    logger.info(
        "fp32 dtype map: %d fp32 non-scale tensors (%d distinct tails) out of %d",
        len(fp32_names),
        len(fp32_tails),
        len(dtypes),
    )

    def predicate(param_name: str) -> bool:
        if _is_scale(param_name):
            return False
        return param_name in fp32_names or _tail(param_name) in fp32_tails

    return predicate
