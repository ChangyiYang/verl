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

import glob
import json
import logging
import os
import struct

logger = logging.getLogger(__name__)

_FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


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

    # The export renames as it converts (model.layers.N.self_attn.* vs
    # layers.N.attn.*), so match on the longest common tail rather than on the
    # full string. Two tensors never share a tail this long in practice, and a
    # collision would have to be between an fp8 and a non-fp8 tensor to matter.
    def _tail(n: str) -> str:
        # Normalise the one rename we know about first: SGLang/HF spell the block
        # ``self_attn`` while Megatron-Bridge's DSv4 mapping writes ``attn``.
        # Without this the tails differ in the very segment we keep.
        n = n.replace(".self_attn.", ".attn.")
        parts = n.split(".")
        # Three components, not two: two would reduce both ``attn.wkv.weight``
        # (fp8) and ``compressor.wkv.weight`` (bf16) to ``wkv.weight`` -- a
        # collision between exactly the pair whose dtypes disagree.
        return ".".join(parts[-3:]) if len(parts) >= 3 else n

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
