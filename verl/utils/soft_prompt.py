# Soft-prompt-only training on a frozen backbone.
#
# WHY NOT peft.PromptTuningConfig, which is the obvious choice:
# PEFT prompt tuning prepends *embeddings* inside forward(). verl generates with
# vLLM and hands the rollout engine weights, not activations -- there is no
# channel through which a prepended embedding could reach the sampler. The
# rollout would silently run without the prompt while training ran with it.
#
# So the prompt lives in the model's own vocabulary instead: rows of
# `embed_tokens` that the tokenizer never emits. Those rows are addressable by
# ordinary token id, so the sampler gets the prompt by prepending ids, and verl's
# existing weight sync ships the trained rows to vLLM with no changes at all.
# This is the same layout the offline trainer in this program uses, so prompts
# are interchangeable between the two.
#
# The backbone stays frozen in the strict sense: the only tensor that receives a
# non-zero gradient is `embed_tokens.weight`, and a hook zeroes every row of that
# gradient except the reserved ones. Making the whole embedding "trainable" and
# masking is deliberate -- it costs optimizer state for rows that never move
# (~470MB/rank of Adam state for a 1.5B on 4 ranks) and buys a patch that needs
# no change whatsoever to the FSDP->vLLM sharding manager. A separate small
# nn.Parameter spliced into forward() would be cheaper and would then require
# teaching the sync path to write it back into the embedding before every
# rollout; that is more code in a place where a silent mistake produces a rollout
# that just quietly ignores the prompt.

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__file__)


def reserved_vocab_ids(model_vocab_size: int, registered_ids, count: int) -> list[int]:
    """The first `count` model rows the tokenizer never emits.

    Qwen checkpoints pad the output vocabulary above the tokenizer's registered
    vocabulary, and the registered ids are NOT contiguous -- `len(tokenizer)` is
    not a usable boundary. So take the sorted complement rather than assuming a
    tail region is free.
    """
    registered = set(int(i) for i in registered_ids)
    free = [i for i in range(model_vocab_size) if i not in registered]
    if len(free) < count:
        raise ValueError(
            f"soft prompt needs {count} reserved vocabulary rows but the model has only "
            f"{len(free)} rows the tokenizer never emits (vocab_size={model_vocab_size}). "
            f"Reduce num_tokens to at most {len(free)}."
        )
    return free[:count]


_ids_cache: dict = {}


def soft_prompt_ids_from_config(config, tokenizer) -> list[int]:
    """The reserved ids for this run, or [] when soft prompting is off.

    Both the trainer and the rollout must agree on these ids exactly -- they are
    what carries the prompt from one to the other -- so both sides derive them
    from the same function rather than each computing "the last N rows".
    """
    try:
        num_tokens = int(config.actor_rollout_ref.model.soft_prompt_num_tokens)
    except Exception:
        return []
    if num_tokens <= 0:
        return []

    from transformers import AutoConfig

    path = config.actor_rollout_ref.model.path
    key = (path, num_tokens)
    if key not in _ids_cache:
        vocab_size = int(AutoConfig.from_pretrained(path, trust_remote_code=True).vocab_size)
        _ids_cache[key] = reserved_vocab_ids(vocab_size, tokenizer.get_vocab().values(), num_tokens)
    return _ids_cache[key]


def install_soft_prompt(
    module: nn.Module,
    soft_ids: list[int],
    *,
    init_ids: list[int] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Freeze `module`, unfreeze only the reserved embedding rows, initialise them.

    Must be called BEFORE FSDP wrapping: it mutates requires_grad and writes
    tensor data, and it registers the masking hook on the unsharded parameter.

    Returns the soft-id tensor so the caller can log / persist it.
    """
    emb = module.get_input_embeddings()
    if emb is None:
        raise ValueError("model has no input embeddings; cannot host a soft prompt")

    weight = emb.weight
    ids = torch.tensor(sorted(soft_ids), dtype=torch.long, device=weight.device)

    # Initialise from real vocabulary rows. Sampling from the existing embedding
    # distribution rather than from N(0, 1) matters: a randomly-scaled row is far
    # outside the manifold the frozen backbone was trained on, and the first
    # forward passes are then dominated by getting back to a sane norm.
    # @changyi_yang specified "从所有token里随机抽样就行了".
    with torch.no_grad():
        if init_ids is None:
            src = torch.randint(
                0, weight.shape[0], (len(ids),), generator=generator, device=weight.device
            )
        else:
            src = torch.tensor(init_ids, dtype=torch.long, device=weight.device)
        weight[ids] = weight[src].clone()

    module.requires_grad_(False)
    weight.requires_grad_(True)

    # Tied embeddings: lm_head shares storage with embed_tokens on some
    # checkpoints. Then training a row also changes that row's output logit.
    # Those rows are ids the tokenizer never emits, but they DO compete in the
    # softmax, so an unbounded row can start winning argmax and the model emits
    # a token that decodes to nothing. Flag it loudly rather than discover it in
    # a degenerate sample.
    if getattr(getattr(module, "config", None), "tie_word_embeddings", False):
        logger.warning(
            "tie_word_embeddings=True: the soft prompt rows are also lm_head rows, so "
            "they enter the output softmax over ids the tokenizer never emits. Watch for "
            "undecodable tokens in rollouts."
        )

    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    logger.info(
        "soft prompt: %d tokens in rows %d..%d; trainable tensors %s",
        len(ids), int(ids[0]), int(ids[-1]), trainable,
    )
    return ids


def register_grad_mask(module: nn.Module, soft_ids: torch.Tensor) -> None:
    """Zero every embedding-gradient row except the reserved ones.

    Call AFTER FSDP wrapping. Every input token contributes gradient to its own
    embedding row, so without this the whole embedding matrix trains and the
    backbone is not frozen -- which is the premise of the entire program, not a
    detail. Registering before `fully_shard` would not work: FSDP2 replaces the
    parameter with a DTensor and a hook on the pre-shard tensor never fires.
    """
    weight = module.get_input_embeddings().weight
    ids = soft_ids.to(weight.device)

    full = torch.zeros(weight.shape[0], 1, dtype=torch.float32, device=weight.device)
    full[ids] = 1.0

    if isinstance(weight, torch.distributed.tensor.DTensor):
        from torch.distributed.tensor import distribute_tensor

        # Same mesh and placements as the parameter, so the elementwise multiply
        # below stays local and needs no collective.
        mask = distribute_tensor(full, weight.device_mesh, weight.placements)
    else:
        mask = full

    state = {"checked": False}

    def _mask_grad(grad):
        out = grad * mask.to(grad.dtype)
        if not state["checked"]:
            state["checked"] = True
            local = out.to_local() if hasattr(out, "to_local") else out
            logger.info("soft-prompt grad mask active; local nonzero rows %d",
                        int((local.abs().sum(dim=-1) > 0).sum()))
        return out

    weight.register_hook(_mask_grad)
