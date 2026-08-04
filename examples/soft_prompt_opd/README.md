# Soft-prompt OPD

Online distillation (OPD) where the **only** trainable thing is a short soft
prompt and the backbone stays frozen. The research question behind it: can a
short soft prompt on a frozen base model recover the gains of a published RL
fine-tune?

The engine change is in `verl/utils/soft_prompt.py` plus small hooks in
`workers/engine/fsdp/transformer_impl.py`, `experimental/agent_loop/*`,
`workers/config/model.py` and `utils/config.py`. This directory holds the data
prep, the test, and the launch scripts that were actually used.

## Enabling it

```
actor_rollout_ref.model.soft_prompt_num_tokens=16     # 0 disables
actor_rollout_ref.model.soft_prompt_init_path=null    # or a .safetensors with "prompt_embeddings"
actor_rollout_ref.actor.optim.lr=1e-3                 # applies to the one trainable tensor
```

Mutually exclusive with `lora_rank > 0`.

## Why reserved vocabulary rows and not `peft.PromptTuningConfig`

PEFT prompt tuning prepends *embeddings* inside `forward()`. verl hands the
rollout engine *weights*, so there is no channel by which a prepended embedding
reaches vLLM — the rollout would sample from the bare base model while the
trainer optimised a prompt nothing conditioned on, **and nothing would error**.
Reserved rows are addressable by token id, so the existing FSDP→vLLM weight sync
carries the prompt with no changes to the sync path.

## Three things that fail silently

None of these raise. All of them still produce a healthy-looking loss curve.
That is why this recipe asserts instead of eyeballing curves.

1. **The teacher must not see the soft tokens.** The objective is
   `KL(student(·|soft,x,y_<t) ‖ teacher(·|x,y_<t))`. The teacher is the published
   RL model and its rows for those ids are untrained, so conditioning on them
   corrupts every downstream hidden state. `_compute_teacher_logprobs` strips the
   ids and left-pads the result back to the student's length — the pad lands in
   the prompt region `response_mask` excludes, but the lengths must match or
   student position *t* pairs with teacher position *t − n_soft*.
2. **The gradient mask.** Every input token contributes gradient to its own
   embedding row, so without the mask the whole embedding matrix trains and the
   backbone is not frozen. `test_soft_prompt.py` puts ordinary token ids in the
   batch alongside the soft ids and asserts only the reserved rows receive
   gradient. `actor/grad_norm` being smaller than a full-model run is suggestive,
   not proof.
3. **A stale vLLM prefix cache.** Prefix caching is keyed on token ids; the soft
   prompt's ids never change while the rows behind them change every step. verl
   already resets the cache on the weight-sync path, but that is gated on
   `free_cache_engine` and skipped in `rollout.mode=standalone`. A soft prefix is
   byte-identical across every step and every prompt *by construction*, so it is
   guaranteed to hit a stale entry where an ordinary varying prompt might not.
   `validate_config` now refuses to start in those configurations.

## Files

| file | what it is |
|---|---|
| `test_soft_prompt.py` | CPU, seconds. Asserts the soft prompt is the only thing that trains. |
| `prepare_deepscaler_verl.py` | DeepScaleR-Preview-Dataset → verl parquet. Note `data_source="math_dapo"`: verl dispatches its reward function on that string and raises `NotImplementedError` on the real dataset id. |
| `verl_ray_up.sh` | Assemble a multi-node ray cluster over ssh. |
| `verl_opd_smoke_fsdp.sh` | Stage-0 smoke, Qwen3-0.6B ← Qwen3-1.7B on gsm8k. Full-model. |
| `verl_softprompt_smoke.sh` | Same, with `soft_prompt_num_tokens=16`. |
| `verl_opd_deepscaler.sh` | Full-model OPD, R1-Distill-Qwen-1.5B ← DeepScaleR-1.5B-Preview. |
| `verl_softprompt_deepscaler.sh` | The research run: same pair, 16-token soft prompt. |

The launch scripts carry absolute paths for one specific cluster. They are here
because their comments record the environment constraints that took a while to
find (no prebuilt flash-attn for torch 2.11; vllm 0.26.0 pins torch 2.11.0
exactly and is built for CUDA 13; vllm's `cumem_allocator.so` needs
`site-packages/nvidia/cu13/lib` on `LD_LIBRARY_PATH` or it reports the
misleading "cumem allocator is not supported on current platform").

## Test

```bash
PYTHONPATH=. python3 examples/soft_prompt_opd/test_soft_prompt.py
# PASS: soft prompt is the only thing that trains
```
