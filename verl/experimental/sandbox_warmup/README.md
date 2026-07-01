# Per-rollout sandbox warm-up

Pre-warm each rollout's execution environment (e.g. a Modal sandbox: **create +
terminal-state setup + file upload**) *ahead* of when the rollout needs it, so the
warm-up cost overlaps with rollout execution instead of blocking the critical path.

Warm-up is **per rollout instance** (not per prompt): under GRPO the `n` rollouts of a
prompt share a `uid` and run concurrently, and each mutates its own sandbox, so each
needs its own env. The framework matches a warmed env back to its exact rollout via a
minted `_warm_key`.

## What you write vs what the framework owns

- **You** implement a `SandboxWarmupPlugin` (backend-specific) and point config at it:
  ```yaml
  actor_rollout_ref:
    rollout:
      agent:
        warmup_plugin_path: my_pkg.my_modal_plugin.MyModalWarmupPlugin
        warmup_max_concurrency: 8   # in-flight warms cap per worker
  ```
  ```python
  class MyModalWarmupPlugin(SandboxWarmupPlugin):
      async def warm(self, warm_key, sample, config):
          # modal spawn + set terminal state + upload files, keyed off `sample`
          return handle
      async def is_alive(self, handle): ...   # optional
      async def release(self, handle): ...
  ```
- **The framework** owns the generic machinery: minting `_warm_key`, scheduling warms
  with bounded concurrency, awaiting in-flight warms, liveness re-warm, inline fallback,
  and leak-free drain (`WarmupRegistry`).

## End-to-end flow (one training-step wave)

1. **Startup** (per rollout worker): if `warmup_plugin_path` is set, the worker loads the
   plugin and creates a per-worker `WarmupRegistry`, registered as the process-current one.
2. **Warm-up** (start of `AgentLoopWorker.generate_sequences(batch)`, before dispatch):
   for each row `i` the framework mints `key = uuid4`, stamps it as the `_warm_key`
   non-tensor column, and calls `registry.start_warm(key, sample_i)` (background,
   concurrency-bounded, non-blocking → warms overlap with rollout execution).
3. **Consume** (inside your agent loop, per rollout): the `_warm_key` arrives in the run
   kwargs (it is a non-tensor column, so it flows via `kwargs = {k: v[i] ...}`):
   ```python
   from verl.experimental.sandbox_warmup import acquire_warm_env, release_warm_env
   env = await acquire_warm_env(kwargs["_warm_key"])   # this rollout's warmed env
   try:
       ...  # drive the rollout using `env`
   finally:
       await release_warm_env(kwargs["_warm_key"])
   ```
   `acquire` returns a *ready & liveness-checked* env — awaiting the in-flight warm if it
   is not done, re-warming a dead one, or warming inline on a miss so a rollout never hangs.
4. **Drain** (end of `generate_sequences`, `finally`): `registry.drain()` releases any
   warmed-but-unconsumed env as a leak backstop; the normal path already released per rollout.

Default is off (`warmup_plugin_path: null`) → zero behavior change.
