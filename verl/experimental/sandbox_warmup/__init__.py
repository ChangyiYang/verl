# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Per-rollout sandbox warm-up.

Pre-warm each rollout's execution environment (e.g. a Modal sandbox: create +
terminal-state setup + file upload) *ahead* of when the rollout needs it, so the
warm-up cost overlaps with rollout execution instead of blocking the critical path.

Design (see README.md for the full end-to-end flow):

* The env is warmed **per rollout instance** and matched back to that exact rollout
  via a framework-minted ``_warm_key`` stamped on the rollout's row.
* A user-supplied :class:`SandboxWarmupPlugin` owns the backend-specific logic
  (``warm`` / ``release`` / optional ``is_alive``); it is wired in via the
  ``actor_rollout_ref.rollout.agent.warmup_plugin_path`` config.
* A per-worker :class:`WarmupRegistry` owns the generic machinery (bounded-concurrency
  scheduling, awaiting in-flight warms, liveness re-warm, inline fallback, drain).
"""

from verl.experimental.sandbox_warmup.plugin import SandboxWarmupPlugin
from verl.experimental.sandbox_warmup.registry import (
    WARM_KEY,
    WarmupRegistry,
    acquire_warm_env,
    current_registry,
    release_warm_env,
    set_current_registry,
)

__all__ = [
    "WARM_KEY",
    "SandboxWarmupPlugin",
    "WarmupRegistry",
    "acquire_warm_env",
    "current_registry",
    "release_warm_env",
    "set_current_registry",
]
