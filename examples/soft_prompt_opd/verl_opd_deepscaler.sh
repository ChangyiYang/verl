#!/usr/bin/env bash
# E2E ACCEPTANCE RUN: DeepScaleR OPD in verl.
#   @changyi_yang's stated bar -- "e2e的验收标准就是deepscaler的opd跑起来".
#
# Student = DeepSeek-R1-Distill-Qwen-1.5B (the frozen base of this program's pair)
# Teacher = DeepScaleR-1.5B-Preview       (the published RL fine-tune)
# Data    = DeepScaleR-Preview-Dataset, 40,315 rows, rendered through THIS
#           program's INSTRUCTION contract (verl-scripts/prepare_deepscaler_verl.py),
#           not verl's example instruction strings. Val = AIME 2024, 30 questions.
#
# This is FULL-MODEL OPD, not soft-prompt OPD. Deliberate ordering: it satisfies
# the stated acceptance bar with config alone now that stage 0 passes, and it
# establishes that the DeepScaleR pair itself runs through verl's distillation
# path before any soft-prompt code is added on top. Confounding the two is the
# mistake this project has already paid for once.
#
# Hyperparameters from @zhen-zhang 2026-08-03, amended by @changyi_yang:
#   loss_mode k1 / use_policy_gradient true / use_task_rewards false
#   distillation_loss_coef 1.0 ("kl是第一个") / temperature 1.0
#   prompts_per_batch 256 x rollouts_per_prompt 4 / ppo_epochs 1 / 50 steps
# soft_prompt_lr 1e-3 does NOT apply here -- there is no soft prompt yet; the
# actor lr is an ordinary full-model RL lr.
#
# GPU split: 8 trainer/rollout on the head node + 8 teacher on the worker.
# real_train_batch_size is 256 x 4 = 1024 and must divide by
# trainer.n_gpus_per_node; 1024/8 = 128. (6 would fail -- 1024/6 is not an
# integer, which is why the first attempt used 4.)
# Bring the cluster up first with jobs/verl_ray_up.sh and export RAY_ADDRESS.
set -xeuo pipefail
MINE=/home/changyi/.slock/agents/7dc66911-8db2-4303-a784-cf0116e84ecb

export PATH=/home/changyi/miniconda3/envs/verl-opd/bin:$PATH
export PYTHONPATH=$MINE/verl-upstream
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# See notes/verl-opd.md blocker 5: vllm's cumem_allocator .so is dlopened without
# torch's preload having run and cannot find libnvrtc.so.13 on its own.
export LD_LIBRARY_PATH=/home/changyi/miniconda3/envs/verl-opd/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export RAY_TMPDIR=/tmp/ray-cy-ds2
mkdir -p "$RAY_TMPDIR"

cd $MINE/verl-upstream

STUDENT_MODEL=${STUDENT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-agentica-org/DeepScaleR-1.5B-Preview}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-8}

prompts_per_batch=${PROMPTS_PER_BATCH:-256}
rollouts_per_prompt=${ROLLOUTS_PER_PROMPT:-4}
total_steps=${TOTAL_STEPS:-50}
# 8,192 rather than the eval contract's 32,768: DeepScaleR's mean response is
# ~10.4k tokens, so this truncates a real fraction. It is a cost/throughput
# choice for the acceptance run, and it is the first thing to raise for a run
# whose NUMBERS are meant to be compared to the offline rounds.
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['/home/changyi/data/deepscaler_verl/train.parquet']" \
  data.val_files="['/home/changyi/data/deepscaler_verl/test.parquet']" \
  data.train_batch_size=${prompts_per_batch} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${prompts_per_batch} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_num_tokens} \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.offload_policy=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n=${rollouts_per_prompt} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_num_tokens} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=verl_opd_deepscaler \
  trainer.experiment_name=r1distill1.5b_from_deepscaler_fsdp2 \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=${total_steps} \
  trainer.total_epochs=1 \
  distillation.enabled=True \
  distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.5 \
  distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.topk=64 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.distillation_loss_coef=1.0 \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  "$@"
