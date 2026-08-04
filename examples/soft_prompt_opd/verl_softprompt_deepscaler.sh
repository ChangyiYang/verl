#!/usr/bin/env bash
# SOFT-PROMPT OPD on the DeepScaleR pair in verl -- the actual research run.
#
# Differs from jobs/verl_opd_deepscaler.sh (the full-model acceptance run) in
# exactly three ways, and they are the whole point:
#   1. soft_prompt_num_tokens=16   -- backbone frozen, only 16 reserved
#      vocabulary rows train. 16 and not 128 because every DeepScaleR recovery
#      number this has to be compared against (rounds 1-4: 49.15 / 58.47 /
#      77.97 / 66.95%) was measured on a 16-token prompt. The DAPO line uses 128;
#      mixing the two would make the comparison meaningless.
#   2. optim.lr=1e-3               -- @zhen-zhang's soft_prompt_lr. Applies to the
#      one trainable tensor. NOTE this is @changyi_yang's verified value for
#      CONTINUING from a trained prompt; his verified cold-start value is 1e-2,
#      and this run is a cold start (random vocabulary init). Flagged in thread.
#   3. max_response_length=16384   -- the acceptance run capped at 8,192 and hit
#      it on 34.3% of samples, which makes its numbers incomparable to the
#      offline rounds (measured under a 32,768 contract). DeepScaleR's mean
#      response is ~10,439 tokens.
#
# Student = DeepSeek-R1-Distill-Qwen-1.5B (the frozen base of this program's pair)
# Teacher = DeepScaleR-1.5B-Preview       (the published RL fine-tune)
# Data    = DeepScaleR-Preview-Dataset, 40,315 rows, rendered through THIS
#           program's INSTRUCTION contract (verl-scripts/prepare_deepscaler_verl.py),
#           not verl's example instruction strings. Val = AIME 2024, 30 questions.
#

# Hyperparameters from @zhen-zhang 2026-08-03, amended by @changyi_yang:
#   loss_mode k1 / use_policy_gradient true / use_task_rewards false
#   distillation_loss_coef 1.0 ("kl是第一个") / temperature 1.0
#   prompts_per_batch 256 x rollouts_per_prompt 4 / ppo_epochs 1 / 50 steps
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
export RAY_TMPDIR=/tmp/ray-cy-sp2
mkdir -p "$RAY_TMPDIR"

cd $MINE/verl-upstream

STUDENT_MODEL=${STUDENT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-agentica-org/DeepScaleR-1.5B-Preview}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-8}

prompts_per_batch=${PROMPTS_PER_BATCH:-256}
rollouts_per_prompt=${ROLLOUTS_PER_PROMPT:-4}
total_steps=${TOTAL_STEPS:-50}
# 16,384: above DeepScaleR's ~10,439-token mean, so truncation should be a small
# tail rather than the 34.3% the 8,192 acceptance run clipped.
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
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
  actor_rollout_ref.model.soft_prompt_num_tokens=16 \
  actor_rollout_ref.actor.optim.lr=1e-3 \
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
  trainer.project_name=verl_opd_deepscaler_softprompt \
  trainer.experiment_name=r1distill1.5b_softprompt16_from_deepscaler \
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
