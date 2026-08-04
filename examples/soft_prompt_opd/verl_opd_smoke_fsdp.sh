#!/usr/bin/env bash
# verl OPD stage-0 smoke, FSDP2 path, text-only, Qwen3-0.6B <- Qwen3-1.7B on GSM8K.
#
# WHY NOT upstream's own smallest example: the only 0.6B example upstream ships is
# run_qwen3_0.6b_opd_veomni.sh, and veomni is not in the env (and is not what we
# need -- every DeepScaleR result we are matching is FSDP). The FSDP examples
# upstream ships start at 5.4B. So this is run_qwen3_5_4b_fsdp.sh with the model
# pair swapped down and the vision bits removed:
#   - data.image_key=images dropped (that script is Qwen3.5-VL on geo3k)
#   - geo3k -> gsm8k, which is already materialised at ~/data/gsm8k
#   - teacher_ep dropped (35B-A3B is MoE, Qwen3-1.7B is dense)
# Every distillation.* value is left exactly as upstream set it.
#
# Stage-0 PASS = the log reaches its first `step:` line. Nothing about the
# soft prompt is exercised here on purpose: "does this env run verl OPD at all"
# and "is our soft-prompt change correct" must not be confounded. Ten FSDP
# attempts were expensive earlier precisely because they were.
#
# ATTENTION BACKEND, stage-0 only: sdpa, not flash_attention_2.
# There is no prebuilt flash-attn wheel for torch 2.11 (newest upstream ships is
# cu12torch2.9), and vllm 0.26.0 pins torch==2.11.0 exactly, so torch cannot be
# moved down without moving vllm too. flash-attn is building from source in
# parallel; until it lands this smoke runs on sdpa with use_remove_padding=False,
# because verl hard-imports flash_attn.bert_padding on the remove-padding path
# (verl/utils/attention_utils.py:30). This costs throughput, not correctness, and
# stage 0 only asks whether verl OPD runs in this env at all.
set -xeuo pipefail
MINE=/home/changyi/.slock/agents/7dc66911-8db2-4303-a784-cf0116e84ecb

# verl-opd is a private env (torch 2.11.0+cu128 / transformers 5.10.4 /
# TransferQueue 0.1.8). It is NOT verl-test-1: that one has Liquid4All/verl
# installed editable and @changyi_yang's and sergei's jobs use it.
export PATH=/home/changyi/miniconda3/envs/verl-opd/bin:$PATH
# PYTHONPATH rather than pip install -e: keeps the upstream clone swappable.
export PYTHONPATH=$MINE/verl-upstream
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
# ray builds AF_UNIX socket paths under RAY_TMPDIR and those cap at 107 bytes.
# The agent workspace path alone is too long once ray appends
# session_<ts>/sockets/plasma_store, so this must stay short. /tmp/ray is owned
# by whoever started ray first on the node, hence the private suffix.
# vllm's cumem_allocator .so is dlopened standalone, without torch's preload
# logic having run, so it cannot find the CUDA 13 libs that pip put under
# site-packages/nvidia/cu13/lib. Without this vllm reports the misleading
# "cumem allocator is not supported on current platform" -- the real error is a
# plain ImportError on libnvrtc.so.13. These are the same cu13 libs torch itself
# uses, so this is a loader-path fix, not a version mix.
export LD_LIBRARY_PATH=/home/changyi/miniconda3/envs/verl-opd/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export RAY_TMPDIR=/tmp/ray-cy
mkdir -p "$RAY_TMPDIR"

cd $MINE/verl-upstream

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-0.6B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-1.7B}
# 6 + 2 = the node's 8. Teacher runs in its own ray placement group.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

train_batch_size=96
ppo_mini_batch_size=96
max_prompt_length=512
max_response_length=1024
ppo_max_token_len_per_gpu=8192
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['$HOME/data/gsm8k/train.parquet']" \
  data.val_files="['$HOME/data/gsm8k/test.parquet']" \
  data.train_batch_size=${train_batch_size} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=verl_opd_smoke \
  trainer.experiment_name=qwen3_0.6b_from_1.7b_fsdp2 \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  distillation.enabled=True \
  distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
  distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.topk=64 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  "$@"
