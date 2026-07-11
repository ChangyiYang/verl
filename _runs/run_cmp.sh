#!/usr/bin/env bash
set -xeuo pipefail
WMODE="${WMODE:?set WMODE=full|delta}"
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate verl-upstream-fresh
unset ROCR_VISIBLE_DEVICES
ray stop --force >/dev/null 2>&1 || true
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
cd /home/changyi/verl_delta_pr
DELTA_DIR=/home/changyi/verl_delta_pr/_runs/cmp_${WMODE}_disk
rm -rf "$DELTA_DIR"/* 2>/dev/null || true; mkdir -p "$DELTA_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=256 \
    data.filter_overlong_prompts=True data.truncation=error \
    data.shuffle=False data.seed=1 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=1234 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.checkpoint_engine.weight_mode=${WMODE} \
    actor_rollout_ref.rollout.checkpoint_engine.weight_transport=disk \
    actor_rollout_ref.rollout.checkpoint_engine.delta_disk_dir="$DELTA_DIR" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 trainer.logger=[console] \
    trainer.project_name=cmp-delta-vs-full trainer.experiment_name=${WMODE} \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.save_freq=-1 trainer.test_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps=5
