#!/usr/bin/env bash
set -xeuo pipefail
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate verl-upstream-fresh
unset ROCR_VISIBLE_DEVICES
ray stop --force >/dev/null 2>&1 || true
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
cd /home/changyi/verl_delta_pr

# Run the actual example script, overriding model/data + capping to a 2-step smoke.
export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
export TRAIN_FILE="$HOME/data/gsm8k/train.parquet"
export TEST_FILE="$HOME/data/gsm8k/test.parquet"
export NGPUS_PER_NODE=8

bash verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_sglang_delta_2_6.sh \
    data.train_batch_size=12 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    data.max_response_length=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=1234 \
    trainer.logger=[console] \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps=2
