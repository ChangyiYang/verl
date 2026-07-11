#!/usr/bin/env bash
# Disaggregated Megatron actor + sglang rollout, small smoke for the delta_sharded_mcore weight-sync path.
#   WMODE=nccl              -> full NCCL broadcast baseline
#   WMODE=delta_sharded_mcore -> option-2 sharded delta (design doc)
set -xeuo pipefail
WMODE="${WMODE:?set WMODE=nccl|delta_sharded_mcore}"
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate delta-mega2
unset ROCR_VISIBLE_DEVICES
ray stop --force >/dev/null 2>&1 || true
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTHONPATH=/home/changyi/verl_delta_pr:${PYTHONPATH:-}
# TE 2.16 links torch's bundled cudnn/nccl -- make them findable at runtime
_SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=${_SP}/nvidia/cudnn/lib:${_SP}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
cd /home/changyi/verl_delta_pr

n_gpus_rollout=4
n_gpus_training=4
train_tp=2
gen_tp=2

if [ "$WMODE" = "delta_sharded_mcore" ]; then
  DELTA_ARGS=(
    actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded_mcore
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded_mcore.encoding=indices
  )
else
  DELTA_ARGS=( actor_rollout_ref.rollout.checkpoint_engine.backend=nccl )
fi

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    --config-name='one_step_off_ppo_megatron_trainer.yaml' \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt data.truncation=left \
    data.max_prompt_length=512 data.max_response_length=256 \
    data.train_batch_size=8 data.shuffle=False \
    actor_rollout_ref.rollout.n=4 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=1234 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    "${DELTA_ARGS[@]}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.model.use_remove_padding=True \
    reward.reward_manager.name=naive \
    trainer.logger=[console] \
    trainer.project_name=disagg-mega-delta trainer.experiment_name=${WMODE} \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps=${STEPS:-3} \
    trainer.resume_mode=disable \
    trainer.nnodes=1 trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.nnodes=1 rollout.n_gpus_per_node=${n_gpus_rollout}
