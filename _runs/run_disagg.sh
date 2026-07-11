#!/usr/bin/env bash
set -xeuo pipefail
WMODE="${WMODE:?set WMODE=full|delta|delta_sharded}"
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate verl-upstream-fresh
unset ROCR_VISIBLE_DEVICES
ray stop --force >/dev/null 2>&1 || true
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
cd /home/changyi/verl_delta_pr

NUM_GPUS=8
n_gpus_rollout=4
n_gpus_training=4

# delta on the disaggregated wire: select the delta checkpoint engine (NCCL transport,
# broadcasts only the changed positions+values over the engine collective group).
DELTA_ARGS=()
if [ "$WMODE" = "delta" ]; then
  DELTA_ARGS=(
    actor_rollout_ref.rollout.checkpoint_engine.backend=delta
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta.encoding=indices
  )
elif [ "$WMODE" = "delta_sharded" ]; then
  DELTA_ARGS=(
    actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=indices
  )
elif [ "$WMODE" = "nccl" ]; then
  # full-weight broadcast over the SAME NCCL collective wire that delta/delta_sharded subclass.
  # This is the apples-to-apples full-update baseline for the disaggregated path (the `naive`
  # colocated engine does not work under DetachActorWorker).
  DELTA_ARGS=( actor_rollout_ref.rollout.checkpoint_engine.backend=nccl )
else
  DELTA_ARGS=( actor_rollout_ref.rollout.checkpoint_engine.backend=naive )
fi

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt data.truncation=left \
    data.max_prompt_length=512 data.max_response_length=256 \
    data.train_batch_size=8 \
    data.shuffle=False \
    actor_rollout_ref.rollout.n=4 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=1234 \
    actor_rollout_ref.rollout.enforce_eager=True \
    "${DELTA_ARGS[@]}" \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    reward.reward_manager.name=naive \
    trainer.logger=[console] \
    trainer.project_name=disagg-delta-vs-full trainer.experiment_name=${WMODE} \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps=${STEPS:-4} \
    trainer.resume_mode=disable \
    trainer.nnodes=1 trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.nnodes=1 rollout.n_gpus_per_node=${n_gpus_rollout}
