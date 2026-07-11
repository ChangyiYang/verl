#!/usr/bin/env bash
# Real 2-node disaggregated test: trainer on node0, rollout on node1 (cross-node weight sync).
# Launch with:  srun --nodes=2 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=32 --time=00:40:00 bash _runs/run_2node.sh
#   WMODE=nccl | delta | delta_sharded
set -xeuo pipefail
WMODE="${WMODE:?set WMODE=nccl|delta|delta_sharded}"
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate verl-upstream-fresh
unset ROCR_VISIBLE_DEVICES
export WANDB_MODE=${WANDB_MODE:-disabled} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
CODE_DIR="${CODE_DIR:-/home/changyi/verl_delta_pr}"
export PYTHONPATH=${CODE_DIR}:${PYTHONPATH:-}
# Some cluster nodes have a stale /tmp/ray owned by another user -- keep ray state user-owned.
export RAY_TMPDIR=/tmp/ray_${USER}
mkdir -p "$RAY_TMPDIR"
cd "$CODE_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
FSDP_SIZE="${FSDP_SIZE:-2}"
SP="${SP:-2}"
GEN_TP="${GEN_TP:-2}"
OFFLOAD="${OFFLOAD:-False}"
NGPU="${NGPU:-4}"
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments -- it breaks sglang's TorchMemorySaver.
NODEID="${SLURM_NODEID:-0}"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${NODES[0]}"
HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"
PORT=6379
FLAGDIR="$HOME/.ray_flags/${SLURM_JOB_ID}"; mkdir -p "$FLAGDIR"

if [ "$NODEID" != "0" ]; then
  # ---- worker node: join the head, stay up until the driver signals done ----
  while [ ! -f "$FLAGDIR/head_ready" ]; do sleep 2; done
  ray stop --force >/dev/null 2>&1 || true
  ray start --address="$HEAD_IP:$PORT" --num-gpus=$NGPU --disable-usage-stats
  while [ ! -f "$FLAGDIR/done" ]; do sleep 5; done
  ray stop --force >/dev/null 2>&1 || true
  exit 0
fi

# ---- head node (node0): start ray head, wait for the worker, run the driver ----
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address="$HEAD_IP" --port=$PORT --num-gpus=$NGPU --disable-usage-stats
touch "$FLAGDIR/head_ready"
# wait until the cluster reports 2 alive nodes (8 GPUs)
for i in $(seq 1 60); do
  n=$(ray status 2>/dev/null | grep -cE "^ *1 node_" || true)
  ng=$(python -c "import ray;ray.init(address='auto');print(int(ray.cluster_resources().get('GPU',0)));ray.shutdown()" 2>/dev/null || echo 0)
  [ "${ng:-0}" -ge $((NGPU*2)) ] && break
  sleep 3
done
export RAY_ADDRESS="$HEAD_IP:$PORT"

# Delta modes register verl's in-process sparse apply in the sglang server (stock hook).
LOADER='["verl.checkpoint_engine.delta_sync.sglang_loader.apply_delta"]'
if [ "$WMODE" = "delta" ]; then
  DELTA_ARGS=( actor_rollout_ref.rollout.checkpoint_engine.backend=delta
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta.encoding=indices
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.custom_weight_loader=${LOADER}" )
elif [ "$WMODE" = "delta_sharded" ]; then
  DELTA_ARGS=( actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=indices
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.custom_weight_loader=${LOADER}" )
else
  DELTA_ARGS=( actor_rollout_ref.rollout.checkpoint_engine.backend=nccl )
fi

python3 -m verl.experimental.one_step_off_policy.main_ppo \
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
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${MEM_UTIL:-0.7} \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=1234 \
    actor_rollout_ref.rollout.enforce_eager=True \
    "${DELTA_ARGS[@]}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${BUCKET_MB:-2048} \
    actor_rollout_ref.actor.strategy=fsdp2 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    reward.reward_manager.name=naive \
    "trainer.logger=[${LOGGER:-console}]" \
    trainer.project_name=twonode-delta trainer.experiment_name=${EXP_NAME:-$WMODE} \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps=${STEPS:-3} \
    trainer.resume_mode=disable \
    trainer.nnodes=1 trainer.n_gpus_per_node=$NGPU \
    rollout.nnodes=1 rollout.n_gpus_per_node=$NGPU || true

touch "$FLAGDIR/done"
ray stop --force >/dev/null 2>&1 || true
