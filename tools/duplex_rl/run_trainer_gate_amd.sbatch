#!/usr/bin/env bash
#SBATCH --job-name=duplex-trainer-gate
#SBATCH --partition=gpuworker
#SBATCH --account=posttraining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=361G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/logs/%x-%j.out

set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-/home/changyi.yang@liquid.ai/duplex-rl-wt}
TRAIN_FILE=${TRAIN_FILE:-/home/changyi.yang@liquid.ai/duplex_rl_data/trainer_gate.parquet}
AMD_IMAGE=${AMD_IMAGE:-/home/teams/posttraining/sqsh/liquidai_liquid_rl_dev0.7.sqsh}
DEPS_DIR=${DEPS_DIR:-/home/changyi.yang@liquid.ai/.duplex_probe_min}
CONTAINER_NAME="duplex-trainer-${SLURM_JOB_ID}"

srun --nodes=1 --ntasks=1 \
    --container-image="${AMD_IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts=/home:/home,/home/shared_tw:/home/shared_tw \
    --container-workdir="${PROJECT_ROOT}" \
    --container-entrypoint \
    --container-remap-root \
    --kill-on-bad-exit=1 \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",TRAIN_FILE="${TRAIN_FILE}",DEPS_DIR="${DEPS_DIR}" \
    bash -lc '
        set -euo pipefail
        export PATH=/opt/venv/bin:${PATH}
        export PYTHONPATH=${DEPS_DIR}:${PROJECT_ROOT}
        export PYTHONNOUSERSITE=1
        export PYTHONUNBUFFERED=1
        export HYDRA_FULL_ERROR=1
        export TOKENIZERS_PARALLELISM=false
        export HSA_ENABLE_IPC_MODE_LEGACY=1
        export HSA_ENABLE_COREDUMP=0
        export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
        export RAY_DEDUP_LOGS=0
        # Ray AMD accelerator manager treats simultaneous ROCR/HIP masks as
        # ambiguous; Slurm/pyxis may inject both for a one-GPU allocation.
        unset ROCR_VISIBLE_DEVICES
        /opt/venv/bin/python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
        PYTHON_BIN=/opt/venv/bin/python TRAIN_FILE=${TRAIN_FILE} TEST_FILE=${TRAIN_FILE} \
            bash examples/duplex_rl/run_minicpmo45_grpo_fsdp.sh
    '
