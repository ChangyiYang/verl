#!/usr/bin/env bash
set -xeuo pipefail
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate verl-upstream-fresh
unset ROCR_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTHONPATH=/home/changyi/verl_pr_v1
python /home/changyi/verl_delta_pr/_runs/traj_roundtrip_test.py
