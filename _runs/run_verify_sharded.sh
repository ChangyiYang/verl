#!/usr/bin/env bash
set -xeuo pipefail
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate verl-upstream-fresh
unset ROCR_VISIBLE_DEVICES
cd /home/changyi/verl_delta_pr
PYTHONPATH=/home/changyi/verl_delta_pr torchrun --nproc_per_node=4 --master_port=29585 _runs/verify_sharded_delta.py
