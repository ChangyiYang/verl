#!/bin/bash
# Docker entrypoint for sglang-verl-rocm images.
# Automatically installs editable sglang and verl from mounted PROJECT_ROOT
# so slurm scripts no longer need to call set_verl_sglang.sh manually.

set -euo pipefail

# The ROCm 7.0 MI300X base image defaults can make RCCL probe the IB net
# provider first. On the Liquid TensorWave AMD nodes, libibverbs reports a
# bnxt_re kernel ABI mismatch, and RCCL aborts with "network IB not found"
# even for single-node 8-GPU collectives. Prefer socket bootstrap/net by
# default; set LIQUID_ROCM_FORCE_SOCKET_NCCL=0 to opt out for clusters with
# working RDMA.
if [[ "${LIQUID_ROCM_FORCE_SOCKET_NCCL:-1}" == "1" ]]; then
  export NCCL_NET="${NCCL_NET:-Socket}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export RCCL_IB_DISABLE="${RCCL_IB_DISABLE:-1}"
  unset NCCL_MIN_NCHANNELS
fi

# Derive PROJECT_ROOT if not explicitly set.
# The container mounts /home, so fall back to $HOME/liquid_rl.
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${HOME}/liquid_rl"
  echo "[entrypoint] PROJECT_ROOT not set, defaulting to ${PROJECT_ROOT}"
fi
export PROJECT_ROOT

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  if [[ -d "${PROJECT_ROOT}/sglang/python" ]]; then
    echo "[entrypoint] pip install -e ${PROJECT_ROOT}/sglang/python --no-deps"
    pip install -e "${PROJECT_ROOT}/sglang/python" --no-deps -q
  else
    echo "[entrypoint] WARNING: ${PROJECT_ROOT}/sglang/python not found, skipping sglang install"
  fi
  if [[ -d "${PROJECT_ROOT}/verl" ]]; then
    echo "[entrypoint] pip install -e ${PROJECT_ROOT}/verl --no-deps"
    pip install -e "${PROJECT_ROOT}/verl" --no-deps -q
  else
    echo "[entrypoint] WARNING: ${PROJECT_ROOT}/verl not found, skipping verl install"
  fi
else
  echo "[entrypoint] PROJECT_ROOT not set, skipping editable installs"
fi

exec "$@"
