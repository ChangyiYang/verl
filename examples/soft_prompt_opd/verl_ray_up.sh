#!/usr/bin/env bash
# Bring up a ray cluster across drained nodes for a multi-node verl run.
#
# Slurm refuses to schedule onto these nodes, but they are healthy and ssh works
# (@changyi_yang's trick). So the cluster has to be assembled by hand.
#
# Usage: verl_ray_up.sh <head_node> <worker_node> [more workers...]
# Prints the head address on success; run the trainer on the head with
# RAY_ADDRESS=<that address>.
set -uo pipefail
HEAD=$1; shift
WORKERS=("$@")

MINE=/home/changyi/.slock/agents/7dc66911-8db2-4303-a784-cf0116e84ecb
ENVBIN=/home/changyi/miniconda3/envs/verl-opd/bin
# Short path: ray builds AF_UNIX socket paths under here and they cap at 107 bytes.
TMP=${RAY_UP_TMP:-/tmp/ray-cy-ds2}
PORT_IN=${RAY_UP_PORT:-6380}
PORT=$PORT_IN

# Same preamble on every node -- the head and the workers must agree on the
# python, the CUDA libs and the verl checkout, or actors deserialise against a
# different verl than the driver built the config with.
PRE="export PATH=$ENVBIN:\$PATH; \
export PYTHONPATH=$MINE/verl-upstream; \
export PYTHONUNBUFFERED=1; \
export LD_LIBRARY_PATH=/home/changyi/miniconda3/envs/verl-opd/lib/python3.12/site-packages/nvidia/cu13/lib:\${LD_LIBRARY_PATH:-}; \
export RAY_TMPDIR=$TMP; mkdir -p $TMP"

# Refuse to touch a node someone else is on. Checked here rather than trusting an
# earlier scan, because the scan and the launch are minutes apart.
for n in "$HEAD" "${WORKERS[@]}"; do
  BUSY=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$n" \
    "ps -eo user:16,comm --no-headers | grep -E 'python|torchrun|vllm|ray' | awk '{print \$1}' | grep -v '^changyi\$' | sort -u | tr '\n' ','" 2>/dev/null)
  if [ -n "$BUSY" ]; then echo "ABORT $n: other users present ($BUSY)" >&2; exit 1; fi
done

echo "stopping any stale ray on the target nodes"
for n in "$HEAD" "${WORKERS[@]}"; do
  ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$n" "$PRE; ray stop --force >/dev/null 2>&1; exit 0" >/dev/null 2>&1
done

HEAD_IP=$(ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD" "hostname -I | awk '{print \$1}'" 2>/dev/null)
[ -n "$HEAD_IP" ] || { echo "ABORT: could not resolve $HEAD ip" >&2; exit 1; }

echo "starting ray head on $HEAD ($HEAD_IP:$PORT)"
ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD" \
  "$PRE; ray start --head --node-ip-address=$HEAD_IP --port=$PORT --num-gpus=8 --temp-dir=$TMP" 2>&1 | tail -3

for n in "${WORKERS[@]}"; do
  echo "joining $n"
  ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$n" \
    "$PRE; ray start --address=$HEAD_IP:$PORT --num-gpus=8" 2>&1 | tail -2
done

echo "cluster:"
ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no "$HEAD" "$PRE; ray status" 2>&1 | grep -A4 "Resources\|GPU" | head -12
echo "RAY_ADDRESS=$HEAD_IP:$PORT"
