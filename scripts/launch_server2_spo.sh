#!/usr/bin/env bash
# Launch SPO training (SINGLE GPU) for the qwen1b MATH configs.
#
# Faithful to the SPO README recipe:
#   deepspeed --include localhost:0 src/treetune/main.py \
#       --configs "configs/polIter_qwen1b_<method>_MATH.jsonnet,configs/gpus/gpu_0.jsonnet" \
#       run_iteration_loop
# ...plus the APP_SEED / APP_DIRECTORY / WANDB_PROJECT / logging conveniences used
# by the VinePPO-grpo launchers, so sbatch_* can call it the same way.
#
# Usage:
#   bash scripts/launch_server2_spo.sh spo_tree MATH      # headline: SPO-tree (6-6-6)
#   bash scripts/launch_server2_spo.sh spo_chain MATH
#   bash scripts/launch_server2_spo.sh grpo MATH          # GRPO baseline
#
# Optional env-vars:
#   GPU_ID=0             local GPU index (default 0; selects configs/gpus/gpu_${GPU_ID}.jsonnet)
#   APP_SEED=42          random seed (default 42)
#   APP_DIRECTORY=...    output dir (default: experiments/qwen1b_<method>_<task>)
#   WANDB_PROJECT=...    wandb project (default: spo-math; entity comes from the env's wandb login)
#   EXTRA_CONFIGS=...    extra comma-separated jsonnet configs to layer last
#   MASTER_PORT=...      deepspeed master port (default: a free port)
#
# NOTE: SPO-tree was designed for / published on a SINGLE 80GB GPU. This wrapper
# intentionally pins ONE GPU (vLLM episode-generation and training share it;
# the config sets wait_until_memory_release=true so vLLM frees VRAM before the
# optimizer step). Do not pass --num_gpus here.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <method: spo_tree|spo_chain|grpo> <task: MATH>" >&2
    exit 64
fi

METHOD="$1"
TASK="$2"
case "$METHOD" in spo_tree|spo_chain|grpo) ;; *) echo "bad method: $METHOD" >&2; exit 64;; esac
case "$TASK" in MATH) ;; *) echo "bad task: $TASK (this wrapper targets qwen1b MATH)" >&2; exit 64;; esac

cd "$(dirname "$0")/.."

CONFIG="configs/polIter_qwen1b_${METHOD}_${TASK}.jsonnet"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 66; }

GPU_ID="${GPU_ID:-0}"
GPUCFG="configs/gpus/gpu_${GPU_ID}.jsonnet"
[ -f "$GPUCFG" ] || { echo "missing gpu config: $GPUCFG (have configs/gpus/gpu_{0,1,2,3}.jsonnet)" >&2; exit 66; }

EXTRA="${EXTRA_CONFIGS:-}"
[ -n "$EXTRA" ] && CONFIGSTR="$CONFIG,$GPUCFG,$EXTRA" || CONFIGSTR="$CONFIG,$GPUCFG"

export APP_SEED="${APP_SEED:-42}"
export APP_DIRECTORY="${APP_DIRECTORY:-experiments/qwen1b_${METHOD}_${TASK}}"
export APP_MINIMIZE_STORED_FILES="${APP_MINIMIZE_STORED_FILES:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-spo-math}"
export MASTER_PORT="${MASTER_PORT:-$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"

mkdir -p "$APP_DIRECTORY"
LOG="${APP_DIRECTORY}/run_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[spo] method=$METHOD task=$TASK GPU_ID=$GPU_ID seed=$APP_SEED"
echo "[spo] APP_DIRECTORY=$APP_DIRECTORY"
echo "[spo] CONFIGSTR=$CONFIGSTR"
echo "[spo] MASTER_PORT=$MASTER_PORT  log -> $LOG"

deepspeed --master_port "$MASTER_PORT" --include "localhost:${GPU_ID}" \
    src/treetune/main.py --configs "$CONFIGSTR" \
    run_iteration_loop 2>&1 | tee "$LOG"
