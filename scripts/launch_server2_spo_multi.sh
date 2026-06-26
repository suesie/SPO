#!/usr/bin/env bash
# Multi-GPU SPO training for MATH configs — deepseekSft2 (7B) or qwen1b (1.5B) via MODEL_FAMILY.
#
# Mirrors the VinePPO-grpo multi-GPU launch (`deepspeed --num_gpus=N`, per-rank
# vLLM servers). Use this for the 7B model — full-FT 7B under DeepSpeed ZeRO-2
# (no offload) does NOT fit one GPU, so the single-GPU launch_server2_spo.sh
# (`--include localhost:0`, gpu_0.jsonnet) does not apply here.
#
# Usage:
#   bash scripts/launch_server2_spo_multi.sh spo_tree    # SPO-tree (6-6-6) on deepseekmath-7b
#   bash scripts/launch_server2_spo_multi.sh spo_chain
#   bash scripts/launch_server2_spo_multi.sh grpo        # GRPO baseline (same framework)
#
# Optional env-vars:
#   NUM_GPUS=8           GPUs for deepspeed (default: all visible)
#   MODEL_FAMILY=...     deepseekSft2 (7B, default) | qwen1b (1.5B)
#   APP_SEED=42
#   APP_DIRECTORY=...    output dir (default: experiments/<family>_<method>_MATH)
#   WANDB_PROJECT=...    default: spo-math (entity from the env's wandb login)
#   EXTRA_CONFIGS=...    extra comma-separated jsonnet configs to layer last
#   MASTER_PORT=...      deepspeed master port (default: a free port)
#
# NOTE: no configs/gpus/gpu_*.jsonnet here — those pin a single GPU / gpu_offset.
# Multi-GPU treetune spawns one vLLM server per rank (swap_space=32G each).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <method: spo_tree|spo_chain|grpo>" >&2
    exit 64
fi
METHOD="$1"
case "$METHOD" in spo_tree|spo_chain|grpo) ;; *) echo "bad method: $METHOD" >&2; exit 64;; esac

cd "$(dirname "$0")/.."

FAMILY="${MODEL_FAMILY:-deepseekSft2}"   # deepseekSft2 (7B) | qwen1b (1.5B)
CONFIG="configs/polIter_${FAMILY}_${METHOD}_MATH.jsonnet"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 66; }
EXTRA="${EXTRA_CONFIGS:-}"
[ -n "$EXTRA" ] && CONFIGSTR="$CONFIG,$EXTRA" || CONFIGSTR="$CONFIG"

export APP_SEED="${APP_SEED:-42}"
export APP_DIRECTORY="${APP_DIRECTORY:-experiments/${FAMILY}_${METHOD}_MATH}"
export APP_MINIMIZE_STORED_FILES="${APP_MINIMIZE_STORED_FILES:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-spo-math}"
export MASTER_PORT="${MASTER_PORT:-$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

mkdir -p "$APP_DIRECTORY"
LOG="${APP_DIRECTORY}/run_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[spo-multi] family=$FAMILY method=$METHOD NUM_GPUS=$NUM_GPUS seed=$APP_SEED"
echo "[spo-multi] APP_DIRECTORY=$APP_DIRECTORY"
echo "[spo-multi] CONFIGSTR=$CONFIGSTR  MASTER_PORT=$MASTER_PORT"
echo "[spo-multi] log -> $LOG"

deepspeed --no_local_rank --master_port "$MASTER_PORT" --num_gpus="$NUM_GPUS" \
    src/treetune/main.py --configs "$CONFIGSTR" \
    run_iteration_loop 2>&1 | tee "$LOG"
