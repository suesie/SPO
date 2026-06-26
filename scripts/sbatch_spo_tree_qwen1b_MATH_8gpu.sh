#!/usr/bin/env bash
# ============================================================================
# Official SPO-tree (6-6-6) on MATH with DeepSeek-R1-Distill-Qwen-1.5B — 8x H200.
#
# This is the MULTI-GPU (8) variant of scripts/sbatch_spo_tree_qwen1b_MATH.sh.
# Same official upstream config (configs/polIter_qwen1b_spo_tree_MATH.jsonnet,
# fixed 2K context); the only deltas vs the single-GPU run are operational:
#   * 8 GPUs via scripts/launch_server2_spo_multi.sh (deepspeed --num_gpus=8,
#     one vLLM server per rank; NO gpus/gpu_0.jsonnet single-GPU pin).
#   * Checkpoints at 25,50,...,1000 (save_steps=checkpoint_keep_steps=25 in
#     configs/polIter_qwen1b_spo_chain_MATH.jsonnet) — matches the 7B run.
#   * wandb: entity=suesie, project=spo-math, online (explicit, fail-fast auth).
#
# CORRECTNESS: target_train_batch_size=128 is the GLOBAL batch. On 8 GPUs with
# per_device_train_batch_size=2, grad-accum auto-becomes 128/(2x8)=8 (vs 64 on
# 1 GPU) — same global batch, same 4 optimizer steps/iter, same learning target.
# Not bit-identical to the 1-GPU paper run (different sharding/RNG order) but
# statistically equivalent; ~Nx faster episode generation.
#
# Usage:
#   mkdir -p /lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42
#   sbatch scripts/sbatch_spo_tree_qwen1b_MATH_8gpu.sh
# Optional: SEED=43  SPO_ENV=vineppo  WANDB_ENTITY=suesie  WANDB_MODE=offline  sbatch ...
# ============================================================================

#SBATCH --account=h200_mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
# --exclusive on a ~2TB node, so --mem=0 (all node RAM): 8 per-rank vLLM servers
# (swap_space=32G each = 256G) + ZeRO-0 staging + tree/MC episode data.
#SBATCH --mem=0
#SBATCH --exclusive
# 7 days (was 2d): 1000 iters ≈ 3.2 days @ ~4.6 min/iter — the 2-day cap would TIMEOUT
# the run at ~iter 650 (observed on job 1406501). Matches the 7B run's 7-day cap.
#SBATCH --time=7-00:00:00
#SBATCH --job-name=spo-tree-qwen1b-MATH-8gpu-s42
#SBATCH --output=/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42/sbatch_slurm-8gpu-%j.log

set -euo pipefail

SEED="${SEED:-42}"
OUT="/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed${SEED}"
mkdir -p "$OUT"

echo "[sbatch] node=$(hostname)  date=$(date -u +%Y%m%dT%H%M%SZ)  jobid=${SLURM_JOB_ID:-?}  seed=$SEED"
echo "[sbatch] GPUs:"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -8
echo "[sbatch] output dir + free space:"; df -h "$OUT" | sed 's/  */ /g'
echo

# ── caches off $HOME (lustre) ───────────────────────────────────────────────
export HF_HOME=/lustre-storage/datasets/zengh/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # R1-Distill-Qwen-1.5B already cached (3.4G)
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRITON_CACHE_DIR=/lustre-storage/datasets/zengh/triton
export TMPDIR=/lustre-storage/datasets/zengh/tmp
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

# ── DeepSpeed JIT op-build CUDA-mismatch fix + JIT cache on lustre ───────────
export DS_SKIP_CUDA_CHECK=1
export TORCH_EXTENSIONS_DIR=/lustre-storage/datasets/zengh/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

# ── guidance LLM cache: per-job, node-local (avoid shared-SQLite corruption) ─
# guidance/llms/caches/_diskcache.py keys the cache only by LLM class name, so
# every rank of every concurrent job otherwise shares ONE SQLite db under
# ~/.cache/guidance on the network FS. Two multi-node jobs doing that corrupted
# it on 2026-06-12 (1405148 "database disk image is malformed" @ iter 209;
# 1405981 "file is not a database" @ iter 0; both died 06:38). Pin to a per-job
# dir on node-local tmpfs (RAM-backed; --mem=0 --exclusive => ample), falling
# back to the per-job TMPDIR if /dev/shm is unavailable.
GUIDANCE_CACHE_BASE=/dev/shm
{ [ -d "$GUIDANCE_CACHE_BASE" ] && [ -w "$GUIDANCE_CACHE_BASE" ]; } || GUIDANCE_CACHE_BASE="$TMPDIR"
export GUIDANCE_CACHE_DIR="${GUIDANCE_CACHE_BASE}/guidance_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$GUIDANCE_CACHE_DIR"
trap 'rm -rf "$GUIDANCE_CACHE_DIR"' EXIT
echo "[sbatch] GUIDANCE_CACHE_DIR=$GUIDANCE_CACHE_DIR"

source /home/zengh/miniconda3/etc/profile.d/conda.sh
conda activate "${SPO_ENV:-vineppo}"   # treetune stack: torch2.1.2/vllm0.4.0.post1/ds0.14.1

cd /storage/home/zengh/projects/SPO

# ── NCCL 2.18.1 overlay guard (VinePPO runbook §1.3) ────────────────────────
NCCL_LIB="$(python -c 'import os,nvidia.nccl as n; print(os.path.dirname(n.__file__)+"/lib/libnccl.so.2")')"
if ! grep -aqF 'NCCL version 2.18.1' "$NCCL_LIB"; then
    echo "[sbatch] FATAL: unexpected NCCL in $NCCL_LIB (want 2.18.1+cuda12.1)." >&2
    echo "[sbatch] fix: pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.18.1" >&2
    exit 2
fi

echo "[sbatch] env: torch=$(python -c 'import torch;print(torch.__version__)') vllm=$(python -c 'import vllm;print(vllm.__version__)') deepspeed=$(python -c 'import deepspeed;print(deepspeed.__version__)') nccl=2.18.1"
echo "[sbatch] git HEAD: $(git rev-parse HEAD)"
echo "[sbatch] config: configs/polIter_qwen1b_spo_tree_MATH.jsonnet (multi-GPU, no gpu_0.jsonnet)"
echo "[sbatch] data/ contents: $(ls data 2>/dev/null | tr '\n' ' ')"
echo

# ── wandb: sync to suesie / spo-math (explicit, not login-default) ──────────
export WANDB_ENTITY="${WANDB_ENTITY:-suesie}"
export WANDB_PROJECT="${WANDB_PROJECT:-spo-math}"
export WANDB_MODE="${WANDB_MODE:-online}"
export APP_EXPERIMENT_NAME="${APP_EXPERIMENT_NAME:-qwen1.5b-spo-tree-666-s${SEED}}"
# Fail fast if online but unauthenticated (would silently NOT sync to suesie).
WANDB_MODE_LOWER="$(printf '%s' "${WANDB_MODE:-}" | tr '[:upper:]' '[:lower:]')"
if [ "$WANDB_MODE_LOWER" != "disabled" ] && [ "$WANDB_MODE_LOWER" != "offline" ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q 'api.wandb.ai' "$HOME/.netrc" 2>/dev/null; then
        echo "[sbatch] FATAL: WANDB_MODE=$WANDB_MODE but no WANDB_API_KEY / ~/.netrc — would not sync to '$WANDB_ENTITY'." >&2
        echo "[sbatch] fix: run 'wandb login' as the suesie account, or export WANDB_API_KEY." >&2
        exit 5
    fi
fi
echo "[sbatch] wandb: entity=$WANDB_ENTITY project=$WANDB_PROJECT mode=$WANDB_MODE name=$APP_EXPERIMENT_NAME"
echo

# Multi-GPU launch (all 8 GPUs), qwen1b configs. Keeps SPO's own knobs (no injection).
APP_SEED="$SEED" APP_DIRECTORY="$OUT" NUM_GPUS=8 MODEL_FAMILY=qwen1b \
    bash scripts/launch_server2_spo_multi.sh spo_tree
