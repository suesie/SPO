#!/usr/bin/env bash
# ============================================================================
# SPO-tree (6-6-6) on MATH with deepseekmath-7b-sft-MATH-v2 — 8x H200.
#
# Apples-to-apples target: VoI run `grpo_voi_v041_p4_mb8_T08unifW5wh_deepseekmath_math_v2_seed42`
#   (grpocredit/scripts/sbatch_launch_p4_mb8_T08unifW5wh_math_v2_seed42.sh):
#   same account/qos, nodes=1, cpus-per-task=96, gres=gpu:h200:8, exclusive.
# That VoI run evaluates on `math_val.parquet` (== SPO `data/math/validation`,
# verified 500/500 problem overlap), so this sbatch relies on the local edit
# in `configs/sft_deepseekmath_for_MATH_eval.jsonnet` that flips the active
# in-loop pipeline to `math_validation_inference_pipeline` (see COMPARE_spo_tree_vs_grpocredit_voi.md
# "Local edits" callout #3). GPU(8)/CPU(96) compute identical to VoI; eval data
# identical; train data identical (math_train_vineppo, verified).
#
# CONFIG: configs/polIter_deepseekSft2_spo_tree_MATH.jsonnet  (after the import fix:
#   now imports the shipped SPO-chain base, so use_prob_mask=true). Base model
#   realtreetune/deepseekmath-7b-sft-MATH-v2; 6-6-6 tree, M=66; resp 1024; ZeRO-2.
#   SPO's own published knobs are kept (lr 1e-6, KL 1e-4, T 0.6) — NOT matched to
#   VoI, since each method's optimum differs.
#
# WALLTIME: --time=7-00:00:00 (cluster max). SPO-tree per-iter is analytically
#   ~3-5 min; 1000 iters → 50-83 hours = 2-3.5 days. VinePPO hit TIMEOUT at
#   326/1000 under a 3-day cap on the same treetune pipeline, so we use the full
#   7-day budget defensively. Walltime is a ceiling, not a compute cost.
#
# Usage:
#   mkdir -p /lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed42
#   sbatch scripts/sbatch_spo_tree_deepseek7b_MATH.sh
# Optional: SEED=43  SPO_ENV=vineppo  WANDB_PROJECT=spo-math  WANDB_MODE=offline  sbatch ...
# ============================================================================

#SBATCH --account=h200_mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
# VoI used --mem=512G, but treetune 7B (8 per-rank vLLM servers x swap_space=32G =
# 256G + ZeRO-2 staging + tree/MC episode data) OOM'd VinePPO at 512G. Job is
# --exclusive on a ~2TB node, so --mem=0 (all node RAM) keeps GPU(8)/CPU(96) — the
# compute resources — identical to VoI; host RAM is a floor, not a compute lever.
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --job-name=spo-tree-deepseek7b-MATH-s42
#SBATCH --output=/lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed42/sbatch_slurm-%j.log

set -euo pipefail

SEED="${SEED:-42}"
OUT="/lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed${SEED}"
mkdir -p "$OUT"

echo "[sbatch] node=$(hostname)  date=$(date -u +%Y%m%dT%H%M%SZ)  jobid=${SLURM_JOB_ID:-?}  seed=$SEED"
echo "[sbatch] GPUs:"; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -8
echo "[sbatch] output dir + free space:"; df -h "$OUT" | sed 's/  */ /g'
echo

# ── caches off $HOME (lustre) ───────────────────────────────────────────────
export HF_HOME=/lustre-storage/datasets/zengh/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # deepseekmath-7b already cached (13G)
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
echo "[sbatch] config: configs/polIter_deepseekSft2_spo_tree_MATH.jsonnet (import-fixed -> spo_chain)"
echo "[sbatch] data/ contents: $(ls data 2>/dev/null | tr '\n' ' ')"
echo

export WANDB_PROJECT="${WANDB_PROJECT:-spo-math}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Multi-GPU launch (all 8 GPUs). Keeps SPO's own hyperparameters (no injection).
APP_SEED="$SEED" APP_DIRECTORY="$OUT" NUM_GPUS=8 \
    bash scripts/launch_server2_spo_multi.sh spo_tree
