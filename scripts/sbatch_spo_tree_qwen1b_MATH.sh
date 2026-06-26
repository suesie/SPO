#!/usr/bin/env bash
# ============================================================================
# Reproduce SPO-tree (6-6-6) on MATH, SINGLE GPU.
#
# WHAT THIS RUNS (provenance):
#   * Config: configs/polIter_qwen1b_spo_tree_MATH.jsonnet (unmodified upstream)
#       -> imports polIter_qwen1b_spo_chain_MATH.jsonnet. Resolved knobs:
#         base = deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#         tree branch factors 6-6-6 (max_depth 3), M=600, 4 MC rollouts/node,
#         512 episodes/iter, 8 traj rollouts/sample, 1000 iters,
#         rollout T=0.6 / max_tokens=4096, model_context_size=2048 (2K),
#         use_prob_mask=true, lam=1, init_kl_coef=1e-4 (KL loss),
#         DeepSpeed ZeRO-0 (no offload; 1.5B fits one 80GB+ GPU),
#         target_train_batch_size=128, per_device_train_batch_size=2,
#         save_steps=5, checkpoint_keep_steps=10, 1 epoch/iter.
#   * Entrypoint: README `run_iteration_loop` via scripts/launch_server2_spo.sh
#       (single GPU, configs/gpus/gpu_0.jsonnet; injects NO hyperparameters).
#
# Cluster env mirrors VinePPO-grpo/scripts/sbatch_vineppo_deepseekSft2_MATH.sh
# (lustre caches, offline HF, DeepSpeed CUDA-skip, NCCL 2.18.1 guard).
# SPO-tree is a SINGLE-GPU method (paper: 1x A100 80GB), so this asks for 1 H200.
#
# Usage:
#   mkdir -p /lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42
#   sbatch scripts/sbatch_spo_tree_qwen1b_MATH.sh
# Optional env overrides at submit time:
#   SEED=43  SPO_ENV=spo  WANDB_PROJECT=spo-math  WANDB_MODE=offline  sbatch ...
# ============================================================================

#SBATCH --account=h200_mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200:1
# 1.5B + one vLLM server (swap_space=32G) + ZeRO-0 staging + episode data.
# 256G is comfortable; this job is NOT --exclusive so the other 7 GPUs stay free.
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=spo-tree-qwen1b-MATH-s42
#SBATCH --output=/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42/sbatch_slurm-%j.log

set -euo pipefail

SEED="${SEED:-42}"
OUT="/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed${SEED}"
mkdir -p "$OUT"

echo "[sbatch] node=$(hostname)  date=$(date -u +%Y%m%dT%H%M%SZ)  jobid=${SLURM_JOB_ID:-?}  seed=$SEED"
echo "[sbatch] GPU:"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1
echo "[sbatch] output dir + free space:"; df -h "$OUT" | sed 's/  */ /g'
echo

# ── caches off $HOME (lustre) ───────────────────────────────────────────────
export HF_HOME=/lustre-storage/datasets/zengh/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # model pre-cached -> avoid proxy flakiness
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRITON_CACHE_DIR=/lustre-storage/datasets/zengh/triton
export TMPDIR=/lustre-storage/datasets/zengh/tmp
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

# ── DeepSpeed JIT op-build CUDA-mismatch fix + JIT cache on lustre ───────────
# Compute nodes ship nvcc 12.8; torch 2.1.2 is built against CUDA 12.1. The
# minor mismatch is op-compatible for fused_adam etc.; skip the hard check.
export DS_SKIP_CUDA_CHECK=1
export TORCH_EXTENSIONS_DIR=/lustre-storage/datasets/zengh/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

# ── guidance LLM cache: per-job, node-local (avoid shared-SQLite corruption) ─
# guidance/llms/caches/_diskcache.py keys the cache only by LLM class name, so
# every rank of every concurrent job otherwise shares ONE SQLite db under
# ~/.cache/guidance on the network FS. Two multi-node jobs doing that corrupted
# it on 2026-06-12 (1405148 "database disk image is malformed" @ iter 209;
# 1405981 "file is not a database" @ iter 0; both died 06:38). Pin to a per-job
# dir on node-local tmpfs (RAM-backed), falling back to the per-job TMPDIR if
# /dev/shm is unavailable. (Single-GPU run has 1 rank, but this still isolates
# it from any concurrent multi-GPU job sharing the node's cache.)
GUIDANCE_CACHE_BASE=/dev/shm
{ [ -d "$GUIDANCE_CACHE_BASE" ] && [ -w "$GUIDANCE_CACHE_BASE" ]; } || GUIDANCE_CACHE_BASE="$TMPDIR"
export GUIDANCE_CACHE_DIR="${GUIDANCE_CACHE_BASE}/guidance_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$GUIDANCE_CACHE_DIR"
trap 'rm -rf "$GUIDANCE_CACHE_DIR"' EXIT
echo "[sbatch] GUIDANCE_CACHE_DIR=$GUIDANCE_CACHE_DIR"

source /home/zengh/miniconda3/etc/profile.d/conda.sh
conda activate "${SPO_ENV:-spo}"

cd /storage/home/zengh/projects/SPO

# ── NCCL overlay guard (VinePPO runbook §1.3): torch 2.1.2 bundles NCCL
#    2.18.1+cuda12.1. A CUDA-13 overlay crashes at the first collective. Fail
#    fast. grep the .so directly (binary-safe -a, fixed -F); never pipe through
#    `strings | grep -q` under pipefail (SIGPIPE -> false FATAL). ──
NCCL_LIB="$(python -c 'import os,nvidia.nccl as n; print(os.path.dirname(n.__file__)+"/lib/libnccl.so.2")')"
if ! grep -aqF 'NCCL version 2.18.1' "$NCCL_LIB"; then
    echo "[sbatch] FATAL: unexpected NCCL in $NCCL_LIB (want 2.18.1+cuda12.1)." >&2
    echo "[sbatch] fix: pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.18.1" >&2
    exit 2
fi

echo "[sbatch] env: torch=$(python -c 'import torch;print(torch.__version__)') vllm=$(python -c 'import vllm;print(vllm.__version__)') deepspeed=$(python -c 'import deepspeed;print(deepspeed.__version__)') nccl=2.18.1"
echo "[sbatch] git HEAD: $(git rev-parse HEAD)"
echo "[sbatch] config: configs/polIter_qwen1b_spo_tree_MATH.jsonnet"
echo "[sbatch] data/ contents: $(ls data 2>/dev/null | tr '\n' ' ')"
echo

# wandb authed in the env (entity=suesie). Override project/mode at submit.
export WANDB_PROJECT="${WANDB_PROJECT:-spo-math}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Single-GPU launch via the thin wrapper (NO hyperparameter injection).
APP_SEED="$SEED" APP_DIRECTORY="$OUT" GPU_ID=0 \
    bash scripts/launch_server2_spo.sh spo_tree MATH
