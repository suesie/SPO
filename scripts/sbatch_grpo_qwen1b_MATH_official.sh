#!/usr/bin/env bash
# ============================================================================
# Reproduce the OFFICIAL GRPO baseline on MATH, SINGLE GPU.
# DeepSeek-R1-Distill-Qwen-1.5B  ->  MATH (train)  ->  MATH500 (eval).
#
# "100% identical to official" provenance (vs upstream AIFrameResearch/SPO@1e64f0c):
#   * Committed code is upstream-clean: HEAD == origin/master == upstream/master,
#     and configs/polIter_qwen1b_grpo_MATH.jsonnet itself is UNMODIFIED.
#   * The GRPO config imports the shared parent polIter_qwen1b_spo_chain_MATH.jsonnet,
#     which has LOCAL uncommitted edits. Exactly ONE of them changes training:
#       max_question_length 200 (upstream) -> 512 (local).  <-- RESET BELOW to 200
#     via EXTRA_CONFIGS=configs/episode_generators/max_question_length_200.jsonnet
#     (layered last; treetune merges --configs with jsonnet `+`).
#   * The remaining working-tree deltas are training-EQUIVALENT and intentionally kept:
#       - model path: content-identical local snapshot (HF refs/main == ad9f0ae0),
#         required for offline (HF_HUB_OFFLINE=1); the repo-ID crashes at iter-0 offline.
#       - on_policy_episode_generator.py shuffle: keep_in_memory=True, SAME seed => SAME
#         permutation (cluster shared-FS cache-race fix only).
#       - _diskcache.py / analyzers __init__: guidance cache dir + inert import.
#       - save_steps/checkpoint_keep_steps = 25/25 (local) vs 5/10 (upstream): checkpoint
#         CADENCE only, no training-dynamics effect. Kept at 25/25 by explicit choice.
#         (iter % 25 == 0 retained => iters 25,50,...,1000; the GRPO headline ckpt is
#          retained since it is a multiple of 25.)
#
# RESOLVED GRPO knobs (config + override):
#   base = deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (local snapshot, content-identical)
#   episode_generator = math_episode_generator_w_group_advantages, adv_method=grpo
#   group size 8 (num_rollouts_per_sample); 512 episodes/iter => 64 questions x 8 rollouts
#   1000 iterations; rollout T=0.6, max_tokens=4096, model_context_size=2048 (2K)
#   use_prob_mask=false; lam=1; gamma=1.0; whiten_advantages=true; cliprange=0.2
#   init_kl_coef=1e-4, KL-as-loss (control_variate), critic-less
#   DeepSpeed ZeRO-0 (no offload; 1.5B fits one 80GB+ GPU)
#   target_train_batch_size=128, per_device_train_batch_size=2, 1 epoch/iter, lr=1e-6
#   max_question_length = 200  (RESET to upstream)
#
# Entrypoint mirrors the README single-GPU recipe via scripts/launch_server2_spo.sh:
#   deepspeed --include localhost:0 src/treetune/main.py --configs
#     "configs/polIter_qwen1b_grpo_MATH.jsonnet,configs/gpus/gpu_0.jsonnet,\
#      configs/episode_generators/max_question_length_200.jsonnet" run_iteration_loop
#
# Usage:
#   mkdir -p /lustre-storage/checkpoints/zengh/spo/grpo_qwen1b_MATH_official_seed42
#   sbatch scripts/sbatch_grpo_qwen1b_MATH_official.sh
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
#SBATCH --job-name=grpo-qwen1b-MATH-official-s42
#SBATCH --output=/lustre-storage/checkpoints/zengh/spo/grpo_qwen1b_MATH_official_seed42/sbatch_slurm-%j.log

set -euo pipefail

SEED="${SEED:-42}"
OUT="/lustre-storage/checkpoints/zengh/spo/grpo_qwen1b_MATH_official_seed${SEED}"
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
# it on 2026-06-12. Pin to a per-job dir on node-local tmpfs (RAM-backed),
# falling back to the per-job TMPDIR if /dev/shm is unavailable.
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
echo "[sbatch] config: configs/polIter_qwen1b_grpo_MATH.jsonnet + EXTRA: configs/episode_generators/max_question_length_200.jsonnet"
echo "[sbatch] data/ contents: $(ls data 2>/dev/null | tr '\n' ' ')"
echo

# ── Hard pre-flight: guarantee this run is the OFFICIAL setting ──────────────
# 1) The guidance-cache patch must be present (else GUIDANCE_CACHE_DIR is ignored).
grep -q GUIDANCE_CACHE_DIR src/guidance/llms/caches/_diskcache.py || {
    echo "[sbatch] FATAL: _diskcache.py missing GUIDANCE_CACHE_DIR patch (git restore reverted it?)." >&2; exit 3; }
# 2) The override file must exist and pin 200 (this is what makes the run official).
grep -q 'max_question_length: 200' configs/episode_generators/max_question_length_200.jsonnet || {
    echo "[sbatch] FATAL: max_question_length_200.jsonnet does not pin 200." >&2; exit 3; }

# wandb authed in the env (entity=suesie). Override project/mode at submit.
export WANDB_PROJECT="${WANDB_PROJECT:-spo-math}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Single-GPU launch via the thin wrapper. EXTRA_CONFIGS layers the official
# max_question_length=200 reset LAST so it overrides the working-tree 512.
APP_SEED="$SEED" APP_DIRECTORY="$OUT" GPU_ID=0 \
EXTRA_CONFIGS="configs/episode_generators/max_question_length_200.jsonnet" \
    bash scripts/launch_server2_spo.sh grpo MATH
