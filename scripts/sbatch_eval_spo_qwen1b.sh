#!/usr/bin/env bash
# Evaluate the SPO-tree DeepSeek-R1-Distill-Qwen-1.5B run end-to-end.
#
# 56 work items: MATH-500 (n=16) @ 2K on all 40 ckpts + AIME-24 (n=32) +
# OlympiadBench (n=16) + CollegeMath (n=16) @ 2K on final 4 ckpts +
# MATH-500 (n=16) @ 4K on final 4 ckpts (extrapolation row matching SPO's
# published 4K column). All at T=0.6, top_p=0.95 (matches SPO's
# evaluate_long_cot.sh + paper numbers).
#
# Estimated wall: ~4h on 8xH200. Conservative 1d allocation.
#
# Output: /home/zengh/projects/SPO/results/eval_spo_qwen1b_seed42.json
# (single consolidated file; appended atomically after each ckpt completes,
#  so restart-safe — resubmit on preempt/timeout to pick up where it left off).

#SBATCH --account=h200_mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval-spo-qwen1b
#SBATCH --output=/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42/eval_slurm-%j.log

set -euo pipefail

echo "[eval-qwen1b] node=$(hostname)  date=$(date -u +%Y%m%dT%H%M%SZ)  jobid=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -8

source /home/zengh/miniconda3/etc/profile.d/conda.sh
conda activate grpocredit-verl-pinned

cd /storage/home/zengh/projects/SPO

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[eval-qwen1b] env: python=$(python --version) verl=$(python -c 'import verl; print(verl.__version__)') vllm=$(python -c 'import vllm; print(vllm.__version__)') torch=$(python -c 'import torch; print(torch.__version__)')"
echo "[eval-qwen1b] git HEAD: $(git rev-parse HEAD)"
echo "[eval-qwen1b] orchestrator SHA: $(sha1sum scripts/eval_spo_qwen1b.py | awk '{print $1}')"
echo "[eval-qwen1b] eval-one-ckpt SHA: $(sha1sum scripts/_eval_one_ckpt.py | awk '{print $1}')"

# Build the Qwen-templated parquets if they don't exist (idempotent).
if [ ! -f data/verl_qwen/math_v2_test.parquet ]; then
    echo "[eval-qwen1b] Building Qwen-templated parquets ..."
    python scripts/build_qwen_templated_parquets.py
fi

python scripts/eval_spo_qwen1b.py
