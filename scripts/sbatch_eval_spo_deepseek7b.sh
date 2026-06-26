#!/usr/bin/env bash
# Evaluate the SPO-tree deepseekmath-7B-SFT run end-to-end.
#
# 52 work items: MATH-500 (n=16) on all 40 ckpts + AIME-24 (n=32) +
# OlympiadBench (n=16) + CollegeMath (n=16) on final 4 ckpts. All at
# max_model_len=4096, max_tokens=1024 (matched to VoI standalone eval), T=0.35, top_p=0.9
# (VinePPO eval protocol, byte-compatible with grpocredit VoI/GRPO eval).
#
# Estimated wall: ~7h on 8xH200. Conservative 1d allocation.
#
# Output: /home/zengh/projects/SPO/results/eval_spo_deepseek7b_seed42.json
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
#SBATCH --job-name=eval-spo-deepseek7b
#SBATCH --output=/lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed42/eval_slurm-%j.log

set -euo pipefail

echo "[eval-deepseek7b] node=$(hostname)  date=$(date -u +%Y%m%dT%H%M%SZ)  jobid=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -8

# Use the grpocredit-verl-pinned env: provides verl.utils.reward_score.math
# (the math grader) and the vllm pin grpocredit's eval was validated against.
source /home/zengh/miniconda3/etc/profile.d/conda.sh
conda activate grpocredit-verl-pinned

cd /storage/home/zengh/projects/SPO

# HF offline — same flake protection as training/grpocredit eval sbatches.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[eval-deepseek7b] env: python=$(python --version) verl=$(python -c 'import verl; print(verl.__version__)') vllm=$(python -c 'import vllm; print(vllm.__version__)') torch=$(python -c 'import torch; print(torch.__version__)')"
echo "[eval-deepseek7b] git HEAD: $(git rev-parse HEAD)"
echo "[eval-deepseek7b] orchestrator SHA: $(sha1sum scripts/eval_spo_deepseek7b.py | awk '{print $1}')"
echo "[eval-deepseek7b] eval-one-ckpt SHA: $(sha1sum scripts/_eval_one_ckpt.py | awk '{print $1}')"

python scripts/eval_spo_deepseek7b.py
