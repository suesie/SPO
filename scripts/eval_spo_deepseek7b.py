#!/usr/bin/env python
"""Evaluate the SPO-tree DeepSeekMath-7B-SFT run on MATH-500 + OOD slate.

Slate:
  * MATH-500 (n=16) on ALL 40 ckpts (iter 25,50,...,1000)
  * AIME-24 (n=32) on final 4 ckpts (iter 925, 950, 975, 1000)
  * OlympiadBench (n=16) on final 4 ckpts
  * CollegeMath (n=16) on final 4 ckpts

All evals use max_model_len=4096, max_tokens=1024 — matched to grpocredit's VoI
standalone eval (eval_final_checkpoint.py leaves max_model_len unset, so vLLM
uses the deepseekmath-7B native max_position_embeddings=4096). This puts SPO
numbers on the IDENTICAL eval protocol as VoI/GRPO so they overlay directly.
NOTE: this is VoI-eval-matched, NOT SPO-training-matched (SPO trained at
model_context_size=2048); but since the longest test prompt is 1244 tok and
max_tokens=1024 binds, only ~2 OlympiadBench prompts differ from a 2048 run.
Sampling: T=0.35, top_p=0.9, seed=42 — VinePPO protocol, identical to grpocredit.

Grader: verl.utils.reward_score.math (boxed extraction + sympy equivalence).

Parquets: reuses grpocredit's parquets (data/verl/{math_test_vineppo,
aime24_test, olympiadbench_test, collegemath_test}.parquet) — same
[MATH_TASK] Problem: template the SPO 7B model was trained under.

Output: SPO/results/eval_spo_deepseek7b_seed42.json (single consolidated file,
appended atomically after each ckpt completes; restart-safe).

Usage:
  python scripts/eval_spo_deepseek7b.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_orchestrator import (  # noqa: E402
    RunSpec, WorkItem, add_common_args, discover_checkpoints, run_orchestrator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS_ROOT = Path(
    "/lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed42/"
    "polIter_deepseekSft2_spo_tree_MATH/checkpoints"
)
BASE_SNAPSHOT = Path(
    "/lustre-storage/datasets/zengh/huggingface/hub/"
    "models--realtreetune--deepseekmath-7b-sft-MATH-v2/snapshots/"
    "8b387c255b3bfaaaef2e650d56fecfde1c56ea96"
)
OUT_JSON = REPO_ROOT / "results" / "eval_spo_deepseek7b_seed42.json"

# Parquets come from grpocredit (built via build_verl_parquet_*.py there).
GRPOCREDIT_PARQUETS = Path("/home/zengh/projects/grpocredit/data/verl")
PARQUETS = {
    "math_v2": GRPOCREDIT_PARQUETS / "math_test_vineppo.parquet",
    "aime24": GRPOCREDIT_PARQUETS / "aime24_test.parquet",
    "olympiadbench": GRPOCREDIT_PARQUETS / "olympiadbench_test.parquet",
    "collegemath": GRPOCREDIT_PARQUETS / "collegemath_test.parquet",
}

# VoI-eval-matched protocol (max_model_len=4096 = VoI standalone) — see docstring.
TEMPERATURE = 0.35
TOP_P = 0.9
MAX_MODEL_LEN = 4096
MAX_TOKENS = 1024


def build_work_list(ckpts: list[tuple[int, int, Path]]) -> list[WorkItem]:
    items: list[WorkItem] = []
    final_four = ckpts[-4:]  # iter 925, 950, 975, 1000 assuming full run

    # MATH-500 on all 40
    for it, step, path in ckpts:
        items.append(WorkItem(
            ckpt_iter=it, ckpt_step=step, ckpt_path=path,
            dataset="math_v2", n=16,
            temperature=TEMPERATURE, top_p=TOP_P,
            max_tokens=MAX_TOKENS, max_model_len=MAX_MODEL_LEN,
            parquet=PARQUETS["math_v2"],
        ))

    # OOD on final 4
    ood_slate = [("aime24", 32), ("olympiadbench", 16), ("collegemath", 16)]
    for it, step, path in final_four:
        for ds, n in ood_slate:
            items.append(WorkItem(
                ckpt_iter=it, ckpt_step=step, ckpt_path=path,
                dataset=ds, n=n,
                temperature=TEMPERATURE, top_p=TOP_P,
                max_tokens=MAX_TOKENS, max_model_len=MAX_MODEL_LEN,
                parquet=PARQUETS[ds],
            ))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    for ds, p in PARQUETS.items():
        if not p.is_file():
            print(f"[FATAL] missing parquet for {ds}: {p}", file=sys.stderr)
            print(f"   build via grpocredit scripts first.", file=sys.stderr)
            sys.exit(3)

    if not CHECKPOINTS_ROOT.is_dir():
        print(f"[FATAL] missing checkpoints root: {CHECKPOINTS_ROOT}", file=sys.stderr)
        sys.exit(3)
    if not BASE_SNAPSHOT.is_dir():
        print(f"[FATAL] missing base snapshot: {BASE_SNAPSHOT}", file=sys.stderr)
        sys.exit(3)

    ckpts = discover_checkpoints(CHECKPOINTS_ROOT)
    print(f"[deepseek7b] discovered {len(ckpts)} checkpoints", file=sys.stderr)
    if len(ckpts) < 4:
        print(f"[FATAL] need at least 4 ckpts for OOD slate; got {len(ckpts)}", file=sys.stderr)
        sys.exit(3)

    work = build_work_list(ckpts)
    spec = RunSpec(
        model_label="spo_tree_deepseekmath_7b_sft_MATH_v2_seed42",
        checkpoints_root=CHECKPOINTS_ROOT,
        base_model_snapshot=BASE_SNAPSHOT,
        out_json=OUT_JSON,
        work_items=work,
    )
    run_orchestrator(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
