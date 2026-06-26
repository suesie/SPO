#!/usr/bin/env python
"""Rewrite grpocredit's verl parquets with the R1-distill chat template.

SPO's Qwen-1.5B runs use the R1-distill prompt format
(see configs/prompt_library/qwen_MATH.jsonnet):

    <｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently
    and clearly.  The last line of your response should be of the following format:
    'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct'
    (without quotes) where ANSWER is just the final number or expression that
    solves the problem. Think step by step before answering.

    {query}<｜Assistant｜><think>\n

NOTE on BOS: the training template above (from the jsonnet) shows a leading
literal <｜begin▁of▁sentence｜>, but the on-disk QWEN_TEMPLATE below intentionally
OMITS it. The eval harness feeds raw strings to vLLM, which auto-prepends exactly
one BOS; embedding it literally would produce a double BOS. See QWEN_TEMPLATE.

The grpocredit parquets wrap each problem in `[MATH_TASK] Problem:\n{q}\n\nSolution:`
which is the deepseekmath-SFT template — wrong for R1-distill, which would
treat those raw tokens as a continuation prompt and ignore the chat structure
it was distilled with.

This script reads the 4 grpocredit parquets, strips the `[MATH_TASK]`
wrapper to recover the bare problem, then rewraps with the R1-distill
template. All other columns (data_source, ability, reward_model, extra_info)
are passed through unchanged so the same grader works.

Output goes to SPO/data/verl_qwen/{math_v2,aime24,olympiadbench,collegemath}_test.parquet.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_qwen_parquets")

GRPOCREDIT_DATA = Path("/home/zengh/projects/grpocredit/data/verl")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "verl_qwen"

# From configs/prompt_library/qwen_MATH.jsonnet, but WITHOUT the leading literal
# <｜begin▁of▁sentence｜>. The eval harness (_eval_one_ckpt.py) passes raw strings
# to vLLM's llm.generate(), which tokenizes with add_special_tokens=True and so
# auto-prepends exactly one BOS (151646). A literal BOS here would double it
# ([151646, 151646, <｜User｜>...]) — a sequence the model never saw in training.
# grpocredit's [MATH_TASK] template likewise carries no literal BOS, so dropping
# it keeps the two evals byte-comparable.
QWEN_TEMPLATE = (
    "<｜User｜>"
    "Solve the following math problem efficiently and clearly.  "
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression "
    "that solves the problem. Think step by step before answering.\n\n"
    "{query}<｜Assistant｜><think>\n"
)

DEEPSEEK_PREFIX = "[MATH_TASK] Problem:\n"
DEEPSEEK_SUFFIX = "\n\nSolution:"

DATASETS = {
    "math_v2": "math_test_vineppo.parquet",
    "aime24": "aime24_test.parquet",
    "olympiadbench": "olympiadbench_test.parquet",
    "collegemath": "collegemath_test.parquet",
}


def unwrap_deepseek_template(content: str) -> str:
    if not content.startswith(DEEPSEEK_PREFIX):
        raise ValueError(f"prompt missing [MATH_TASK] prefix: {content[:80]!r}")
    if not content.endswith(DEEPSEEK_SUFFIX):
        raise ValueError(f"prompt missing Solution: suffix: ...{content[-80:]!r}")
    return content[len(DEEPSEEK_PREFIX) : -len(DEEPSEEK_SUFFIX)]


def rewrap_one(row: pd.Series) -> pd.Series:
    prompt_list = row["prompt"]
    assert len(prompt_list) == 1 and prompt_list[0]["role"] == "user"
    bare_problem = unwrap_deepseek_template(prompt_list[0]["content"])
    new_content = QWEN_TEMPLATE.format(query=bare_problem)
    new_row = row.copy()
    new_row["prompt"] = [{"role": "user", "content": new_content}]
    return new_row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("writing Qwen-templated parquets to %s", OUT_DIR)

    for ds_name, src_fname in DATASETS.items():
        src = GRPOCREDIT_DATA / src_fname
        dst = OUT_DIR / f"{ds_name}_test.parquet"
        if not src.exists():
            log.error("MISSING: %s — build it via grpocredit first", src)
            continue

        df = pd.read_parquet(src)
        log.info("[%s] read %d rows from %s", ds_name, len(df), src.name)

        df_out = df.apply(rewrap_one, axis=1)
        df_out.to_parquet(dst, index=False)
        log.info("[%s] wrote %d rows to %s", ds_name, len(df_out), dst)

        # Spot-check first row so a future me can eyeball template correctness
        sample_in = df.iloc[0]["prompt"][0]["content"][:120].replace("\n", "\\n")
        sample_out = df_out.iloc[0]["prompt"][0]["content"][:160].replace("\n", "\\n")
        log.info("  in : %s", sample_in)
        log.info("  out: %s", sample_out)


if __name__ == "__main__":
    main()
