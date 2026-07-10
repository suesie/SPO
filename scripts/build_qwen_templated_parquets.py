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

NOTE on BOS: the R1-distill tokenizer has add_bos_token=True, so it auto-prepends
exactly one BOS (151646) on encode() — token-verified: encode("x") -> [151646, ...].
This harness feeds RAW strings to vLLM, which tokenizes via tokenizer.encode(text)
with HF's default add_special_tokens=True (vLLM's encode_tokens(..., None) path), so
vLLM ALSO auto-prepends the BOS. The on-disk QWEN_TEMPLATE below must therefore NOT
embed a literal <｜begin▁of▁sentence｜>, or the tokenized prompt gets a DOUBLE BOS
([151646, 151646, ...]) the model never saw. Omitting it yields exactly one BOS —
matching SPO's lighteval eval (--use-chat-template) and the training template.
See QWEN_TEMPLATE.

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

# Bump whenever QWEN_TEMPLATE changes. A marker with this value is written ONLY after
# all four parquets are (re)built, so an interrupted rebuild leaves the marker stale
# and the next run rebuilds; a template change (new version) also forces a rebuild.
# 'nobos-v2' = the 2026-07 fix REVERTING the erroneous embedded BOS. bos-v1 embedded a
# literal <｜begin▁of▁sentence｜> and produced a DOUBLE BOS (vLLM already auto-prepends
# one — see QWEN_TEMPLATE); this version bump forces regeneration of any bos-v1 parquets.
TEMPLATE_VERSION = "r1distill-nobos-v2"
MARKER = OUT_DIR / ".template_version"

# From configs/prompt_library/qwen_MATH.jsonnet, WITHOUT the leading literal
# <｜begin▁of▁sentence｜> (the training jsonnet shows it, but see below).
#
# CRITICAL (token-verified 2026-07 against deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# @ ad9f0ae): this tokenizer has add_bos_token=True and DOES auto-prepend the BOS on
# encode() — encode("x") -> [151646, ...]. The eval harness (_eval_one_ckpt.py) passes
# RAW strings to vLLM, which tokenizes with tokenizer.encode(text) (HF default
# add_special_tokens=True), so vLLM auto-prepends exactly one BOS. We therefore must
# NOT embed a literal BOS here, or the tokenized prompt has TWO ([151646, 151646, ...])
# — a sequence the model never saw in training. Omitting it yields exactly one 151646,
# byte-identical to SPO's published lighteval eval (`evaluate_long_cot.sh`,
# --use-chat-template) AND to the training template.
# (bos-v1 wrongly embedded the literal on the mistaken belief that vLLM does NOT add
# it; that produced a double BOS and is reverted here.)
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


def _up_to_date() -> bool:
    """True iff all four parquets exist AND were built with the current template."""
    if not MARKER.exists() or MARKER.read_text().strip() != TEMPLATE_VERSION:
        return False
    return all((OUT_DIR / f"{ds}_test.parquet").is_file() for ds in DATASETS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if _up_to_date():
        log.info("Qwen parquets already at template %s — nothing to do", TEMPLATE_VERSION)
        return
    # Stale/missing/partial: rebuild ALL and (re)write the marker only on full success,
    # so an interrupted rebuild cannot leave a subset stale-but-marked-fresh.
    MARKER.unlink(missing_ok=True)
    log.info("writing Qwen-templated parquets to %s (template %s)", OUT_DIR, TEMPLATE_VERSION)

    built = 0
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
        built += 1
        log.info("[%s] wrote %d rows to %s", ds_name, len(df_out), dst)

        # Spot-check first row so a future me can eyeball template correctness
        sample_in = df.iloc[0]["prompt"][0]["content"][:120].replace("\n", "\\n")
        sample_out = df_out.iloc[0]["prompt"][0]["content"][:160].replace("\n", "\\n")
        log.info("  in : %s", sample_in)
        log.info("  out: %s", sample_out)

    if built == len(DATASETS):
        MARKER.write_text(TEMPLATE_VERSION + "\n")
        log.info("all %d parquets built; wrote marker %s = %s", built, MARKER, TEMPLATE_VERSION)
    else:
        log.warning("only %d/%d parquets built; marker NOT written (next run will retry)",
                    built, len(DATASETS))


if __name__ == "__main__":
    main()
