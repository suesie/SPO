#!/usr/bin/env python
"""Single-checkpoint eval for SPO. Vendored from grpocredit eval_final_checkpoint.py.

Differences from upstream:
  * Adds --max-model-len (threaded to vLLM). SPO trained Qwen at effective 2K
    even though the R1-distill base ships max_position_embeddings=131072 — so
    the eval-time max_model_len MUST be set explicitly to honestly match
    training conditions.
  * Drops the FSDP-merge code path (SPO ckpts are already HF-format under
    <ckpt>/hf_pretrained/). The caller is responsible for passing a directory
    that satisfies the weights+tokenizer+config sanity check.
  * Drops the gsm8k grading branch (we don't eval on GSM8K).
  * Drops --dump-responses (length-hack diagnosis is a grpocredit debug
    feature, not in scope for SPO trajectory eval).
  * Emits a single JSON line to stdout in a delimited block so an orchestrator
    can subprocess this script without needing per-ckpt files.

Grading is selected by --grader (default 'verl' = UNCHANGED original behavior):
  * verl (default) = `verl.utils.reward_score.math.compute_score` (Hendrycks
    is_equiv) imported at runtime — identical to grpocredit so SPO vs VoI/GRPO
    numbers are byte-comparable. This path (`_grade_verl_only`) is the original
    code and the ONLY path the DeepSeekMath-7B eval uses.
  * math_verify / both = SPO's PUBLISHED extractive_match grader (open-r1
    `custom|math_500` config, in `_math_verify_grader.py`). Opted into only by
    the R1-Distill-Qwen-1.5B orchestrator (`_grade_multi`); 'both' also records
    the verl number under result["graders"]["verl"].
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_one_ckpt")


def extract_and_grade_math(text: str, gold: str) -> bool:
    try:
        from verl.utils.reward_score.math import compute_score

        return float(compute_score(text, gold)) > 0.5
    except Exception:
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        return bool(m) and m.group(1).strip() == str(gold).strip()


def get_prompt(row: pd.Series) -> str:
    p = row["prompt"]
    if isinstance(p, (list, np.ndarray)) and len(p) > 0:
        first = p[0]
        if isinstance(first, dict):
            return first.get("content", "")
    return str(p)


def get_ground_truth(row: pd.Series) -> str:
    rm = row["reward_model"]
    if isinstance(rm, dict):
        return str(rm.get("ground_truth", ""))
    if isinstance(rm, str):
        try:
            return str(json.loads(rm.replace("'", '"'))["ground_truth"])
        except Exception:
            pass
    return ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-ckpt", required=True, help="HF-format checkpoint dir (has config.json + weights + tokenizer)")
    ap.add_argument("--test-parquet", required=True, help="verl-schema parquet")
    ap.add_argument(
        "--dataset",
        choices=["math_v2", "aime24", "olympiadbench", "collegemath"],
        required=True,
    )
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens per response")
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Total context budget (prompt+response) passed to vLLM. None = use model config. "
             "SPO trained at 2048 effective for both models; set this explicitly to match.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument(
        "--grader",
        choices=["verl", "math_verify", "both"],
        default="verl",
        help="Grader(s) to score with. DEFAULT 'verl' = the original Hendrycks "
             "is_equiv path (unchanged behavior + output). 'math_verify' = SPO's "
             "published extractive_match (open-r1 math_500 config); 'both' records "
             "both (primary = math_verify). Only the Qwen-1.5B orchestrator opts in.",
    )
    ap.add_argument(
        "--mv-precision", type=int, default=5,
        help="float_rounding for the math_verify grader (open-r1 math_500 used 5). "
             "Ignored when --grader verl.",
    )
    ap.add_argument(
        "--stop",
        action="append",
        default=None,
        help="vLLM stop string. Pass multiple times for multiple stops. Default = ['\\n\\n\\nProblem:'] "
             "(SFT data is multi-problem-concatenated). Pass --stop '' to disable.",
    )
    return ap.parse_args()


def _grade_verl_only(outputs, golds, args, ckpt, test, gen_seconds) -> dict:
    """ORIGINAL grading path (verl Hendrycks is_equiv). Byte-identical to the
    pre-grader-option behavior — used by the DeepSeekMath-7B eval and anyone not
    opting into math_verify. Do not change without changing the 7B eval."""
    per_q_pass1: list[float] = []
    per_q_once_hit: list[float] = []
    for q_idx, out in enumerate(outputs):
        gold = golds[q_idx]
        grading = [extract_and_grade_math(s.text, gold) for s in out.outputs]
        per_q_pass1.append(sum(grading) / len(grading))
        per_q_once_hit.append(1.0 if any(grading) else 0.0)

    N = len(per_q_pass1)
    n = args.n
    p_arr = np.asarray(per_q_pass1)
    se_cluster = float(np.std(p_arr, ddof=1) / np.sqrt(N)) if N > 1 else 0.0
    se_bernoulli = float(np.sqrt((p_arr * (1 - p_arr)).mean() / (n * N)))
    ci95 = float(1.96 * se_cluster)

    result = {
        "hf_ckpt": str(ckpt),
        "test_parquet": str(test),
        "dataset": args.dataset,
        "n_questions": int(N),
        "n_samples_per_question": int(n),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "pass1": float(np.mean(per_q_pass1)),
        "pass_at_n": float(np.mean(per_q_once_hit)),
        "se_cluster": se_cluster,
        "se_bernoulli": se_bernoulli,
        "ci95": ci95,
        "gen_seconds": gen_seconds,
        "tensor_parallel_size": args.tensor_parallel_size,
    }

    print(
        f"\n=== Result ===\n"
        f"pass@1   = {result['pass1']:.4f} ± {ci95:.4f}  (95% CI, cluster SE; n={n}, N={N})\n"
        f"pass@{n:<3} = {result['pass_at_n']:.4f}\n"
        f"gen_seconds = {gen_seconds:.1f}\n"
    )
    return result


def _grade_multi(outputs, golds, args, ckpt, test, gen_seconds) -> dict:
    """math_verify (SPO extractive_match) grading, optionally alongside verl.

    args.grader is 'math_verify' or 'both'. Primary metrics (top-level pass1/…)
    come from math_verify; when 'both', verl numbers are recorded under
    result["graders"]["verl"]. Only the Qwen-1.5B orchestrator selects this."""
    n = args.n
    N = len(outputs)

    grader_configs: dict = {}
    mv_grader = None
    if args.grader in ("math_verify", "both"):
        try:
            from _math_verify_grader import MathVerifyGrader

            mv_grader = MathVerifyGrader(precision=args.mv_precision)
            grader_configs["math_verify"] = mv_grader.describe()
        except Exception as e:
            # grader==math_verify was explicitly requested -> hard fail; grader==both
            # -> degrade gracefully to verl only (matches _math_verify_grader's
            # "caller decides policy" and the --grader both docs).
            if args.grader == "math_verify":
                log.error("math_verify grader requested but unavailable: %s", e)
                sys.exit(66)
            log.warning("math_verify grader unavailable (%s) — scoring with verl only", e)
    if args.grader == "both":
        grader_configs["verl"] = {
            "grader": "verl_hendrycks_is_equiv",
            "engine": "verl.utils.reward_score.math.compute_score",
            "note": "last-boxed extraction + strip_string + literal ==; no sympy",
            "decision": "compute_score(text, gold) > 0.5",
        }
    active = list(grader_configs.keys())
    primary = "math_verify" if "math_verify" in active else "verl"

    # Pre-parse gold once per question for math_verify (reused across n samples).
    mv_gold_parsed = None
    mv_gold_fail = 0
    if mv_grader is not None:
        mv_gold_parsed = []
        for g in golds:
            gp = mv_grader.parse_gold(g)
            if len(gp) == 0:
                mv_gold_fail += 1
            mv_gold_parsed.append(gp)
        if mv_gold_fail:
            log.warning(
                "math_verify: %d/%d gold answers did not extract to sympy "
                "(raw-string fallback used for those)", mv_gold_fail, N,
            )

    per_q = {name: {"pass1": [], "hit": []} for name in active}
    t_grade = time.time()
    for q_idx, out in enumerate(outputs):
        gold = golds[q_idx]
        texts = [s.text for s in out.outputs]
        for name in active:
            if name == "math_verify":
                gp = mv_gold_parsed[q_idx]
                grades = [mv_grader.grade(t, gold, gold_parsed=gp) for t in texts]
            else:
                grades = [extract_and_grade_math(t, gold) for t in texts]
            per_q[name]["pass1"].append(sum(grades) / len(grades) if grades else 0.0)
            per_q[name]["hit"].append(1.0 if any(grades) else 0.0)
    grade_seconds = time.time() - t_grade

    def _stats(pass1_list, hit_list):
        p = np.asarray(pass1_list)
        se_cluster = float(np.std(p, ddof=1) / np.sqrt(N)) if N > 1 else 0.0
        se_bernoulli = float(np.sqrt((p * (1 - p)).mean() / (n * N))) if N > 0 else 0.0
        return {
            "pass1": float(np.mean(pass1_list)),
            "pass_at_n": float(np.mean(hit_list)),
            "se_cluster": se_cluster,
            "se_bernoulli": se_bernoulli,
            "ci95": float(1.96 * se_cluster),
        }

    graders_stats = {name: _stats(per_q[name]["pass1"], per_q[name]["hit"]) for name in active}
    ps = graders_stats[primary]

    result = {
        "hf_ckpt": str(ckpt),
        "test_parquet": str(test),
        "dataset": args.dataset,
        "n_questions": int(N),
        "n_samples_per_question": int(n),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "grader": primary,
        "grader_mode": args.grader,
        "pass1": ps["pass1"],
        "pass_at_n": ps["pass_at_n"],
        "se_cluster": ps["se_cluster"],
        "se_bernoulli": ps["se_bernoulli"],
        "ci95": ps["ci95"],
        "graders": graders_stats,
        "grader_configs": grader_configs,
        "mv_gold_extract_failures": int(mv_gold_fail),
        "mv_grade_errors": int(mv_grader.grade_errors) if mv_grader is not None else 0,
        "gen_seconds": gen_seconds,
        "grade_seconds": grade_seconds,
        "tensor_parallel_size": args.tensor_parallel_size,
    }

    summary = "\n".join(
        f"  {name:11s} pass@1={graders_stats[name]['pass1']:.4f} "
        f"± {graders_stats[name]['ci95']:.4f}   pass@{n}={graders_stats[name]['pass_at_n']:.4f}"
        + ("   [PRIMARY]" if name == primary else "")
        for name in active
    )
    print(
        f"\n=== Result ({args.dataset}, n={n}, N={N}) ===\n"
        f"{summary}\n"
        f"gen_seconds = {gen_seconds:.1f}  grade_seconds = {grade_seconds:.1f}\n"
    )
    return result


def main() -> None:
    args = parse_args()

    ckpt = Path(args.hf_ckpt).resolve()
    if not ckpt.is_dir():
        log.error("hf-ckpt does not exist or is not a dir: %s", ckpt)
        sys.exit(64)
    has_weights = (
        any(ckpt.glob("*.safetensors"))
        or (ckpt / "pytorch_model.bin").exists()
        or (ckpt / "pytorch_model.bin.index.json").exists()
    )
    if not has_weights:
        log.error("hf-ckpt has no model weights: %s", ckpt)
        sys.exit(64)
    if not (ckpt / "config.json").exists():
        log.error("hf-ckpt missing config.json: %s", ckpt)
        sys.exit(64)
    if not list(ckpt.glob("tokenizer*.json")):
        log.error("hf-ckpt missing tokenizer*.json: %s", ckpt)
        sys.exit(64)

    test = Path(args.test_parquet).resolve()
    if not test.is_file():
        log.error("test parquet not found: %s", test)
        sys.exit(64)

    df = pd.read_parquet(test)
    log.info("Loaded %d questions from %s", len(df), test)

    prompts = df.apply(get_prompt, axis=1).tolist()
    golds = df.apply(get_ground_truth, axis=1).tolist()

    if not prompts[0]:
        log.error("First prompt empty — parquet schema mismatch")
        sys.exit(2)

    if args.stop is None:
        stop = ["\n\n\nProblem:"]
    elif args.stop == [""]:
        stop = None
    else:
        stop = [s for s in args.stop if s]

    log.info(
        "Loading %s on TP=%d mem=%.2f max_model_len=%s ...",
        ckpt,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.max_model_len,
    )
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=str(ckpt),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=stop,
    )

    log.info(
        "Generating %d × %d = %d rollouts (T=%.2f, top_p=%.2f, max_tokens=%d)...",
        args.n,
        len(prompts),
        args.n * len(prompts),
        args.temperature,
        args.top_p,
        args.max_tokens,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_seconds = time.time() - t0

    # --grader verl (default) is the ORIGINAL code path, byte-identical output.
    # math_verify / both add SPO's published extractive_match grader; only the
    # Qwen-1.5B orchestrator opts in, so the 7B eval is unaffected.
    if args.grader == "verl":
        result = _grade_verl_only(outputs, golds, args, ckpt, test, gen_seconds)
    else:
        result = _grade_multi(outputs, golds, args, ckpt, test, gen_seconds)

    print("===RESULT_JSON_BEGIN===")
    print(json.dumps(result))
    print("===RESULT_JSON_END===")


if __name__ == "__main__":
    main()
