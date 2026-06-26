#!/usr/bin/env python
"""Shared eval-orchestrator helpers for SPO trajectory evals.

Used by eval_spo_deepseek7b.py and eval_spo_qwen1b.py to enumerate
checkpoints, build a work list, run _eval_one_ckpt.py as a subprocess
per item, and accumulate results into a single consolidated JSON.

The two model-specific orchestrators just configure RunSpec and call
run_orchestrator(); everything else is shared.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("eval_orch")

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "scripts" / "_eval_one_ckpt.py"
PREP_SCRIPT = REPO_ROOT / "scripts" / "_prepare_spo_ckpt_for_vllm.sh"

CKPT_DIR_RE = re.compile(r"^ckpt--iter_(\d+)--epoch_[\d.]+--step_(\d+)$")


@dataclass(frozen=True)
class WorkItem:
    """One eval to run."""

    ckpt_iter: int
    ckpt_step: int
    ckpt_path: Path
    dataset: str
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    max_model_len: int  # always set explicitly for SPO; never None
    parquet: Path

    def key(self) -> tuple:
        """Identity tuple for resume-skip logic."""
        return (self.ckpt_iter, self.dataset, self.n, self.max_model_len, self.max_tokens)


@dataclass
class RunSpec:
    model_label: str
    checkpoints_root: Path  # contains ckpt--iter_*/ dirs
    base_model_snapshot: Path
    out_json: Path
    work_items: list[WorkItem] = field(default_factory=list)
    tensor_parallel_size: int = 8
    gpu_memory_utilization: float = 0.85
    seed: int = 42


def discover_checkpoints(root: Path) -> list[tuple[int, int, Path]]:
    """Return [(iter, step, path), ...] sorted by iter."""
    items: list[tuple[int, int, Path]] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = CKPT_DIR_RE.match(d.name)
        if not m:
            continue
        hf_dir = d / "hf_pretrained"
        if not hf_dir.is_dir():
            log.warning("ckpt %s has no hf_pretrained/, skipping", d.name)
            continue
        items.append((int(m.group(1)), int(m.group(2)), hf_dir))
    items.sort(key=lambda t: t[0])
    return items


def load_existing_results(out_json: Path) -> tuple[Optional[dict], set]:
    """Load prior consolidated JSON if present. Returns (doc, set_of_done_keys)."""
    if not out_json.exists():
        return None, set()
    try:
        doc = json.loads(out_json.read_text())
    except Exception as e:
        log.warning("failed to parse existing %s: %s — starting fresh", out_json, e)
        return None, set()
    done = set()
    for ev in doc.get("evals", []):
        key = (
            int(ev["ckpt_iter"]),
            str(ev["dataset"]),
            int(ev["n_samples_per_question"]),
            int(ev["max_model_len"]),
            int(ev["max_tokens"]),
        )
        done.add(key)
    log.info("loaded %d completed evals from %s", len(done), out_json)
    return doc, done


def atomic_write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def prepare_ckpt(hf_dir: Path, base_snapshot: Path) -> None:
    subprocess.run(
        ["bash", str(PREP_SCRIPT), str(hf_dir), str(base_snapshot)],
        check=True,
    )


def run_one_eval(item: WorkItem, spec: RunSpec) -> dict:
    """Subprocess _eval_one_ckpt.py, parse the JSON block, return the result dict."""
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--hf-ckpt", str(item.ckpt_path),
        "--test-parquet", str(item.parquet),
        "--dataset", item.dataset,
        "--n", str(item.n),
        "--temperature", str(item.temperature),
        "--top-p", str(item.top_p),
        "--max-tokens", str(item.max_tokens),
        "--max-model-len", str(item.max_model_len),
        "--seed", str(spec.seed),
        "--tensor-parallel-size", str(spec.tensor_parallel_size),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
    ]
    log.info("CMD: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    # Stream child output to our log so we can debug a crash from the parent log.
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"_eval_one_ckpt.py exited {proc.returncode} for iter={item.ckpt_iter} ds={item.dataset}"
        )

    # Pull the delimited JSON block out of stdout.
    start = proc.stdout.rfind("===RESULT_JSON_BEGIN===")
    end = proc.stdout.rfind("===RESULT_JSON_END===")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("could not find ===RESULT_JSON_BEGIN/END=== markers in subprocess stdout")
    payload = proc.stdout[start + len("===RESULT_JSON_BEGIN===") : end].strip()
    result = json.loads(payload)
    result["wall_seconds"] = wall
    result["ckpt_iter"] = item.ckpt_iter
    result["ckpt_step"] = item.ckpt_step
    return result


def run_orchestrator(spec: RunSpec, dry_run: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    log.info("=== %s eval orchestrator ===", spec.model_label)
    log.info("checkpoints_root: %s", spec.checkpoints_root)
    log.info("base_model_snapshot: %s", spec.base_model_snapshot)
    log.info("out_json: %s", spec.out_json)
    log.info("work items: %d", len(spec.work_items))

    if dry_run:
        log.info("--- DRY RUN: work list ---")
        for i, it in enumerate(spec.work_items):
            log.info(
                "[%3d] iter=%4d ds=%-13s n=%2d max_model_len=%4d max_tokens=%4d T=%.2f top_p=%.2f",
                i, it.ckpt_iter, it.dataset, it.n, it.max_model_len, it.max_tokens,
                it.temperature, it.top_p,
            )
        log.info("--- DRY RUN: end ---")
        return

    doc, done_keys = load_existing_results(spec.out_json)
    if doc is None:
        doc = {
            "model_label": spec.model_label,
            "run_dir": str(spec.checkpoints_root),
            "base_model_snapshot": str(spec.base_model_snapshot),
            "seed": spec.seed,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "evals": [],
        }

    pending = [it for it in spec.work_items if it.key() not in done_keys]
    log.info("skipping %d already-done items; %d items remain", len(done_keys), len(pending))

    failures: list[tuple[WorkItem, str]] = []
    prepped: set[Path] = set()
    for i, item in enumerate(pending):
        log.info(
            "============================================================\n"
            "[%d/%d] iter=%d ds=%s n=%d max_model_len=%d max_tokens=%d",
            i + 1, len(pending), item.ckpt_iter, item.dataset, item.n,
            item.max_model_len, item.max_tokens,
        )
        try:
            if item.ckpt_path not in prepped:
                prepare_ckpt(item.ckpt_path, spec.base_model_snapshot)
                prepped.add(item.ckpt_path)
            result = run_one_eval(item, spec)
        except Exception as e:
            log.exception("FAILED item iter=%d ds=%s: %s", item.ckpt_iter, item.dataset, e)
            failures.append((item, str(e)))
            continue

        doc["evals"].append(result)
        doc["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(spec.out_json, doc)
        log.info(
            "  -> pass1=%.4f ± %.4f  pass@%d=%.4f  wall=%.0fs",
            result["pass1"], result["ci95"], result["n_samples_per_question"],
            result["pass_at_n"], result["wall_seconds"],
        )

    log.info("============================================================")
    log.info("DONE. wrote %d total evals to %s", len(doc["evals"]), spec.out_json)
    if failures:
        log.error("FAILURES (%d):", len(failures))
        for it, err in failures:
            log.error("  iter=%d ds=%s n=%d ctx=%d: %s", it.ckpt_iter, it.dataset, it.n, it.max_model_len, err)
        sys.exit(1)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dry-run", action="store_true", help="Print work list and exit; do not run vLLM")
