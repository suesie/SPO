#!/usr/bin/env python
r"""math-verify grader matching SPO's PUBLISHED MATH-500 evaluation exactly.

SPO reported its headline MATH-500 numbers (README: SPO-tree-666 = 0.736 @2K,
0.828 @4K, 0.848 @32K) with lighteval's ``extractive_match`` metric, configured
by open-r1's ``custom|math_500`` task. This module reproduces that grader so our
offline trajectory harness (``_eval_one_ckpt.py``) can grade the way lighteval
did, instead of with verl's Hendrycks ``is_equiv`` (strip_string + literal
``==``), which requires a ``\boxed{}`` and under-counts non-boxed / symbolically
equal answers.

PROVENANCE (verified against source, 2026-07):

* open-r1 ``src/open_r1/evaluate.py`` @ rev ``eeca246b`` (2025-02-22 — the last
  change to that file before SPO's 2025-04-07 run; the SPO result JSONs record
  ``custom|math_500`` version=1 / ``math_prompt_fn``, matching this revision).
  The ``math_500`` task's metric::

        latex_gold_metric = multilingual_extractive_match_metric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(LatexExtractionConfig(),),
            # Match boxed first before trying other regexes
            pred_extraction_target=(ExprExtractionConfig(),
                                    LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )

  and gold for ``math_500`` comes from ``line["solution"]`` (the full worked
  solution ending in ``\boxed{...}``), extracted with the LaTeX target above.

* lighteval's ``extractive_match`` is a VENDORED copy of the parse+verify logic
  (``lighteval.metrics.utils.{extractive_match_utils, math_comparison}``); the
  HuggingFace ``math_verify`` pip package is the same code spun out. We use the
  pip package as the engine — semantically 1:1 with lighteval's vendored copy.

* per-sample reduction (lighteval ``MultilingualExtractiveMatchMetric.compute``)::

        score = max_over_preds(
            1.0 if any(verify(gold, pred, precision) for gold in extracted_golds)
            else 0.0)

  with the empty-gold fallback: if a gold fails to extract, compare against the
  RAW gold string. We grade one sample at a time, so the outer ``max`` is a no-op
  here; the harness aggregates mean@k across the n samples per question.

FIDELITY CAVEATS:

* lighteval used its vendored parse+verify pinned to ``latex2sympy2_extended==1.0.6``;
  we use the ``math_verify`` pip package (0.9.0 in the reference env) with whatever
  ``latex2sympy2_extended`` it pulls (1.11.0 in the reference env). The grading
  ALGORITHM is identical (verified against math_verify's own ``metric.py`` and
  lighteval's ``math_comparison``); only the LaTeX-parser/normalization revision
  differs, which can shift a handful of exotic strings. Empirically robust: the same
  cases grade identically under math_verify 0.5.2 and 0.9.0. If BIT-exact parity with
  the 2025-04-07 run is required, pin ``latex2sympy2_extended==1.0.6``.
  ``test_grader_parity.py`` pins the expected labels.
* Gold source: open-r1 parses ``line["solution"]`` (the full worked solution, which
  contains one ``\boxed{}``). Our parquet ``reward_model.ground_truth`` is the bare
  pre-extracted answer, so we wrap it in ``\boxed{}`` before parsing with the gold
  target — verified to yield the identical sympy value as parsing the full solution
  (see ``test_grader_parity.py`` SOLUTION_PARITY). The ONE divergence is a solution
  with MULTIPLE ``\boxed{}``: open-r1's ``any_match`` collapses them into a sympy set
  that matches neither scalar (grading such items False), whereas our single-box gold
  grades by the canonical answer. Standard MATH-500 solutions have exactly one box,
  so this affects at most a few / 500 and, where it differs, our grader is the more
  correct of the two (never inflates our number relative to the paper).
* ``precision=5`` is passed positionally as ``float_rounding`` (``numeric_precision``
  stays at the library default), matching open-r1/lighteval.
* TIMEOUTS: math_verify's parse/verify use ``signal.alarm``, which only works on
  the MAIN thread. The harness grades in the main thread. Do not call this from a
  worker thread without passing ``timeout_seconds=None`` and supplying your own
  timeout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("math_verify_grader")

# open-r1 math_500 config (rev eeca246b). Do not change without updating provenance.
OPENR1_PRECISION = 5


@dataclass
class MathVerifyGrader:
    """extractive_match grader replicating open-r1's ``custom|math_500`` metric.

    Import of ``math_verify`` is deferred to construction so this module can be
    imported cheaply (and so the harness can degrade gracefully if the eval env
    lacks math_verify).
    """

    precision: int = OPENR1_PRECISION
    grade_errors: int = 0  # count of samples that raised during grading (scored False)

    _parse: Callable[..., Any] = field(init=False, repr=False)
    _verify: Callable[..., Any] = field(init=False, repr=False)
    _gold_target: tuple = field(init=False, repr=False)
    _pred_target: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Raises ImportError if math_verify is unavailable — caller decides policy.
        from math_verify import (  # noqa: F401
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )

        self._parse = parse
        self._verify = verify
        # gold: open-r1 uses (LatexExtractionConfig(),) on line["solution"].
        self._gold_target = (LatexExtractionConfig(),)
        # pred: (ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)).
        # boxed_match_priority=0 => boxed answers matched before other regexes;
        # try_extract_without_anchor keeps its default (True) as in open-r1.
        self._pred_target = (
            ExprExtractionConfig(),
            LatexExtractionConfig(boxed_match_priority=0),
        )

    @staticmethod
    def _wrap_gold(gold: str) -> str:
        r"""Our parquet gold is a bare answer; open-r1 gold is the boxed solution.

        Wrapping bare answers in ``\boxed{}`` makes the LaTeX gold target extract
        them identically to how it extracts from ``line["solution"]`` — required
        for e.g. bare integer AIME answers, which the LaTeX target will not pick
        up without a box.
        """
        g = str(gold)
        return g if "\\boxed" in g else ("\\boxed{" + g + "}")

    def parse_gold(self, gold: str):
        """Extract gold to sympy candidates (parse once per question, reuse per sample)."""
        return self._parse(
            self._wrap_gold(gold),
            extraction_config=self._gold_target,
            fallback_mode="first_match",
        )

    def parse_pred(self, pred_text: str):
        return self._parse(
            pred_text,
            extraction_config=self._pred_target,
            fallback_mode="first_match",
        )

    def grade(self, pred_text: str, gold: str, gold_parsed: Optional[list] = None) -> bool:
        r"""True iff ``pred_text`` extractive-matches ``gold`` under open-r1's config.

        Pass ``gold_parsed`` (from :meth:`parse_gold`) to avoid re-parsing gold on
        every one of the n samples for a question.
        """
        try:
            egold = gold_parsed if gold_parsed is not None else self.parse_gold(gold)
            if len(egold) == 0:
                # lighteval empty-gold fallback: compare against the raw gold string.
                egold = [str(gold)]
            epred = self.parse_pred(pred_text)
            # precision is the 3rd positional arg = float_rounding (matches lighteval).
            # verify() accepts the candidate LISTS and does an any-match over the
            # gold x pred cross-product internally.
            return bool(self._verify(egold, epred, self.precision))
        except Exception as e:  # noqa: BLE001
            # lighteval wraps every comparison in try/except -> False; match that,
            # AND never let a single pathological sample abort a whole-checkpoint eval
            # (the verl path is likewise exception-guarded).
            self.grade_errors += 1
            log.debug("math_verify grade error (counted, scored 0): %s", e)
            return False

    def describe(self) -> dict:
        return {
            "grader": "math_verify_extractive_match",
            "engine": "math_verify (pip) — 1:1 with lighteval vendored extractive_match",
            "provenance": "open-r1 evaluate.py@eeca246b custom|math_500 (SPO published grader)",
            "precision_float_rounding": self.precision,
            "gold_extraction_target": "(LatexExtractionConfig(),) on \\boxed-wrapped gold",
            "pred_extraction_target": (
                "(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0))"
            ),
            "fallback_mode": "first_match",
            "aggregation": "max over preds per-sample; harness aggregates mean@k",
        }
