#!/usr/bin/env python
r"""Parity / self-test for the math_verify (extractive_match) grader.

Run in an env that has math_verify (the reference `llmagent` env, or the
grpocredit training env used on the cluster)::

    python scripts/test_grader_parity.py

Asserts the grader reproduces expected extractive_match labels on a battery of
MATH-representative cases (integer / AIME golds, non-boxed recovery, open-vs-
closed intervals, sign, fraction reduction, coordinates), and that grading the
full boxed *solution* (open-r1's actual gold = ``line["solution"]``) gives the
identical result to grading our ``\boxed``-wrapped bare parquet answer. Exits
nonzero on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _math_verify_grader import MathVerifyGrader  # noqa: E402

# (pred_text, bare_gold, expected_label)
CASES = [
    (r"final answer is $\boxed{\frac{1}{2}}$.", r"\frac{1}{2}", True),
    (r"</think> The answer is \boxed{0.5}.", r"\frac{1}{2}", True),      # 0.5 == 1/2
    (r"reasoning... = 42 at the very end (no box)", r"42", True),        # non-boxed recovery, int gold
    (r"so the answer is \boxed{204}", r"204", True),                    # AIME integer
    (r"so the answer is \boxed{205}", r"204", False),                   # AIME wrong
    (r"\boxed{\left(3,\frac{\pi}{2}\right)}", r"\left( 3, \frac{\pi}{2} \right)", True),
    (r"The final answer is \boxed{[1,2]}", r"(1,2)", False),            # closed vs open interval
    (r"\boxed{2/4}", r"\frac{1}{2}", True),                            # fraction reduction
    (r"\boxed{-\frac{2}{3}}", r"\frac{2}{3}", False),                  # sign sensitivity
    (r"Therefore the final answer is $\boxed{7}$.", r"7", True),
    (r"I think it is 8.", r"7", False),
    (r"the value is \boxed{x+1}", r"1+x", True),                       # symbolic simplify
]

# grade(full boxed solution) must equal grade(wrapped bare answer)
SOLUTION_PARITY = [
    (r"the answer is \boxed{204}",
     r"Adding gives 204. Therefore, the final answer is: $\boxed{204}$.",
     r"204"),
    (r"\boxed{\left(3,\frac{\pi}{2}\right)}",
     r"Convert. Therefore, the final answer is: $\boxed{\left(3,\frac{\pi}{2}\right)}$.",
     r"\left( 3, \frac{\pi}{2} \right)"),
]


def _mock_loop_sanity(g: MathVerifyGrader) -> int:
    """Exercise the harness mean@k reduction with a mock (question -> n samples)."""
    # Two questions, gold=1/2; q0: 3/4 samples correct, q1: 1/4 correct.
    gold = r"\frac{1}{2}"
    gp = g.parse_gold(gold)
    q0 = [r"\boxed{1/2}", r"\boxed{0.5}", r"\boxed{2/4}", r"\boxed{9}"]
    q1 = [r"\boxed{9}", r"\boxed{9}", r"\boxed{9}", r"\boxed{1/2}"]
    p0 = sum(g.grade(t, gold, gold_parsed=gp) for t in q0) / len(q0)
    p1 = sum(g.grade(t, gold, gold_parsed=gp) for t in q1) / len(q1)
    pass1 = (p0 + p1) / 2
    ok = abs(p0 - 0.75) < 1e-9 and abs(p1 - 0.25) < 1e-9 and abs(pass1 - 0.5) < 1e-9
    print(f"  {'ok ' if ok else 'FAIL'} mean@k: q0={p0:.3f} q1={p1:.3f} pass@1={pass1:.3f} (exp .75/.25/.50)")
    return 0 if ok else 1


def main() -> int:
    g = MathVerifyGrader()
    fails = 0
    print("config:", g.describe())

    print("\n[label cases]")
    for pred, gold, exp in CASES:
        got = g.grade(pred, gold)
        ok = got == exp
        fails += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} exp={exp!s:<5} got={got!s:<5} gold={gold!r}")

    print("\n[solution-vs-bare gold parity]")
    for pred, sol, bare in SOLUTION_PARITY:
        a = g.grade(pred, sol)
        b = g.grade(pred, bare)
        ok = a == b
        fails += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} solution={a!s:<5} bare={b!s:<5} bare_gold={bare!r}")

    print("\n[mean@k reduction sanity]")
    fails += _mock_loop_sanity(g)

    print(f"\n{'PASS' if fails == 0 else 'FAIL'}: {fails} mismatch(es)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
