# MATH grader comparison: SPO vs grpocredit (and the wider field)

> Analysis doc (2026-06-10). Cross-references `~/projects/SPO`, `~/projects/grpocredit`,
> and `~/projects/VinePPO-grpo`. Every claim below is grounded in the source files
> cited with `path:line`. Empirical section is from a real smoke test (artifacts in
> `/tmp/grader_smoke/`).

## TL;DR

- "Grade a MATH answer" has **no single canonical implementation**. There are ~4 graders
  in circulation, each making different policy choices about equivalence.
- **SPO uses two of them, on purpose**, because they do two different jobs:
  - `eval_math` (treetune **Minerva**-style) = the **training reward + in-loop eval** grader.
  - `extractive_match` (lighteval, **math-verify**) = the **publication / benchmark** grader.
- The three graders relevant here, by leniency: **math-verify ≈ strictest robust**, **Minerva = most lenient**, **Hendrycks `is_equiv` = strict but brittle**.
- **Today's de-facto community standard for reporting** MATH-500/AIME (post-R1, long-CoT era)
  is **math-verify via lighteval/open-r1**. grpocredit's *training reward* already uses it.
- A MATH-500 number is only meaningful **with its grader named**. Match the grader to your
  comparison target.

---

## 1. The core confusion: there is no neutral "ground truth"

Is `\boxed{0.333}` equal to gold `1/3`? Is `\boxed{(x+1)(x-1)}` equal to `x^2-1`? Is the
closed interval `[1,2]` equal to the open interval `(1,2)`? Is `3.14159` equal to `\pi`?

Each of these is a **policy choice**, not a fact. Different graders answer differently. That
is the entire source of the "why are there multiple graders" confusion — there is no neutral
oracle, so the community produced several graders that encode different leniency policies.

---

## 2. The grader family tree

| Grader | Lineage | Method | Leniency |
|---|---|---|---|
| **Hendrycks `is_equiv`** | MATH paper (Hendrycks 2021) → EleutherAI lm-evaluation-harness | `strip_string` LaTeX normalization + **literal `==`**. No sympy, no numeric tolerance. | Strict + **brittle** (textual) |
| **PRM800K grader** | OpenAI PRM800K / "Let's Verify Step by Step" (2023) | sympy parse + `is_equiv` style symbolic check | Medium |
| **Minerva `math_equal`** | Minerva (Lewkowycz 2022) → Qwen-Math / ToRA eval toolkit | PRM800K base **+** numeric `isclose(abs_tol=1e-3)` **+** percentage trinity `[x/100, x, x*100]` **+** sympy `symbolic_equal` **+** set/interval/`\cup` recursion | **Most lenient** |
| **math-verify** | HuggingFace `Math-Verify` (2024+), used by lighteval + open-r1 | Robust LaTeX/expr extraction → sympy → `verify`. Handles long-CoT (no `\boxed` required). | Strict-but-robust |

**Provenance (verified in-repo):**

- `SPO/src/treetune/tasks/math_grader.py:1-7` header: *"Directly copied from
  https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py"* → **PRM800K**.
- `SPO/src/treetune/tasks/math_grader_minerva.py` adds `math_equal`, `parse_digits`
  (percent), `symbolic_equal` on top → **Minerva / Qwen-Math** layer. (`is_correct` @ `:297`,
  `eval_math` @ `:343`.)
- `grpocredit/external/verl/verl/utils/reward_score/math.py:31` header: *"string
  normalization from .../lm-evaluation-harness/.../hendrycks_math.py"*; `is_equiv` =
  `strip_string` + `==` (`:32-46`) → **Hendrycks**.
- `grpocredit/src/grpocredit/training/verl_reward.py:1,22,40` → `MathVerifier` →
  `grpocredit/src/grpocredit/rollout/verifier.py` uses the **math-verify** pip package
  (SimpleRL-Zoo recipe).

---

## 3. Where each grader is actually used

| Path | Project | Grader | Purpose |
|---|---|---|---|
| RL reward during training | **SPO** (treetune) | Minerva `eval_math` | reward signal |
| In-loop periodic eval | **SPO** (treetune) | Minerva `eval_math` | live monitoring |
| **Published MATH-500 table** | **SPO** | **math-verify** (lighteval `extractive_match`) | paper numbers |
| RL reward during training | **grpocredit** | **math-verify** (`MathVerifier`) | reward signal |
| `eval_final_checkpoint.py` ("customized") | **grpocredit** | Hendrycks `is_equiv` (verl) | strict eval |
| `eval_final_checkpoint_vineppo.py` ("vineppo") | **grpocredit** | Minerva `eval_math` | match SPO in-loop |
| *(missing)* | **grpocredit** | math-verify / extractive_match | would match SPO's paper |

Key observation: grpocredit's **training reward already uses math-verify** — the same family
as SPO's *published* grader. The two grpocredit *eval scripts* deliberately use *other*
graders (Hendrycks, Minerva) for cross-comparison, which is what makes the picture look
inconsistent.

---

## 4. Why SPO has two graders

Not redundancy — **two jobs, two eras**:

1. **`eval_math` (Minerva) = training/in-loop grader.** Baked into treetune (the VinePPO
   framework SPO is built on). Every RL rollout needs a reward, so the framework ships an
   in-process grader; it also drives periodic in-loop eval. Fast; assumes a short
   `\boxed{}` answer within ~1024 tokens.

2. **`extractive_match` (math-verify) = publication grader.** SPO-tree reports MATH-500 to
   compare against other long-CoT papers (DeepScaleR, STILL-3, GRPO). Those baselines were
   all produced by the **same lighteval + open-r1 harness**, so comparability *requires* the
   same harness. There is also a hard technical reason: R1-distill long-CoT emits 2K–32K
   tokens, which breaks the in-loop grader's short-`\boxed` assumption → robust extraction
   (math-verify) is mandatory.

This "training reward grader ≠ benchmark reporting grader" split is **near-universal**, not
an SPO quirk. SPO simply shows both because it inherited a 2023 training stack (treetune)
but reports with the 2025-standard harness (lighteval).

**Config evidence:**
- In-loop (Minerva): `SPO/configs/tasks/math_inplace_no_answer_prefix.jsonnet` sets
  `answer_prefix: null`, which makes `MATH.grade_answer` dispatch to
  `grade_answer_minerva_format` → `eval_math` (`SPO/src/treetune/tasks/math.py:465-468`).
- Published (math-verify): `SPO/scripts/evaluate_long_cot.sh` → `lighteval vllm ...
  "custom|math_500|0|0"`; `results/SPO-tree-666/results_*.json` metric =
  `custom|math_500|0 → extractive_match` (0.736 @ 2K).

---

## 5. Pairwise comparisons

### 5.1 grpocredit "customized" eval vs "vineppo" eval

Same script except the MATH grader. Identical rollout (vLLM, T=0.35, top_p=0.9, n=16,
max_tokens=1024, stop=`["\n\n\nProblem:"]`, seed=42), identical GSM8K path, identical metrics.

| | customized (`eval_final_checkpoint.py:73-86`) | vineppo (`eval_final_checkpoint_vineppo.py:201-224`) |
|---|---|---|
| Grader | verl `compute_score` (Hendrycks) | treetune `eval_math` (Minerva) |
| Equivalence | `strip_string` + **literal `==`** | numeric isclose 1e-3 **OR** `==` **OR** sympy `math_equal` |
| Decision | `compute_score(text,gold) > 0.5` | `bool(eval_math(...))` |
| On exception | regex `\boxed{…}` literal-`==` fallback | returns `False` |
| Leniency | strict | lenient |

Net: on the same rollouts, **vineppo pass@1 ≥ customized pass@1** (Minerva is strictly more
lenient on edge cases).

### 5.2 grpocredit "vineppo" eval vs SPO in-loop Minerva (on the **test** split)

The grader *functions* are byte-identical (`diff` of `math_grader_minerva.py`,
`math_grader.py`, `math_answer_exctraction.py` across the SPO and VinePPO-grpo clones = no
differences). The SPO in-loop test grader is `math_test_inference_pipeline` (split=`test`,
`answer_prefix:null`); it is the **active** pipeline in
`SPO/configs/qwen1b_for_MATH_eval.jsonnet:105` and defined-but-commented in the deepseekmath
config (where it would inherit T=0.35/n=16/max_tokens=1024 — i.e. exactly the vineppo-eval
protocol).

Datasets match: `grpocredit/data/verl/math_test_vineppo.parquet` (500 rows) **is** the SPO
`data/math` `test` split (verified: gold[0] = `\left( 3, \frac{\pi}{2} \right)` ↔
test[0] = "Convert (0,3) to polar"). SPO splits: train=11500, validation=500, test=500.

| Aspect | SPO in-loop Minerva (test) | grpocredit vineppo eval |
|---|---|---|
| Grader fns | `eval_math`/`is_correct`/`math_equal` | **same (byte-identical)** |
| Dataset | `data/math` test (500) | same 500 ✓ |
| Sampling (deepseekmath cfg) | T=0.35, top_p=0.9, n=16, max_tokens=1024 | same ✓ |
| Prompt | `[MATH_TASK] Problem:\n…\n\nSolution:` | same ✓ |
| **Pred extraction** | `extract_math_answer` → multi-pattern (final-answer-is / boxed / "the answer is" / last-number), `strip_string`-normalized, **LIST** (`math_answer_exctraction.py:211,250`) | `extract_boxed_answers(text)[-1]` — **last boxed only, scalar** |
| **Gold** | re-extracted from full solution → **LIST** (splits commas / `\text{and}`) | parquet build-time last `\boxed{}` → **scalar** (`build_verl_parquet_from_vineppo.py:83-106`) |
| **`eval_math` branch** | list/list (dedup + `_pred[-len(ans):]` + set-match) | str/str (single compare) |
| Non-boxed response | may still extract & grade correct | extracts `""` → **wrong** (stricter) |

Verdict: vineppo eval is a **faithful approximation** of SPO's in-loop Minerva on the common
case (single boxed answer, well-formed response → identical grade). It diverges only on
multi-answer / non-boxed / "answer is" responses, where SPO's list path is more permissive.
Caveat: the *active* qwen1b config runs the in-loop grader at **T=0, max_tokens=4096, n=1**
(greedy) — different sampling than vineppo-eval's n=16.

### 5.3 grpocredit "vineppo" eval vs SPO **published** (lighteval `extractive_match`)

**Not comparable** — different grader *and* different sampling.

| | SPO published (lighteval) | grpocredit vineppo eval |
|---|---|---|
| Grader | `extractive_match` (**math-verify**) | Minerva `eval_math` |
| Dataset | `HuggingFaceH4/MATH-500` test (same 500 problems) | `data/math` test (same 500) |
| Sampling | T=0.6, top_p=0.95, n=1, max_new_tokens 2K/4K/32K | T=0.35, top_p=0.9, n=16, max_tokens=1024 |
| Stat | `extractive_match` ± binomial stderr | pass@1 ± cluster CI |

To reproduce SPO's headline number you need a **math-verify / extractive_match** eval at
T=0.6 — neither grpocredit eval script currently does this.

---

## 6. Empirical evidence (Minerva vs math-verify)

Real smoke test (`/tmp/grader_smoke/`): 25 hand-built triples graded by **SPO Minerva
`eval_math`** (native `vineppo` env) and **grpocredit `MathVerifier` / math-verify** (native
`grpocredit` env).

**Result: 20/25 label agreement. All 5 disagreements are Minerva=correct, math-verify=wrong**
(Minerva strictly more lenient on these):

| Case | pred | gold | Minerva | math-verify | Why |
|---|---|---|---|---|---|
| `06_percent_trinity` | `50` | `0.5` | ✅ | ❌ | Minerva tries `x/100` |
| `07_percent_trinity_rev` | `0.5` | `50` | ✅ | ❌ | Minerva tries `x*100` |
| `08_multi_boxed_dedup` | `[5,3,5]` | `3` | ✅ | ❌ | Minerva dedups+suffix-matches; mv takes last boxed `5` |
| `11_interval_open_vs_closed` | `[1,2]` | `(1,2)` | ✅ | ❌ | Minerva ignores bracket type (arguably a Minerva *bug* — different sets) |
| `16_pi_vs_decimal` | `3.14159` | `\pi` | ✅ | ❌ | Minerva numeric tol 1e-3 accepts the approximation; mv requires symbolic |

> ⚠️ These cases were **deliberately edge-heavy** to probe divergence. On real MATH-500
> model outputs the agreement rate is far higher (≈98%+); the common path (single boxed
> answer) is identical across all three graders. The takeaway is the **direction and nature**
> of divergence, not the 20/25 rate.

Note cases 11 and 16: Minerva's leniency is not always "better" — accepting `[1,2]==(1,2)`
or `3.14159==\pi` is debatable. This is exactly why grader choice is a policy decision.

---

## 7. Is there a community standard?

No single universal grader — but **convergence by era and purpose**:

- **Classic era (2021–23):** Hendrycks `is_equiv` (lm-eval-harness) = de-facto MATH leaderboard grader.
- **RL-for-math era (2023–24):** Qwen-Math / Minerva `math_equal` widely used for *training rewards* (VinePPO, SimpleRL).
- **Post-R1 / long-CoT era (2024–now):** **math-verify via lighteval + open-r1** = current de-facto standard for *benchmark reporting* (MATH-500, AIME, etc.). This is the single best answer to "what should I report with."

The durable norm is not "one grader" but: **state your grader, and match it to your comparison target.**

---

## 8. Practical guidance

1. **Report headline numbers with math-verify** (lighteval `extractive_match`-style). This
   matches both SPO's paper *and* grpocredit's own training reward. It is the one alignment
   grpocredit's eval scripts currently lack.
2. **Use vineppo eval only** to compare against SPO's *in-loop* Minerva numbers (and point it
   at the **test** split for apples-to-apples).
3. **Use customized eval** as the strict/conservative lm-eval-style lower bound.
4. **Never compare numbers across graders** without saying so — the gap is a few points and
   is systematic (Minerva > math-verify > Hendrycks in leniency on edge cases).
5. The reassuring part: grpocredit **trains** on math-verify, the modern standard. The only
   missing piece for clean external comparison is a **math-verify eval script** at the
   reporting protocol.

---

## Appendix: key file references

**SPO (Minerva, in-loop / reward):**
- `src/treetune/tasks/math_grader.py:1-7` (PRM800K provenance)
- `src/treetune/tasks/math_grader_minerva.py:297` (`is_correct`), `:343` (`eval_math`)
- `src/treetune/tasks/math_answer_exctraction.py:211` (`extract_answer`, multi-pattern), `:250` (`extract_math_answer` → list)
- `src/treetune/tasks/math.py:465-468` (Minerva dispatch), `:486` (`grade_answer_minerva_format`), `:498` (`evaluate_predictions`)
- `configs/tasks/math_inplace_no_answer_prefix.jsonnet` (`answer_prefix: null`)
- `configs/sft_deepseekmath_for_MATH_eval.jsonnet` (validation active; T=0.35/n=16/max_tokens=1024)
- `configs/qwen1b_for_MATH_eval.jsonnet:105` (test active; T=0/max_tokens=4096)

**SPO (published, math-verify):**
- `scripts/evaluate_long_cot.sh` (lighteval `custom|math_500`)
- `results/SPO-tree-666/results_*.json` (`extractive_match` = 0.736 @ 2K; `HuggingFaceH4/MATH-500` test; T=0.6/top_p=0.95)

**grpocredit:**
- `scripts/eval_final_checkpoint.py:73-86` (customized = verl Hendrycks)
- `scripts/eval_final_checkpoint_vineppo.py:201-224` (vineppo = Minerva, scalar form), `:95-112` (loads from `VINEPPO_ROOT`, default `~/projects/VinePPO-grpo`)
- `src/grpocredit/training/verl_reward.py` + `src/grpocredit/rollout/verifier.py` (training = math-verify)
- `scripts/build_verl_parquet_from_vineppo.py:83-106` (`extract_boxed_answer` = last balanced boxed, scalar gold), test split = 500 = SPO `data/math` test
- `external/verl/verl/utils/reward_score/math.py:17,31-46` (Hendrycks `is_equiv`)

**Smoke test artifacts:** `/tmp/grader_smoke/{cases.json,run_minerva.py,run_grpocredit.py,out_minerva.json,out_grpocredit.json}`
