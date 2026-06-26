# SPO-tree (Qwen-1.5B) vs grpocredit VoI — training comparison (published 1.5B regime)

**Both fine-tune the SAME base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.** This is the
**published-SPO setting** (long-CoT, MATH train → MATH500 / AIME24 at 2K/4K/32K eval tiers),
i.e. the apples-to-apples head-to-head that the 7B sibling doc deliberately *avoided* (see
`COMPARE_spo_tree_vs_grpocredit_voi.md` Appendix). The two docs are complementary:

| | base | knob philosophy | regime |
|---|---|---|---|
| `COMPARE_spo_tree_vs_grpocredit_voi.md` (7B) | deepseekmath-7B-SFT | each method at its **own** optimum | short-CoT (resp 1024) |
| **this doc (1.5B)** | **R1-Distill-Qwen-1.5B** | **knob-MATCHED to SPO** (Table-1-comparable) | **long-CoT (resp 4096; 2K/4K/32K eval)** |

- **SPO-tree (1.5B)** — `SPO/configs/polIter_qwen1b_spo_tree_MATH.jsonnet` → imports
  `polIter_qwen1b_spo_chain_MATH.jsonnet`. All values **resolved from the shipped jsonnet**.
  Framework: **treetune + DeepSpeed ZeRO-0**, **SINGLE GPU**.
  Launch: `scripts/sbatch_spo_tree_qwen1b_MATH.sh` → `launch_server2_spo.sh spo_tree MATH` (1×H200).
- **VoI (1.5B)** — grpocredit **planned experiment 3A** ("SPO direct apples-to-apples"),
  `grpocredit/more_dataset_plan.md` §5.1. Framework: **verl + FSDP2 + Ray**.
  ⚠️ **Not yet run** — a qwen VoI launcher now exists
  (`grpocredit/scripts/launch_verl_grpo_voi_v041_p4_mb8_T06unifAclip2_qwen1p5b_0615.sh`) but no
  checkpoints exist yet; config knobs below that cite the 0615 launcher are **pinned by it** (no
  longer [TBD]), while the rest is plan-pinned + algorithm carryover from the 7B VoI run.

> ⚠️ **Status.** SPO-tree-1.5B is the **upstream-validated, published** SPO config
> (`RUN_spo_tree_qwen1b_MATH_seed42.md`: MATH500 0.736/0.828/0.848 @ 2K/4K/32K; headline
> checkpoint ≈ iter 400, not iter 1000). The **VoI-1.5B side is PLANNED**, not run; every VoI
> value is tagged by provenance below.

> 🔧 **Local edits — SPO 1.5B path is near-upstream (one bugfix).** The qwen1b SPO path is
> almost upstream-clean, with **one deliberate local config edit**:
> `polIter_qwen1b_spo_chain_MATH.jsonnet` `max_question_length` **200 → 512** (bugfix; see §2 /
> §2.1). Upstream shipped **200** — the GSM8K value (cf. `polIter_qwen05b_spo_chain_gsm8k.jsonnet`)
> carried over to MATH, which silently drops **~12% of MATH train**; every other MATH config is
> larger (base default **1512** `episode_generators/math_episode_generator.jsonnet`, the 7B run
> inherits **1512**, `polIter_qwen1_5b_base_spo_chain_MATH.jsonnet` uses **512**). Otherwise every
> `configs/polIter_qwen1b_*.jsonnet`, `configs/qwen1b_for_MATH_eval.jsonnet`, and
> `src/treetune/main.py` are used **unmodified** vs upstream `AIFrameResearch/SPO@1e64f0c`. The
> other local additions are two scaffolding scripts + a `data/` symlink
> (`RUN_spo_tree_qwen1b_MATH_seed42.md` §0–§1).

### Provenance legend (VoI column)
- **[3A]** — pinned by `grpocredit/more_dataset_plan.md` §5.1 (the planned apples-to-apples card).
- **[7B→]** — carried over from the resolved 7B VoI launcher
  `grpocredit/scripts/launch_verl_grpo_voi_v041_p4_mb8_T08unifAclip2.sh` (algorithm internals
  expected to transfer unchanged to 1.5B).
- **[TBD]** — not yet configured; **must be set when the VoI-Qwen launcher is created**
  (especially the long-CoT length budget and GPU topology).

---

## 1. Model
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (`polIter_qwen1b_spo_chain_MATH.jsonnet:1`) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` **[3A]** (same base; no own SFT) |
| params trained | full fine-tune | full fine-tune **[3A]** |
| chat template | R1-Distill format: `<｜begin▁of▁sentence｜><｜User｜>Solve the following math problem … Think step by step…\n\n{query}<｜Assistant｜><think>\n` (`prompt_library/qwen_MATH.jsonnet:2`) | **SPO's literal `qwen_MATH.jsonnet` template** injected via `custom_chat_template` (`R1_CHAT_TEMPLATE`, `launch_…qwen1p5b_0615.sh:252,271`) — **byte-identical to SPO; token-id parity VERIFIED (single BOS=151646, identical query ids — both sides encode add_special_tokens=False; see note below)** **[0615 launcher; supersedes the earlier `{{ m.content }}` passthrough that was [TBD]]** |
| remove-padding | n/a (treetune) | `use_remove_padding=True` **[7B→]** |
| grad checkpointing | True (`trainers/ppo_MATH.jsonnet:81`) | True **[7B→]** |
| dropout / attn | `disable_dropout=true`, flash-attn-2 (`ppo_MATH.jsonnet:16-18`) | flash-attn-2 (verl default) |

> **Template caveat (resolved).** SPO trains on the R1-Distill `<think>` chat format; the 7B VoI
> launcher uses a bare concat template. The qwen 0615 launcher resolves this for the 1.5B
> head-to-head — it injects SPO's exact `qwen_MATH.jsonnet` template via `custom_chat_template` on
> the bare-problem parquet, rendering a prompt byte-identical to SPO's (the reasoning-trace format
> matters for long-CoT credit assignment).
>
> **Verified (2026-06-21).** The rendered prompt is byte-identical **and** tokenizes to identical ids
> with a **single BOS** on both sides. The tokenizer ships `add_bos_token=true` (BOS `<｜begin▁of▁sentence｜>`
> =151646), but **both** pipelines encode with `add_special_tokens=False` (verl `rl_dataset.py:258,297`;
> treetune `episode_generator_with_reward_function.py:124` with the manual BOS-prepend commented out at
> `:149`), so no second BOS is added — the literal `<｜begin▁of▁sentence｜>` in `qwen_MATH.jsonnet:2` is the
> only one. (Counterfactual: `add_special_tokens=True` would double it → `[151646, 151646, …]`.)

## 2. Data & sequence lengths
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| train data | treetune `math` train split (HF arrow) — **11,500 rows** | `math_train_bareproblem.parquet` (bare MATH-v2 problem in `content`; SPO prompt applied at launch via `custom_chat_template`) — **11,496 rows** (vs SPO's 11,500; **4 dropped** for unextractable `\boxed{}` in `build_verl_parquet_from_vineppo.py:223`; same VinePPO source & order, row 0 matches) **[0615 launcher:147; ⚠️ was `math_train_vineppo.parquet`/`[MATH_TASK]`]** |
| in-loop val/eval data | MATH **test** (`qwen1b_for_MATH_eval.jsonnet:105` — `math_test_inference_pipeline` active) | `math_test_bareproblem.parquet` = **MATH-500 test** (`0615 launcher:148`) — **bit-identical to SPO test: 500 rows, identical ground-truth fingerprint (verified)**; in-loop eval on TEST to match SPO **[supersedes the earlier 7B `math_val.parquet` plan]** |
| **max response length** | **4096 requested** (`max_tokens`, `polIter_qwen1b_spo_chain_MATH.jsonnet:81`) → **~2048 effective** as-shipped, throttled by `model_context_size=2048` via `min(4096, 2048−prompt)` (`expansion.py:401-408`; see §2.1 footgun) | **≈4096 [TBD]** — VoI currently 1024; must raise to match SPO long-CoT (user-confirmed) |
| model_context_size | **2048** as-shipped (`chain:88`); published ckpt = **4096** (`RUN_…md` §10.7) | **[TBD]** — set 2K→4K tiers to mirror SPO's schedule |
| max prompt length | `max_question_length=512` (**local bugfix**; upstream shipped 200, a GSM8K carryover — see §2.1) | `max_prompt_length=512` (matched to the SPO bugfix; VoI launcher set 200→512) |
| max_sequence_length | null (`chain:48`) | n/a — verl caps at prompt+response |
| overlong handling | tree handles; `fill_missing_episodes=true` (`chain:60`) | `filter_overlong_prompts=True`, `truncation=error` **[7B→]** |
| shuffle / replacement | `dataset_shuffle_on_each_iteration=true` (`chain:57`), `sample_with_replacement=true` (`chain:59`) | `data.shuffle=True` **[7B→]** |
| append bos/eos | `append_bos_to_query=false`, `append_eos_to_response=false` (`chain:54-55`) | chat-template managed |
| data seed | 42 | `+data.seed=42` **[7B→]** |

> **SPO uses the SAME MATH data across model sizes (1.5B ⟷ 7B).** Both the 1.5B
> (`polIter_qwen1b_spo_chain_MATH.jsonnet:3`) and the 7B (`polIter_deepseekSft2_ppo_MATH.jsonnet:14`)
> configs take their **train** data from the *identical* task wrapper
> `tasks/math_inplace_no_answer_prefix.jsonnet` → `tasks/math.jsonnet` (`dataset_dict_path: 'data/math'`)
> — the same MATH `DatasetDict` in the same repo, consumed from the same pool with `dataset_sample_with_replacement=true` over 1000 iters. The
> per-iter draw (`num_dataset_samples_per_iteration`) **differs by size**: the **1.5B SPO-tree draws 32
> q/iter** (→512 episodes; `polIter_qwen1b_spo_tree_MATH.jsonnet:3,1`), the **7B SPO-tree 16/iter** (→1024;
> `polIter_deepseekSft2_spo_tree_MATH.jsonnet:3,1`); the chain configs use 64 (512/8). So the **data
> source** is size-invariant, not the per-iter draw count.
> **Validation/test** also come from that same `data/math` DatasetDict: both eval configs
> (`qwen1b_for_MATH_eval.jsonnet`, `sft_deepseekmath_for_MATH_eval.jsonnet`) define the same
> `math_train` / `math_validation` / `math_test` pipelines over it. ⚠️ The only delta is *which*
> in-loop split is **active** — the 1.5B evals on **test** (`qwen1b_for_MATH_eval.jsonnet:105`) while
> the 7B evals on **validation** (`sft_deepseekmath_for_MATH_eval.jsonnet:114`): same data pool,
> different active split. So SPO's data pipeline is **size-invariant** — going 7B→1.5B swaps only the
> base model, not the train/val/test data.

> **Sampling regime — intended difference (same data pool, different draw/rollout).** SPO draws **32
> questions/iter i.i.d. _with replacement_** (`num_dataset_samples_per_iteration=32`,
> `dataset_sample_with_replacement=true`, `chain:57-59`), expanding each into a **6-6-6 tree** → segment-level
> episodes with MC advantages. verl draws **32 prompts/step via shuffled epochs _without replacement_**
> (`data.shuffle=True`, `train_batch_size=32`) and rolls out **flat groups of n=8** (`rollout.n=8`) with the
> `voi_td_sparse_pivot` advantage. Both run 1000 steps (~32k draws ≈ 2.78 epochs over 11.5k) — coverage is
> **stochastic** on the SPO side, **near-uniform** on the verl side. This is intrinsic to tree-SPO vs GRPO/VoI;
> document it, don't "fix" it.

## 2.1 Token / context / sequence-length budget — spo-chain vs spo-tree vs VoI
*(the long-CoT length knobs, broken out per SPO variant; expands the length rows of §2)*

**All length knobs are identical for spo-chain and spo-tree** — `polIter_qwen1b_spo_tree_MATH.jsonnet`
imports the chain config and only swaps in the hybrid 6-6-6 tree (`M=600`, `max_depth=3`,
`tree:8-26`); it overrides **no** length knob. Both inherit `max_tokens=4096` (`chain:81`),
`model_context_size=2048` (`chain:88`), `max_sequence_length=null` (`chain:48`),
`max_question_length=512` (**local bugfix**; upstream shipped 200).

| length knob | SPO-chain 1.5B | SPO-tree 1.5B | VoI 1.5B (planned) |
|---|---|---|---|
| **response / token length** | **4096 req / ~2048 eff** (`max_tokens` throttled by `model_context_size=2048`) | **4096 req / ~2048 eff** (inherited) | **≈4096** **[TBD]** (raise from 1024) |
| **context length** | **2048** shipped / **4096** published | **2048 / 4096** (inherited) | **2K→4K tiers** **[TBD]** |
| **max_sequence_length** | **null** (no extra cap) | **null** (inherited) | n/a — verl prompt+response cap |
| max prompt length | **512** (`max_question_length`; local bugfix from 200) | **512** (inherited) | **512** (VoI launcher, matched) |

- **The shipped-vs-published context gap is the single biggest SPO footgun** (`RUN_…md` §10.7):
  as-shipped `model_context_size=2048` bounds generation to `min(4096, 2048 − prompt)` even though
  `max_tokens=4096`. To reproduce the **0.828@4K** headline you must locally set
  `model_context_size:4096`. The repo ships **no** 2K→4K schedule config for MATH.
- **Effective generation.** SPO caps new tokens at `min(max_tokens, model_context_size − prompt)`
  (`src/treetune/inference_strategies/tree_inference/expansion.py:401-408`). VoI (verl) generates up
  to `max_response_length` with prompts hard-capped and overlong prompts dropped.
- **Matched vs not.** This regime is the whole point of the 1.5B doc: VoI must be **reconfigured for
  long-CoT** (response 1024→4096, context 2K/4K tiers) to be comparable — it is *not* yet so
  configured. Until then the columns are **not** length-matched.

## 3. Algorithm / advantage
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| estimator | **SPO-tree** segment-MC | **`voi_td_sparse_pivot`** **[7B→]** |
| tree / probes | branch **6-6-6**, `max_depth=3` (`tree:16-24`), **M=600** (`tree:15`); value-MC `samples=4` (`chain:14`) | sparse VoI-selected TD pivots; MC probe `budget_per_trajectory=5`, `rollouts_per_probe=8`, `signal_key=h_fwd_max`, `h_fwd_k=8`, `w_pos_shape=uniform`, `alpha_voi=1.0`, `james_stein_tau=4.0`, `a_base_clip=2.0` **[7B→]** |
| segment advantage | `A = V(seg_end) − V(seg_start)` from MC cutpoint values; within-segment tokens share it (`math_episode_generator_with_mc_advantages.py:420-435`) | per-token TD residual at VoI-chosen pivots (verl recipe) |
| token-level mask | **probability mask** `use_prob_mask=true` (`chain:143`): tokens with `prob ≥ 0.9` get **zero advantage** + dropped from loss (`ppo_trainer.py:1057-1059`, `mc_advantages.py:437-439`) | **prob-mask OFF [3A]** (confound control) — a separate **VoI + prob-mask** stacking ablation is planned |
| learned critic | none (`critic_model=null`, `chain:150`) | none (`reward_model.enable=False`) **[7B→]** |
| γ (gamma) | 1 (`ppo_MATH.jsonnet:57`) | 1 (estimator-internal) **[7B→]** |
| λ (lam) | 1 — pure MC (`trainers/lam1.jsonnet`) | — (TD pivots) **[7B→]** |
| advantage whitening | `whiten_advantages=true`, `whiten_rewards=false` (`ppo_MATH.jsonnet:63-64`) | estimator-internal (verl) **[7B→]** |
| score scaling/norm | `use_score_scaling=false`, `use_score_norm=false` (`ppo_MATH.jsonnet:51-52`) | — |

> **The one intentional method delta.** Everything in §5–§6 is **knob-matched** for this doc; the
> credit-assignment is the variable under test: **SPO tree segment-MC + prob-mask** vs **VoI
> TD-sparse-pivot (no prob-mask)**. The confound-control design (`more_dataset_plan.md` §5.1) runs
> GRPO/VoI *without* prob-mask and SPO *with* it (mirroring SPO's published St arm), plus a
> VoI+prob-mask stack — isolating the credit-assignment win from the masking win.

## 4. Rollout / generation
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| engine | vLLM (single per-GPU server) | vLLM (`tensor_model_parallel_size` TBD for 1.5B) **[7B→]** |
| **train temperature** | **0.6** (`chain:12,79`) | **0.6 [3A-match]** (7B used 0.8) |
| **train top_p** | **1.0** (`chain:80`) | **[TBD]** (7B used 0.95) |
| top_k | (default) | -1 **[7B→]** |
| max_tokens | 4096 (`chain:81`) | ≈4096 **[TBD]** |
| rollouts per question | tree 6-6-6 expansion (M=600 cap) | `rollout.n` **[TBD]** (7B used 8, flat) |
| stop / logprobs | `"\n\n\nProblem:"` (unused), `logprobs=0` (`chain:82-83`) | (eos) |
| gpu_memory_utilization | `auto` (`chain:39`), `min_avail=4096MB` (`chain:40`) | **[TBD]** (7B used 0.6) |
| swap_space / max_num_seqs / prefix-cache | 32G / 512 / true (`chain:43-45`) | verl-managed **[7B→]** |
| mem release | `wait_until_memory_release=true` (`chain:41`) — vLLM frees VRAM before optim step (single-GPU) | n/a (verl colocates / separate GPUs) |
| **val/eval sampling (in-loop)** | **T=0 greedy, n=1, top_p=0.9, max_tokens=4096, ctx 2048, seed 42** (`qwen1b_for_MATH_eval.jsonnet:1-2,21-23,30,44`) | **T=0.35, top_p=0.9, n=16** **[7B→]** (verl in-loop) |
| **headline eval (offline)** | lighteval+vLLM `custom\|math_500`, **2K/4K/32K** tiers, T=0.6/top_p=0.95 @4K/32K, **T=0 greedy @2K** (`RUN_…md` §7) | lighteval+vLLM, **same 2K/4K/32K** tiers + protocol **[3A]** |

## 5. Optimization (knob-MATCHED)
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| **learning rate** | **1e-6** (`ppo_MATH.jsonnet:72`) | **1e-6** **[3A]** (matched; 7B VoI used 5e-7) |
| warmup | `warmup_ratio=0.03` (`ppo_MATH.jsonnet:74`) | `warmup_style=constant`, ratio 0.0 **[7B→]** (⚠️ mismatch vs SPO's 0.03 — reconcile) |
| weight decay | 0 (`ppo_MATH.jsonnet:73`) | 0 **[7B→]** |
| clip ratio | 0.2 (value clip 0.2) (`ppo_MATH.jsonnet:60-61`) | `clip_ratio=0.2` **[7B→]** |
| grad clip | `max_grad_norm=1.0` (`ppo_MATH.jsonnet:76`) | `grad_clip=1.0` **[7B→]** |
| entropy coef | — (`report_entropy=false`, `chain:160`) | `entropy_coeff=0.0` **[7B→]** |
| PPO epochs / iter | 1 (`chain:146`, `tree:29`) | 1 **[7B→]** |
| precision | bf16 (`ppo_MATH.jsonnet:82`) | bf16 **[7B→]** |

## 6. KL control (knob-MATCHED)
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| placement | in **loss** (`kl_penalty_loss_type=control_variate`, `trainers/klLoss.jsonnet:4`) | in **loss** (`use_kl_loss=True`) **[7B→]** |
| KL in reward | no | `use_kl_in_reward=False` **[7B→]** |
| **KL coef** | **1e-4** (`init_kl_coef`, `trainers/refKl0.0001.jsonnet:4`) | **1e-4** **[3A]** (matched; 7B VoI used 1e-3) |
| KL estimator | control_variate (clip_min 0, clip_max 1e8) (`klLoss.jsonnet:5-6`, `refKl0.0001.jsonnet:5`) | `low_var_kl` **[7B→]** |
| adaptive KL | `adap_kl_ctrl=false` (`ppo_MATH.jsonnet:54`) | fixed **[7B→]** |

## 7. Batching / throughput
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| **questions / iter** | **32** (`num_dataset_samples_per_iteration`, `tree:3,10`) | **[TBD]** (`train_batch_size`; 7B used 64) |
| **episodes / iter** | **512** (`num_episodes_per_iteration`, `chain:7`,`tree:27`) | **[TBD]** (7B: 64 prompts × `rollout.n=8` = 512) |
| segments / question | **~258 generated** → **≤32 kept/iter** (replay buffer) → **≈16/q** after the 512 cap (see note) | **[TBD]** (7B: 8 flat trajectories) |
| **global optimizer batch** | `target_train_batch_size=128` (`tree:31`) | **128** **[3A]** ("target batch 128"; note unit: SPO counts **episodes/segments**, verl counts **prompts/tokens**) |
| per-device train batch | 2 (`tree:32`) | `ppo_micro_batch_size_per_gpu` **[TBD]** (7B used 2) |
| grad accumulation | auto → **64** (128 ÷ (2 × **1 GPU**)) | dynamic (token-packed) **[7B→]** |
| optimizer steps / iter | 512 ÷ 128 = **4** | **[TBD]** (verl, dynamic) |
| dynamic batching | no (fixed counts) | `use_dynamic_bsz=True` **[7B→]** |

> **What a "segment / episode" is here (NOT a trajectory or a leaf).** An SPO-tree training
> example is a **segment-edge**: `query = text up to the parent node`, `response = one tree hop`,
> `advantage = V(child) − V(parent)` (`tree_episode_generator.py:128-159`). Counting:
> a 6-6-6 tree has **216 leaves / ~258 non-root nodes ⇒ ~258 candidate segments/question**; only
> **nonzero-advantage** segments are kept (`only_adv_greater_than_zero=True`); the replay buffer then
> samples **≤32/question/iter** (`hybrid_episode_generator.py:54-55`, 8-iter sliding window); finally
> the merged batch is capped to `num_episodes_per_iteration=512`
> (`on_policy_episode_generator.py:423-437`). So **~16/question is only a post-cap average**, not a
> fixed per-question quota — the hard per-question cap is **32**.

> **Batching is NOT matched and largely can't be** (framework delta). SPO uses fixed-count
> segment batching on a single GPU (grad-accum 64); verl uses token-dynamic batching across
> ranks. The plan matches the **global optimizer batch (128)** as the comparable anchor; the rest is
> framework-determined.

## 8. Distributed / sharding / engine
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| actor sharding | **DeepSpeed ZeRO-0** (no sharding/offload — 1.5B fits one GPU) (`chain:156`, `deepspeed/zero_0.jsonnet:3`) | **FSDP2** **[7B→]** (⚠️ overkill for 1.5B; ZeRO-0-equivalent on few GPUs) |
| reference model | on CPU (`move_reference_model_to_cpu=true`, `chain:157`), ds-auto | FSDP2, `param_offload=True` **[7B→]** |
| rollout placement | **single GPU**, vLLM + trainer time-share via `wait_until_memory_release` | vLLM colocated / TP **[TBD]** |
| precision | bf16 | bf16 **[7B→]** |
| orchestration | deepspeed launcher (`--include localhost:0`) | Ray (`ray_init.num_cpus=32`) **[7B→]** |

## 9. Schedule / eval / checkpoint / logging
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| total steps/iters | 1000 (`chain:10`) | **[TBD]** (7B used `total_training_steps=1000`) |
| headline checkpoint | ≈ **iter 400** (published `Qwen1.5B-MATH-C4096-400`; *not* iter 1000) (`RUN_…md` §10.7-8) | **[TBD]** (best-by-val) |
| in-loop eval | MATH **test**, T=0 greedy n=1 (`qwen1b_for_MATH_eval.jsonnet`) | MATH-val, T=0.35 n=16 **[7B→]** |
| **headline eval** | lighteval `math_500` @ **2K/4K/32K** (`scripts/evaluate_long_cot.sh`, `RUN_…md` §7) | **same harness + tiers** **[3A]** |
| save cadence | `save_steps=5`, `checkpoint_keep_steps=10` → retains every iter % 10 == 0 (`chain:167-168`, `ppo_trainer.py:2560-2565`) | `save_freq` **[TBD]** (7B: 100) |
| logging | `logging_steps=1` (`ppo_MATH.jsonnet:84`), `episodes_cloud_log_steps=50` (`chain:137`) | `logger=[console,wandb]` **[7B→]** |
| wandb | entity `suesie`, project `spo-math` | project `grpocredit-grpo-verl` **[7B→]** |

## 10. Compute / resources (NOT matched — fundamental method/framework split)
| | SPO-tree 1.5B (sbatch) | VoI 1.5B (planned 3A) |
|---|---|---|
| account / qos | h200_mrs_2 / h200_mrs_2_high | h200_mrs_2 / h200_mrs_2_high **[7B→]** |
| **GPUs** | **gpu:h200:1** (single-GPU method) | **[TBD]** — verl+Ray is multi-GPU; for 1.5B likely 1–2 H200, but the orchestration differs fundamentally from SPO's single-GPU time-share |
| cpus-per-task | 24 | **[TBD]** (7B used 96) |
| nodes / ntasks | 1 / 1 | 1 / 1 |
| exclusive | **no** (leaves the node's other 7 GPUs free) | **[TBD]** |
| host RAM | `--mem=256G` | **[TBD]** |
| walltime cap | 3-00:00:00 (1000 iters single-GPU is multi-day) | **[TBD]** |

> **SPO-tree is a single-GPU method by design** (paper: 1× A100 80GB; `RUN_…md` §2): the vLLM
> episode-generation server and the trainer **share one GPU** (`wait_until_memory_release=true`).
> verl+FSDP2+Ray does not have a faithful single-GPU mode — so unlike the 7B doc (which matched
> 8×H200 on both sides), **compute cannot be matched here**. Compare by **time-to-accuracy** /
> **GPU-hours-to-accuracy**, not by iterations or by node count.

## 11. Reward / environment
| | SPO-tree 1.5B | VoI 1.5B (planned 3A) |
|---|---|---|
| reward | `math_reward_function`, `penalize_unfinished_response=true`, penalty 0.0 (`chain:27-31`) | custom `verl_reward.compute_score` **[7B→]** |
| verifier | treetune MATH grader | `\boxed{}` registry + R1 `<answer>` extractor (grpocredit README §verifier) **[7B→]** |
| conda env | `spo`/`vineppo` (torch 2.1.2 / vllm 0.4.0.post1 / ds 0.14.1) | `grpocredit-verl-pinned` **[7B→]** |
| seed | 42 | 42 **[3A]** |

---

## 12. Bottom line
- **Matched (by design — the whole point of the 1.5B card, `more_dataset_plan.md` §5.1):** base
  model (**R1-Distill-Qwen-1.5B**), train data (MATH train), **lr=1e-6**, **β_KL=1e-4** (KL-as-loss),
  **target optimizer batch=128**, clip 0.2, grad-clip 1.0, 1 PPO epoch, γ=1, wd 0, critic-free,
  seed 42, and the **eval harness/tiers** (lighteval `math_500` @ 2K/4K/32K).
- **The single variable under test:** credit assignment — **SPO tree segment-MC + prob-mask (ρ=0.9)**
  vs **VoI TD-sparse-pivot (prob-mask OFF)**. Confound-controlled: prob-mask ON for the SPO arm only,
  with a **VoI + prob-mask** stacking ablation to separate the credit-assignment win from the
  masking win.
- **Framework deltas (unavoidable):** treetune + DeepSpeed-ZeRO0 **single-GPU** vs verl + FSDP2 + Ray
  **multi-GPU**; fixed-count episode/segment batching vs token-dynamic batching. Compute is **not**
  matched (SPO is single-GPU by design) → compare on **GPU-hours-to-accuracy**.
- **⚠️ Open work before this is runnable (all [TBD] above):**
  1. **Build a VoI-Qwen launcher** (none exists today).
  2. **Reconfigure VoI for long-CoT** — response 1024→≈4096, context 2K→4K tiers (matches SPO; the
     largest single gap).
  3. **Adopt the R1-Distill `<think>` chat template** on the VoI side.
  4. **Reconcile warmup** (SPO 0.03 vs VoI constant-0.0) and pick the VoI GPU topology.
  5. Decide questions/episodes-per-iter so the **128 optimizer batch** is the comparable anchor.

---

## Appendix — relationship to the 7B doc & published SPO numbers
- **7B doc** (`COMPARE_spo_tree_vs_grpocredit_voi.md`) = same-base **deepseekmath-7B**, short-CoT
  (resp 1024), each method at its **own** optimum, **8×H200 matched**. It exists because there is no
  published 7B SPO number — a structurally-correct but unvalidated 7B SPO instantiation.
- **This doc** = the **published 1.5B regime**, long-CoT, **knob-matched** to be **SPO-Table-1
  comparable**. SPO-tree-666 targets (`RUN_spo_tree_qwen1b_MATH_seed42.md`): MATH500
  **0.736 / 0.828 / 0.848** @ 2K/4K/32K; the published checkpoint `gyr66/spo-tree-666-qwen1.5B-math`
  = `Qwen1.5B-MATH-C4096-400` (context 4096, **iter ~400**). GRPO baseline ≈ 0.62/0.752/0.84.
- **Eval-only replication probes** (`more_dataset_plan.md` §6): **P4** `gyr66/spo-tree-666-qwen1.5B-math`
  and **P5** `gyr66/grpo-qwen1.5B-math` confirm the harness reproduces SPO Table-1 before trusting
  the trained VoI number.

*Sources — SPO: resolved `SPO/configs/polIter_qwen1b_spo_tree_MATH.jsonnet` →
`polIter_qwen1b_spo_chain_MATH.jsonnet` (+ `trainers/ppo_MATH.jsonnet`, `lam1`, `refKl0.0001`,
`klLoss`; `qwen1b_for_MATH_eval.jsonnet`; `prompt_library/qwen_MATH.jsonnet`; `deepspeed/zero_0.jsonnet`)
via the shipped jsonnet; `scripts/sbatch_spo_tree_qwen1b_MATH.sh`; `RUN_spo_tree_qwen1b_MATH_seed42.md`.
VoI: `grpocredit/more_dataset_plan.md` §5.1 (planned 3A) + carryover from
`grpocredit/scripts/launch_verl_grpo_voi_v041_p4_mb8_T08unifAclip2.sh`. VoI column is PLANNED, not yet run.*

---

## 13. Post-training evaluation harness (BOTH models, locked 2026-06-15)

Standalone post-training eval orchestrators land both runs' numbers in a single consolidated JSON
per model, restart-safe, end-to-end on 8×H200. Built to overlay byte-comparably with
grpocredit's `scripts/eval_final_checkpoint.py` outputs (same grader, same
`verl.utils.reward_score.math` import) — see `SPO/scripts/eval_spo_{deepseek7b,qwen1b}.py`.

### 13.1 Sampling protocol (locked)

| Model | Dataset | n | T | top_p | max_model_len | max_tokens | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| DeepSeek 7B | math_v2 (40 ckpts) | 16 | 0.35 | 0.9 | 4096 | 1024 | VinePPO protocol, matches grpocredit VoI standalone exactly |
| DeepSeek 7B | aime24 (final 4) | 32 | 0.35 | 0.9 | 4096 | 1024 | |
| DeepSeek 7B | olympiadbench (final 4) | 16 | 0.35 | 0.9 | 4096 | 1024 | |
| DeepSeek 7B | collegemath (final 4) | 16 | 0.35 | 0.9 | 4096 | 1024 | |
| Qwen 1.5B | math_v2 (40 ckpts) @ 2K | 16 | 0.6 | 0.95 | 2048 | 2048 | matches SPO `evaluate_long_cot.sh` 2K row |
| Qwen 1.5B | aime24 (final 4) @ 2K | 32 | 0.6 | 0.95 | 2048 | 2048 | |
| Qwen 1.5B | olympiadbench (final 4) @ 2K | 16 | 0.6 | 0.95 | 2048 | 2048 | |
| Qwen 1.5B | collegemath (final 4) @ 2K | 16 | 0.6 | 0.95 | 2048 | 2048 | |
| Qwen 1.5B | math_v2 (final 4) @ 4K | 16 | 0.6 | 0.95 | 4096 | 4096 | extrapolation row, matches SPO published 4K column |

**Why these knobs.**
- *DeepSeek 7B at T=0.35 / max_tokens=1024:* identical to grpocredit's `eval_final_checkpoint.sh`
  (VinePPO protocol) so the resulting `pass1`/`pass_at_n` overlay grpocredit's VoI/GRPO numbers
  on the same parquets without any normalization.
- *Qwen 1.5B at T=0.6 / top_p=0.95:* matches SPO's own `scripts/evaluate_long_cot.sh` and the
  paper's published Qwen-1.5B MATH-500 number; long-CoT decoding wants higher T than the
  short-CoT VinePPO setting.
- *max_model_len pinned explicitly:* SPO Qwen trained at **effective 2K total** (see §2 —
  `model_context_size=2048` was the binding cap inside `_compute_max_tokens`), even though
  `max_tokens=4096` and the R1-distill base ships `max_position_embeddings=131072`. Not setting
  `max_model_len` would let vLLM sample sequences the model never saw in training; the 4K row
  is therefore a deliberate **extrapolation probe**, not a "matched-training" eval.
- *AIME-24 n=32:* AIME has only 30 problems; n=16 leaves per-question pass-rate dominated by
  Bernoulli noise. Matches grpocredit's AIME convention (`sbatch_eval_ood_*.sh`).
- *Final-4 OOD slate (iter 925, 950, 975, 1000):* the OOD numbers are most informative at
  late-training; a full 40×4 OOD sweep would be ~25h on collegemath alone.

### 13.2 Output JSON schema
Output location:``` /home/zengh/projects/SPO/results/eval_spo_deepseek7b_seed42.json```

Each orchestrator writes ONE consolidated JSON, appended atomically after every ckpt:

```json
{
  "model_label": "spo_tree_deepseekmath_7b_sft_MATH_v2_seed42",
  "run_dir": "/lustre-storage/.../checkpoints",
  "base_model_snapshot": "/lustre-storage/.../snapshots/8b387c2...",
  "seed": 42,
  "generated_utc": "2026-06-15T...",
  "last_updated_utc": "2026-06-15T...",
  "evals": [
    {
      "ckpt_iter": 25, "ckpt_step": 200, "dataset": "math_v2",
      "n_samples_per_question": 16, "max_model_len": 4096, "max_tokens": 1024,
      "temperature": 0.35, "top_p": 0.9,
      "n_questions": 500,
      "pass1": 0.412, "pass_at_n": 0.604,
      "se_cluster": 0.022, "se_bernoulli": 0.014, "ci95": 0.043,
      "gen_seconds": 287.4, "wall_seconds": 312.0
    }
  ]
}
```

`se_cluster` is the recommended SE (between-question, treats each problem as the unit of analysis);
`se_bernoulli` is reported alongside as the optimistic iid-sample formula. `ci95 = 1.96 *
se_cluster` is the headline error bar.

### 13.3 Launch

```bash
sbatch /home/zengh/projects/SPO/scripts/sbatch_eval_spo_deepseek7b.sh   # ~7h on 8×H200, 1d allocation
sbatch /home/zengh/projects/SPO/scripts/sbatch_eval_spo_qwen1b.sh       # ~4h on 8×H200, 1d allocation
```

Outputs land at:
- `SPO/results/eval_spo_deepseek7b_seed42.json` (52 evals: 40 MATH-500 + 12 OOD)
- `SPO/results/eval_spo_qwen1b_seed42.json` (56 evals: 40 MATH-500@2K + 12 OOD@2K + 4 MATH-500@4K)

**Restart-safe.** Each orchestrator appends to the consolidated JSON after each ckpt completes
(atomic tmp + rename); on resubmit it parses the existing JSON, builds a set of done
`(ckpt_iter, dataset, n, max_model_len, max_tokens)` tuples, and skips them. Resubmit freely
on preempt / timeout / OOM.

### 13.4 Stack

| File | Purpose |
|---|---|
| `scripts/_eval_one_ckpt.py` | Single-eval worker. Vendored from `grpocredit/scripts/eval_final_checkpoint.py`; adds `--max-model-len`; emits delimited JSON block on stdout for orchestrator capture. |
| `scripts/_eval_orchestrator.py` | Shared orchestrator helpers (ckpt discovery, resume-skip, atomic JSON append). |
| `scripts/_prepare_spo_ckpt_for_vllm.sh` | Symlinks tokenizer files from base-model snapshot into `<ckpt>/hf_pretrained/` (SPO ckpts only save weights+config, not tokenizer). Idempotent. |
| `scripts/build_qwen_templated_parquets.py` | Rewrites grpocredit's 4 parquets with the R1-distill chat template → `SPO/data/verl_qwen/*.parquet`. Same rows, only `prompt[0].content` changes. |
| `scripts/eval_spo_deepseek7b.py` | DeepSeek 7B orchestrator (52 work items). Reuses grpocredit's parquets directly (`[MATH_TASK] Problem:` template matches the 7B SFT base). |
| `scripts/eval_spo_qwen1b.py` | Qwen 1.5B orchestrator (56 work items). Uses Qwen-templated parquets. |
| `scripts/sbatch_eval_spo_{deepseek7b,qwen1b}.sh` | 8×H200 exclusive sbatches, `grpocredit-verl-pinned` env, HF offline. |

Both orchestrators support `--dry-run` for sanity-checking the work list before launching.
