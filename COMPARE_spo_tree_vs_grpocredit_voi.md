# Length knobs glossary — `max_model_len`, `max_tokens`, `model_context_size`, etc.

These are all "length" knobs at different layers:

| Knob | Layer | What it caps |
|---|---|---|
| **`max_tokens`** | vLLM sampling (also SPO's `max_tokens`) | **New response tokens** per generation (the answer length) |
| **`max_response_length`** | verl | Same as `max_tokens` — verl's name for the response cap |
| **`max_model_len`** | vLLM engine | **Total** sequence = prompt + response. Hard limit; vLLM **errors** if the prompt alone exceeds it |
| **`model_context_size`** | SPO/treetune | SPO's own version of `max_model_len` — the total budget enforced inside `_compute_max_tokens` |
| **`max_position_embeddings`** | HF model config | The model's **architectural** max context (e.g. 4096 for deepseekmath-7B, 131072 for R1-Distill). `max_model_len` defaults to this if unset |
| **`max_prompt_length`** | verl | **Prompt cap** — verl truncates/drops prompts longer than this |
| **`max_question_length`** | SPO | SPO's prompt-side filter — drops dataset rows whose templated prompt exceeds it |

**The one formula tying them together** (how many response tokens you actually get):

```
effective_response = min( max_tokens , total_ctx − prompt_tokens )
```

where `total_ctx` = `max_model_len` (vLLM) or `model_context_size` (SPO).

So:
- **Response** is bounded by `max_tokens`/`max_response_length`.
- **Total (prompt+response)** is bounded by `max_model_len`/`model_context_size`.
- If `total_ctx` is small, a long prompt **steals** from the response budget — that's the whole "footgun" we kept hitting (SPO `max_tokens=4096` but `model_context_size=2048` → real response ≈ `2048 − prompt`).
- `max_prompt_length`/`max_question_length` bound the **prompt** before that math even applies (truncate vs drop).

---

# SPO-tree (deepseekmath-7B) vs grpocredit VoI — training comparison

**Both fine-tune the SAME 7B base: `realtreetune/deepseekmath-7b-sft-MATH-v2`.** This is
the apples-to-apples head-to-head (not the 1.5B published-SPO setting — see Appendix).

- **SPO-tree (7B)** — `SPO/configs/polIter_deepseekSft2_spo_tree_MATH.jsonnet`
  🔧 **after the local import-fix** (upstream-broken `vineppo_MATH` import repointed to
  the shipped `polIter_deepseekSft2_spo_chain_MATH.jsonnet`).
  All values **resolved via `_jsonnet`**. Framework: **treetune + DeepSpeed ZeRO-2**.
  Launch: `scripts/sbatch_spo_tree_deepseek7b_MATH.sh` → `launch_server2_spo_multi.sh spo_tree` (8×H200).
- **VoI** — grpocredit `grpo_voi_v041_p4_mb8_T08unifAclip2_deepseekmath_math_v2_seed42`,
  resolved from `hydra_20260603T003239Z.overrides.txt`. Framework: **verl + FSDP2 + Ray**.

> ⚠️ 7B SPO-tree is **not a published/validated SPO setting** (no 7B SPO number exists;
> `M=66`, 1024-episode batch are unreferenced). Correct structural instantiation, fine
> for a head-to-head; not "reproducing SPO". Per your call, **SPO keeps its own knobs**
> (lr/KL/T) — each method at its own optimum, not knob-matched.

> 🔧 **Local edits (suesie fork) vs upstream `AIFrameResearch/SPO@1e64f0c`.** Every row
> below labelled 🔧 is an edit we made on top of pristine upstream; everything else is
> the shipped value. Four edits total:
> 1. **`polIter_deepseekSft2_spo_tree_MATH.jsonnet`** — repointed the broken
>    `vineppo_MATH` import to `polIter_deepseekSft2_spo_chain_MATH.jsonnet` so the
>    SPO probability-mask update (`use_prob_mask=true`) is actually applied. Without
>    this fix the file fails to load.
> 2. **`polIter_deepseekSft2_ppo_MATH.jsonnet`** — added `evaluate_every_n_iterations: 25`
>    (runtime default is 10) to match grpocredit's `test_freq=25`. This base is imported
>    by spo_chain → spo_tree, so the cadence flows through.
> 3. **`sft_deepseekmath_for_MATH_eval.jsonnet`** — flipped the active in-loop pipeline
>    from `math_test_inference_pipeline` to `math_validation_inference_pipeline` so the
>    apples-to-apples eval uses validation, not test.
> 4. **`sft_deepseekmath_for_MATH_eval.jsonnet` + new `analyzers/mean_at_k.jsonnet` + new
>    `src/treetune/analyzers/mean_at_k_analyzer.py`** — wired a verl-port analyzer onto
>    the validation pipeline so SPO logs the exact `val-core/math_v2/reward/mean@k`
>    wandb keys grpocredit emits.

---

## 1. Model
| | SPO-tree 7B | VoI |
|---|---|---|
| base model | `realtreetune/deepseekmath-7b-sft-MATH-v2` | `realtreetune/deepseekmath-7b-sft-MATH-v2` |
| params trained | full fine-tune | full fine-tune |
| chat template | `[MATH_TASK] Problem:\n{query}\n\nSolution:` (treetune `prompt_library.tree.question_template`) | `{% for m in messages %}{{ m.content }}{% endfor %}` |
| remove-padding | n/a (treetune) | `use_remove_padding=True` |
| grad checkpointing | True | True |

## 2. Data & sequence lengths
| | SPO-tree 7B | VoI |
|---|---|---|
| train data | **MATH train** — treetune `data/math` (HF arrow), **11,500** | **MATH train** — `math_train_vineppo.parquet`, **11,496** (`vineppo:math/train`) |
| val data | **MATH validation** — `data/math` (in-loop, 🔧 edit #3), **500** | **MATH validation** — `math_val.parquet`, **500** |
| **max response length** | **1024** (`max_tokens`) | **1024** (`max_response_length`) |
| max prompt length | `max_question_length=1512` | `max_prompt_length=512` |
| model_context_size | 2048 | (prompt 512 + resp 1024 = 1536 effective) |
| max_sequence_length | null (uncapped beyond above) | — |
| overlong handling | tree handles; fill_missing_episodes | `filter_overlong_prompts=True`, `truncation=error` |
| shuffle | `dataset_shuffle_on_each_iteration=True` | `data.shuffle=True` |
| sample w/ replacement | True | (epoch sampler) |
| data seed | 42 | `+data.seed=42` |

**Same training data — both comparisons share one MATH dataset; only the wrapping differs.**
- **VoI ↔ SPO (this doc):** same MATH problems + gold answers. SPO trains on treetune `data/math` train (**11,500**); VoI on `math_train_vineppo.parquet` (**11,496**, source `vineppo:math/train`) — the same set (~4-row gap = overlong-prompt filtering). In-loop **validation is the same 500** (`math_val.parquet` == `data/math/validation`, verified 500/500). *(Both are RL — identical prompts + gold answers; the rollouts/trajectories are self-generated and so differ by method.)*
- **SPO 7B ↔ SPO 1.5B (R1-Distill, see Appendix):** **identical** dataset — the same `data/math` (train **11,500** / val **500** / test **500**), loaded via the same `tasks/math_inplace_no_answer_prefix.jsonnet` in both `polIter_deepseekSft2_ppo_MATH.jsonnet` and `polIter_qwen1b_spo_chain_MATH.jsonnet`. Only the **prompt template** (`[MATH_TASK] Problem:…` vs R1-Distill `<think>`) and **length regime** (resp 1024 vs 4096) differ — **not the data**.

## 2.1 Token / context / sequence-length budget — spo-chain vs spo-tree vs VoI
*(the length knobs you asked about, broken out per SPO variant; expands the response/context/seq rows of §2)*

**All length knobs are identical for spo-chain and spo-tree.** `polIter_deepseekSft2_spo_tree_MATH.jsonnet`
imports `polIter_deepseekSft2_spo_chain_MATH.jsonnet` (🔧 after the local import-fix) and only
swaps in the hybrid 6-6-6 tree (`M=66`, `max_depth=3`); it overrides **no** length knob. Both
inherit from `polIter_deepseekSft2_ppo_MATH.jsonnet` (`max_tokens` :66, `model_context_size` :72,
`max_sequence_length` :47). These three length values are the **official upstream** values
(commit `1e64f0c`, AIFrameResearch/SPO); the only local edit to that base file is
`evaluate_every_n_iterations: 25` at :96-97 (🔧, see §9), which doesn't touch any length knob.

| length knob | SPO-chain 7B | SPO-tree 7B | VoI 7B |
|---|---|---|---|
| **response / token length** *(SPO: per tree-node / per step)* | **1024** (`max_tokens`, per step) | **1024** (inherited, per step) | **1024** (`data.max_response_length`, whole response) |
| **context length** *(total budget: prompt + whole trajectory)* | **2048** (`model_context_size`) | **2048** (inherited) | **1536** effective (`max_prompt_length` 512 + `max_response_length` 1024) |
| **max_sequence_length** | **null** (no extra cap) | **null** (inherited) | n/a — hard cap = 512+1024; `filter_overlong_prompts=True`, `truncation=error` |
| max prompt length | 1512 (`max_question_length`) | 1512 (inherited) | 512 (`max_prompt_length`) |

- **`max_sequence_length` (the null / n-a row).** SPO-chain/tree leave it `null` and it never fires — generation is already bounded by `model_context_size`=2048 (total budget: prompt + whole root→leaf trajectory) and `max_tokens`=1024 (per tree-node / per step) — and VoI needs no equivalent because verl already caps every sequence at `max_prompt_length`+`max_response_length`=512+1024, dropping overlong prompts (`filter_overlong_prompts=True`, `truncation=error`).
- **Effective generation (per step vs whole trajectory).** For SPO, `max_tokens`=1024 is a **per-node / per-step** cap: at each expansion SPO emits up to `min(max_tokens, model_context_size − prefix_tokens)` new tokens (`_compute_max_tokens`, `src/treetune/inference_strategies/tree_inference/expansion.py:401-408`; `max_tokens` is threaded in via `_sample_node` :326-350 and `expand` :426) — i.e. **effective response length per step ≤1024**, shrinking as the prefix grows. The `prefix` accumulates question + all ancestor steps, so the **full root→leaf solution ≤ 2048 − prompt** (`model_context_size − prompt_tokens`), **not** 1024. VoI instead generates up to `max_response_length=1024` as the **whole** response in one shot, with prompts hard-capped at 512 (overlong prompts are **dropped**, not truncated).
- **Matched vs not.** The **per-call** number is matched **1024 ↔ 1024**, but the *unit* differs: SPO's 1024 is **per step** (so a multi-step root→leaf solution can total up to `2048 − prompt`), whereas VoI's 1024 is the **entire** response. SPO also allows a larger context
  window (**2048** vs VoI's 1536 effective) and far longer prompts (**1512** vs **512**), and sets
  no separate training-time `max_sequence_length` (null); VoI's sequence ceiling is just the
  512+1024 budget enforced by verl.
- **GSM8K kept out of a second table (decision).** No runnable SPO GSM8K counterpart exists for
  spo-chain/spo-tree: `polIter_deepseekSft2_spo_chain_GSM8K.jsonnet` imports the non-shipped
  `polIter_deepseekSft2_ppo_MATH_mc_advantages.jsonnet` (import fails) and there is no
  `polIter_deepseekSft2_spo_tree_GSM8K.jsonnet`, so a GSM8K SPO column would be empty. For
  reference, VoI's GSM8K run uses the **same** budget — `max_response_length=1024`,
  `max_prompt_length=512` (`grpocredit/scripts/launch_verl_grpo_voi_v041_p4_mb8_T08unifAclip2.sh:70-86`).

## 3. Algorithm / advantage
| | SPO-tree 7B | VoI |
|---|---|---|
| estimator | **SPO-tree** segment-MC | **`voi_td_sparse_pivot`** |
| tree / probes | branch **6-6-6**, `max_depth=3`, **M=66**; value-MC `samples=9` | sparse VoI-selected TD pivots |
| token-level update | **probability mask** (`use_prob_mask=true`) | per-token PPO ratio |
| learned critic | none (`critic_model=null`) | none (`reward_model.enable=False`) |
| γ (gamma) | 1 | 1 (estimator-internal) |
| λ (lam) | 1 (pure MC) | — (TD pivots) |
| advantage whitening | `whiten_advantages=true`, `whiten_rewards=false` | estimator-internal (verl) |
| score scaling/norm | `use_score_scaling=false`, `use_score_norm=false` | — |

## 4. Rollout / generation
| | SPO-tree 7B | VoI |
|---|---|---|
| engine | vLLM (per-rank servers) | vLLM (`tensor_model_parallel_size=2`) |
| **train temperature** | **0.6** | **0.8** |
| **train top_p** | **0.9** | **0.95** |
| top_k | (default) | -1 |
| do_sample | yes | `do_sample=True` |
| dtype | bf16 | `bfloat16` |
| rollouts per question | tree 6-6-6 expansion (M=66) | `rollout.n=8` (flat) |
| stop | `"\n\n\nProblem:"` | (eos) |
| gpu_memory_utilization | **0.4** | **0.6** |
| swap_space | 32G | (verl-managed) |
| max_num_seqs | 512 | (verl-managed) |
| prefix caching | True | (chunked_prefill=False, enforce_eager=False, free_cache_engine=False) |
| append bos/eos | bos to query, eos to response | (chat-template managed) |
| **val/eval sampling** | **T=0.35, top_p=0.9, n=16, max_tokens=1024, model_context_size=4095** (in-loop, `sft_deepseekmath_for_MATH_eval.jsonnet`) | **T=0.35, top_p=0.9, n=4** |

## 5. Optimization
| | SPO-tree 7B | VoI |
|---|---|---|
| **learning rate** | **1e-6** | **5e-7** |
| warmup | `warmup_ratio=0.03` | `warmup_style=constant`, `lr_warmup_steps_ratio=0.0` |
| min_lr_ratio / cycles | — | `min_lr_ratio=0.0`, `num_cycles=0.5` |
| weight decay | 0 | 0 |
| clip ratio | 0.2 (value clip 0.2) | `clip_ratio=0.2` |
| grad clip | `max_grad_norm=1.0` | `grad_clip=1.0` |
| entropy coef | — (report_entropy off) | `entropy_coeff=0.0` |
| PPO epochs / iter | 1 | 1 |
| critic warmup | n/a (no critic) | `critic_warmup=0` |

## 6. KL control
| | SPO-tree 7B | VoI |
|---|---|---|
| placement | in **loss** (`kl_penalty_loss_type=control_variate`) | in **loss** (`use_kl_loss=True`) |
| KL in reward | no | `use_kl_in_reward=False` |
| **KL coef** | **1e-4** (`init_kl_coef`) | **1e-3** (`kl_loss_coef`) |
| KL estimator | control_variate (clip_min 0, clip_max 1e9) | `low_var_kl` |
| adaptive KL | `adap_kl_ctrl=false` | (fixed) |

## 7. Batching / throughput  ← (the rows you asked about)
| | SPO-tree 7B | VoI |
|---|---|---|
| **questions / iter** | **16** (`dataset_num_samples_per_iteration`) | **64** (`train_batch_size`) |
| **episodes / iter** | **1024** (`num_episodes_per_iteration`) | **512** (64 prompts × `rollout.n=8`) |
| episodes per question | **64** (= `num_episodes_per_iteration`/`num_dataset_samples_per_iteration` = 1024/16; tree budget cap is `M=66`) | 8 (flat trajectories) |
| **global optimizer batch** | `target_train_batch_size=128` | `ppo_mini_batch_size=8` |
| per-device train batch | 8 | `ppo_micro_batch_size_per_gpu=2` |
| grad accumulation | auto → **2** (128 ÷ (8×8 GPU)) | dynamic (token-packed) |
| optimizer steps / iter | 1024 ÷ 128 = **8** | mini-batches over 512 (verl, dynamic) |
| dynamic batching | no (fixed counts) | `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=3000` |
| logprob micro-batch | (treetune internal) | actor/ref `log_prob_micro_batch_size_per_gpu=2`, dyn, `max_token_len/gpu=4096/8192` |

### Explanation of three rows (one from §2, two from §7)
- **prompt cap (1512 vs 512) — §2 row.** Max tokens allowed on the *prompt/question* side
  (independent of response length). SPO (`max_question_length=1512`) admits much
  longer prompts; VoI (`max_prompt_length=512`) caps at 512 and `filter_overlong_prompts`
  drops longer ones. For MATH most prompts are <512 so this rarely changes which items
  are kept — but the budgets differ. (Response length is the separate 1024-vs-1024 row.)
- **data/iter ("16 q → tree → 1024 episodes" vs "64 prompts × 8 = 512") — §7 row.** SPO draws
  **16 unique questions** per iteration (`num_dataset_samples_per_iteration`) and a
  total of **1024 episodes/iter** (`num_episodes_per_iteration`) — i.e. **64 episodes
  per question** by the budget ratio 1024/16, with the tree topology independently
  capped at `M=66` nodes/question (6-6-6 expansion: depth-0→6, depth-1→6, depth-2→6).
  Every tree node/segment becomes a training example, yielding dense *segment-level*
  credit from few questions. VoI draws **64 prompts** and samples **8 flat
  trajectories each** → **512 rollouts/iter** — sparse *trajectory-level* signal from
  many prompts. Same compute spent very differently: SPO = few questions × deep tree;
  VoI = many prompts × shallow group.
- **opt batch ("target 128, per-dev 8 (accum 2 × 8 GPU)" vs "mini 8, micro 2/gpu,
  dynamic-bsz") — §7 row.** SPO uses **fixed-count** batching: a global optimizer batch of
  **128 episodes**, split 8/GPU across 8 GPUs ⇒ gradient-accumulation = 128/(8×8)=**2**;
  with 1024 episodes/iter that's **8 optimizer steps/iter**. VoI uses **token-dynamic**
  batching: PPO updates over mini-batches of **8** (`ppo_mini_batch_size`), each formed
  from **micro-batches of 2/GPU** that verl packs to ≈`3000` tokens/GPU
  (`use_dynamic_bsz`), so the *number of sequences* per micro-batch floats with length.
  Net: SPO counts **episodes/segments** with a fixed batch; VoI counts **prompts/tokens**
  with a length-adaptive batch.

## 8. Distributed / sharding / engine
| | SPO-tree 7B | VoI |
|---|---|---|
| actor sharding | **DeepSpeed ZeRO-2**, no optimizer offload | **FSDP2**, `param_offload=False`, `optimizer_offload=False` |
| reference model | on CPU (`move_reference_model_to_cpu=true`), ds-auto | FSDP2, `param_offload=True` |
| rollout placement | per-rank vLLM, `wait_until_memory_release=true`, `min_avail=4096MB` | vLLM TP=2, colocated |
| precision | bf16 | bf16 |
| orchestration | deepspeed launcher | Ray (`ray_init.num_cpus=32`) |

## 9. Schedule / eval / checkpoint / logging
| | SPO-tree 7B | VoI |
|---|---|---|
| total steps/iters | 1000 | `total_training_steps=1000` |
| in-loop eval cadence | 🔧 **every 25 iters** (`evaluate_every_n_iterations: 25` — local edit; upstream default is 10) | `test_freq=25` |
| eval | 🔧 in-loop **MATH-validation** (T=0.35, top_p=0.9, n=16; was MATH-test upstream — locally flipped, see §9 callout) + lighteval `math_500` offline | `test_freq=25`, `val_before_train=True` |
| save cadence | `save_steps=10`, `checkpoint_keep_steps=40` | `save_freq=100`, `max_actor_ckpt_to_keep=2` |
| resume | (treetune) | `resume_mode=auto` |
| logging | `logging_steps=1`, `episodes_cloud_log_steps=50` | `logger=[console,wandb]` |
| wandb | entity `suesie`, project `spo-math` | project `grpocredit-grpo-verl` |

### Eval: SPO vs VoI on DeepSeek-7B

| Eval path | `model_context_size` (training-time prompt + response, for reference only) | eval `max_model_len` | eval `max_tokens` | prompt room (len − max_tokens) |
|---|---|---|---|---|
| **SPO** (`eval_spo_deepseek7b.py:65-66`) | 2048 | **4096** (matched to VoI standalone) | 1024 | **3072** |
| **VoI standalone** (`eval_final_checkpoint.py`) | 1536 (=512+1024) | **4096** (model native, unset) | 1024 | **3072** |
| **VoI in-loop val** (during training) | 1536 | **1536** (verl: prompt+resp, `vllm_rollout.py:104-105`) | 1024 | **512** |

## 10. Compute / resources (MATCHED for fair comparison)
| | SPO-tree 7B (sbatch) | VoI (sbatch) |
|---|---|---|
| account / qos | h200_mrs_2 / h200_mrs_2_high | h200_mrs_2 / h200_mrs_2_high |
| GPUs | **gpu:h200:8** | **gpu:h200:8** |
| cpus-per-task | **96** | **96** |
| nodes / ntasks | 1 / 1 | 1 / 1 |
| exclusive | yes | yes |
| host RAM | `--mem=0` † | `--mem=512G` |
| walltime cap | 3-00:00:00 | 2-00:00:00 |

† GPU(8)/CPU(96) — the compute — are identical. treetune's 8 per-rank vLLM
(swap_space 32G ×8 = 256G) + ZeRO-2 staging OOM'd VinePPO at 512G, so `--mem=0`
(whole-node, exclusive) is used; host RAM is a floor, not a compute lever (same
reasoning as the VinePPO runbook). Walltime is a ceiling, not a resource.

## 11. Reward / environment
| | SPO-tree 7B | VoI |
|---|---|---|
| reward | `math_reward_function`, unfinished penalty 0.0 | custom `verl_reward.compute_score` |
| conda env | `vineppo`/`spo` (torch2.1.2 / vllm0.4.0.post1 / ds0.14.1) | `grpocredit-verl-pinned` |
| seed | 42 | 42 |

---

## 12. Bottom line
- **Matched:** base model (**deepseekmath-7B**), **response length 1024**, 8×H200 / 96 CPU /
  exclusive, critic-free, KL-as-loss, clip 0.2, grad-clip 1.0, 1 PPO epoch, γ=1, wd 0,
  1000 steps, seed 42.
- **Intentional method deltas (kept):** credit assignment (**tree segment-MC + prob-mask**
  vs **TD sparse-pivot**), and each method's own tuned **lr / KL / temperature**
  (1e-6 / 1e-4 / 0.6 vs 5e-7 / 1e-3 / 0.8) — left unmatched by design.
- **Framework deltas (unavoidable):** treetune+DeepSpeed-ZeRO2 vs verl+FSDP2+Ray;
  fixed-count vs token-dynamic batching; 16-questions×tree vs 64-prompts×8.
- **Compare by time-to-accuracy / step-to-accuracy**, not raw iterations (per-iter cost
  differs: SPO's tree+MC vs VoI's flat rollouts).

---

## Appendix — why there was a 1.5B comparison earlier
SPO's **published** SPO-tree result is on **DeepSeek-R1-Distill-Qwen-1.5B** (long-CoT,
response 4096), config `polIter_qwen1b_spo_tree_MATH.jsonnet`, giving MATH500
0.736/0.828/0.848 @ 2K/4K/32K. That is a *different model/regime* from your 7B VoI run,
so a 1.5B-vs-7B table is not a fair head-to-head — **this document uses the 7B SPO-tree
config instead** so both sides share the deepseekmath-7B base. For the faithful 1.5B
reproduction (the actual published numbers) see `RUN_spo_tree_qwen1b_MATH_seed42.md`.

*Sources: resolved `SPO/configs/polIter_deepseekSft2_spo_tree_MATH.jsonnet`
(🔧 import-fixed → `polIter_deepseekSft2_spo_chain_MATH.jsonnet`; see "Local edits"
callout near the top for the full list of four local edits) via `_jsonnet`;
`grpocredit/.../grpo_voi_v041_p4_mb8_T08unifAclip2_deepseekmath_math_v2_seed42/hydra_20260603T003239Z.overrides.txt`
+ `scripts/sbatch_launch_p4_mb8_T08unifAclip2_math_v2_seed42.sh`.*
