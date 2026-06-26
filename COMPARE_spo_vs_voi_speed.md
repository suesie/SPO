# SPO (chain & tree) vs grpocredit VoI — training-speed & dynamics comparison

**Date**: 2026-06-09
**Base model (all three arms)**: `realtreetune/deepseekmath-7b-sft-MATH-v2`
**Hardware**: 8 × H200 (identical for all arms)
**Companion docs**: config-level table in `SPO/COMPARE_spo_tree_vs_grpocredit_voi.md`; the
VinePPO analog this is modeled on in `VinePPO-grpo/COMPARE_vineppo_vs_voi_speed.md`.

> ⚠️ **Read this first — what is measured vs analytical.**
> - **VoI numbers are MEASURED** (wandb), from
>   `grpocredit/experiments/main/grpo_voi_v041_p4_mb8_T08unifAclip2_deepseekmath_math_v2_seed42`
>   (`launcher_20260603T003239Z.sh`), the same run used in the VinePPO speed doc.
> - **SPO numbers are ANALYTICAL.** No DeepSeekMath-7B SPO run exists yet (the `results/`
>   dir holds only the authors' published **1.5B** eval JSONs; this box has no GPU). The SPO
>   columns below are derived from the *resolved configs + episode-generator source*, with
>   every data-dependent quantity given as an explicit formula + sensitivity table and a
>   pointer to the exact runtime metric to confirm it. Treat them as **predictions to
>   validate**, not observations.

---

## 0. Which SPO 7B training scripts exist (provenance)

The SPO GitHub (`AIFrameResearch/SPO`) ships these DeepSeekMath-7B configs:

| Config | State at upstream | Runnable? |
|---|---|---|
| `configs/polIter_deepseekSft2_spo_chain_MATH.jsonnet` | shipped | ✅ as-is (canonical 7B SPO) |
| `configs/polIter_deepseekSft2_spo_chain_GSM8K.jsonnet` | shipped | ✅ as-is (GSM8K) |
| `configs/polIter_deepseekSft2_spo_tree_MATH.jsonnet` | shipped | ⚠️ **broken as-shipped** — imports a missing `polIter_deepseekSft2_vineppo_MATH.jsonnet`; our fork has a 6-line import-fix repointing it at the shipped chain base (`git diff upstream/master` = 1 file, +6/−1) |

Both arms therefore exist (per the ask, both are compared). Launcher
`scripts/launch_server2_spo_multi.sh {spo_chain|spo_tree|grpo}` drives all of them on 8×H200;
`scripts/sbatch_spo_tree_deepseek7b_MATH.sh` is the matched-resource submit script.

---

## 1. Summary (the headline)

All three arms do the **same ~8 optimizer steps per cycle** and use the **same 1024-token
response cap**. The entire speed story is in **(a) how many continuations each spends on
credit assignment** and **(b) the framework pipeline tax.**

1. **Credit-assignment generation is the bottleneck, and SPO spends 1–2 orders of magnitude
   more of it than VoI.**
   - **VoI (measured):** ~2,560 sparse probe continuations ≈ **0.8M tokens/step**.
   - **SPO-chain (analytical):** `512 × (N₍<0.9₎/5) × 9` MC rollouts. Even at a *modest*
     ~10 cutpoints/traj that is **~46k MC rollouts ≈ 14M tokens/iter** — i.e. **VinePPO-class**
     (VinePPO ≈ 34.5k / 17.3M), and **~17–26× VoI**.
   - **SPO-tree (analytical):** a 6-6-6 tree over only **16 questions/iter** ≈ **4.1k node
     generations ≈ 1.0–1.3M tokens/iter** — ~1.5× VoI's per-step tokens but spread over **4×
     fewer prompts**, so ~**6× VoI per prompt**.

2. **SPO inherits VinePPO's serialized pipeline tax.** Both SPO arms are treetune+DeepSpeed:
   they `kill_vllm_server()` and rebuild the actor **every iteration** (chain
   `…mc_advantages.py:136`, tree `hybrid_episode_generator.py:228`). That is the same
   ~95s/iter of pure overhead (vLLM restart + DeepSpeed rebuild) the VinePPO doc measured.
   VoI's colocated, always-on FSDP2 + vLLM (TP=2, 4 replicas) pays **0s** of it.

3. **Net prediction:** on matched 8×H200, **SPO-chain ≈ VinePPO-speed** (≈10× slower per
   cycle than VoI, likely worse since cutpoints are uncapped); **SPO-tree** is cheaper than
   chain on raw generation but still pays the pipeline tax and concentrates its budget on 16
   prompts/iter. **Compare on time-to-accuracy, not iteration count.** The single number that
   decides chain's cost — `episodes_metric/num_cutpoints` — must be read off a live run.

---

## 2. Matched settings (identical across arms)

| Parameter | SPO-chain | SPO-tree | VoI |
|---|---|---|---|
| Base model | `deepseekmath-7b-sft-MATH-v2` | same | same |
| **Max response tokens** | **1024** | **1024** | **1024** |
| Dataset | MATH (train) | MATH (train) | MATH (train) |
| GPUs / CPU / exclusivity | 8×H200 / 96 / yes | same | same |
| Critic | none | none | none |
| KL placement | in loss | in loss | in loss |
| PPO clip | 0.2 | 0.2 | 0.2 |
| Grad clip | 1.0 | 1.0 | 1.0 |
| γ | 1 | 1 | 1 |
| Total iters/steps | 1000 | 1000 | 1000 |
| Optimizer steps / cycle | **8** (512/128 × 2ep) | **8** (1024/128 × 1ep) | **8** (64/`mb8` × 1ep) |
| Seed | 42 | 42 | 42 |

> The optimizer-step count is matched at **8/cycle**, so per-cycle *training* compute is
> comparable. Everything expensive is upstream of training, in credit assignment.

---

## 3. Intentional method deltas (kept — each at its own optimum)

| Knob | SPO-chain | SPO-tree | VoI |
|---|---|---|---|
| Estimator | cutpoint segment-MC | 6-6-6 tree segment-MC | `voi_td_sparse_pivot` |
| Token-level update | prob-mask `𝟙[π_old<0.9]` | prob-mask `𝟙[π_old<0.9]` | per-token PPO ratio |
| Train temperature | **0.6** | **0.6** | **0.8** |
| Train top_p | 0.9 | 0.9 | 0.95 |
| Learning rate | **1e-6** | **1e-6** | **5e-7** |
| KL coef / estimator | **1e-4** / control_variate | 1e-4 / control_variate | **1e-3** / low_var_kl |
| PPO epochs / iter | **2** | **1** | 1 |
| λ | 1 (pure MC) | 1 (pure MC) | — (TD pivots) |
| Actor sharding | ZeRO-2 (no offload) | ZeRO-2 (no offload) | FSDP2 |
| vLLM | per-rank 8×TP=1, **restarted/iter** | per-rank 8×TP=1, **restarted/iter** | TP=2 ×4, **always-on** |

---

## 4. Per-cycle compute structure (the substance)

A "cycle" = one outer generate→credit→train loop: VoI **step** ↔ SPO **iteration**.
SPO-chain and VoI both touch **64 prompts/cycle**; SPO-tree touches **16 questions/cycle**.

| Quantity | SPO-chain | SPO-tree | VoI (measured) |
|---|---|---|---|
| Unique prompts / cycle | 64 | **16** | 64 |
| Main rollouts (gradient trajs) | 512 (64×8) | — (tree *is* the rollout) | 512 (64×8) |
| Credit-assignment generations | `512 × ⌈N₍<0.9₎/5⌉ × 9` MC | ~258 nodes × 16 ≈ **4,128** | **2,560** probes |
| → token budget / cycle | **~10–20M** (see §5) | **~1.0–1.3M** (see §6) | **~0.8M** |
| Edges/episodes trained / cycle | 512 | 1024 (replay-buffered) | 512 |
| Optimizer steps / cycle | 8 | 8 | 8 |
| vLLM restart + actor rebuild | **yes (~95s)** | **yes (~95s)** | **no (0s)** |
| Pipeline | serialized | serialized | colocated/overlapped |

**Where the SPO timing dominators are** (mapping onto the VinePPO doc's measured split, which
was: MC/credit **56%**, train 13%, vLLM restart 9%, other overhead 21%, main rollout 2%):
because SPO uses the *same treetune pipeline* as VinePPO, expect the same shape — **credit
assignment + restart + rebuild ≈ 85% of wall time**, training ~10%, main rollout ~2%.

---

## 5. SPO-chain credit-assignment budget (data-dependent — the crux)

Source: `math_episode_generator_with_mc_advantages.py:919-937` (active path; the per-reasoning-
step path is commented out, and `max_step_for_value_estimation` is **not enforced**):

```python
response_probs = np.exp(response_logprobs)
cutpoints = np.where(response_probs < 0.9)[0]   # ALL low-prob ("critical") positions
cutpoints = cutpoints - 1                         # boundary = token before sample pos
cutpoints = cutpoints[::self.cutpoint_interval]   # stride-5 subsample (interval5.jsonnet)
# → 9 MC rollouts per cutpoint (value_estimation samples=9, full continuations)
```

So per trajectory: `c = ⌈ (#tokens with π_old<0.9) / 5 ⌉` cutpoints, **uncapped**. Total MC
rollouts/iter `= 512 × c × 9`. There is **no per-trajectory ceiling** — cost rises linearly
with how "uncertain" the model's tokens are. The exact `c` is logged as
**`episodes_metric/num_cutpoints`** (mean over the batch) — read this off iteration 1 to pin
the column. Sensitivity (assuming ~300-token mean MC continuation, as VinePPO measured for
this model):

| cutpoints/traj `c` | implied low-prob frac @ L≈400 | MC rollouts / iter | tokens / iter | vs VinePPO (34.5k / 17.3M) | vs VoI (2.6k / 0.8M) |
|---|---|---|---|---|---|
| 5 | ~6% | 23,040 | ~6.9M | 0.7× / 0.4× | 9× / 9× |
| **10** | ~13% | **46,080** | **~13.8M** | 1.3× / 0.8× | **18× / 17×** |
| 15 | ~19% | 69,120 | ~20.7M | 2.0× / 1.2× | 27× / 26× |
| 20 | ~25% | 92,160 | ~27.6M | 2.7× / 1.6× | 36× / 35× |
| 30 | ~38% | 138,240 | ~41.5M | 4.0× / 2.4× | 54× / 52× |

**Reading:** unless DeepSeekMath-7B is unusually confident (<~7.5% of tokens below ρ=0.9),
SPO-chain's MC budget **meets or exceeds VinePPO's** and is **~17–50× VoI's**. This is the
single biggest speed risk for the chain arm, and it is exactly the quantity the prob-mask ρ
controls. (The chain also runs **2 PPO epochs**, doubling the per-iter *training* compute vs
VoI/tree — minor next to MC, but real.)

---

## 6. SPO-tree credit-assignment budget (structure-determined)

Source: `hybrid_inference_strategy.py:354-371` (DFS expansion) +
`hybrid_episode_generator.py:26-62, 180-245` (edge extraction + replay buffer). Config:
`M=66`, `max_depth=3`, branch `6-6-6`, `dataset_num_samples_per_iteration=16`,
`num_episodes_per_iteration=1024`.

Per question, the tree expands (no early-stop, worst case):

| Depth | nodes | max_tokens/node | tokens (gen) |
|---|---|---|---|
| 0 | 6 | M=66 | ~396 |
| 1 | 36 | M=66 | ~2,376 |
| 2 (leaves) | 216 | **free** (until EOS / ctx 2048) | **~80,000** (216 × ~370) |
| **total/question** | **258 gens** | | **~83k tokens** |

`× 16 questions ≈ 4,128 node generations ≈ 1.0–1.3M tokens/iter` (asymmetric early-stop — any
node that answers within its budget becomes a leaf and stops — pulls this *down*; the leaf
layer still dominates). Key structural facts:

- **The tree *is* the generation** — there is no separate 8×flat rollout (contrast chain/VoI).
- **Reuse via replay buffer** (`OnPolicyReplayBuffer`, lines 26-62): edges expire after **8
  iterations** and are capped at **≤32/question/iter**, so a 216-edge question drains over
  ~7 iters. You **generate 16 questions/iter but train on ~1024 edges/iter** (reused) → 8
  optimizer steps/iter. This is the sample-efficiency lever — and the source of **stale
  advantages** (V̂/Â computed once under π_old, reused up to 8 iters).
- **Only 16 unique prompts/iter.** Over 1000 iters with-replacement, tree sees far fewer
  distinct prompt-instances per cycle than VoI/chain (64). Dense per-question, sparse across
  questions.

Net: tree's raw generation (~1.0–1.3M tok) is in VoI's league *per iter*, but per **prompt**
it is ~6× VoI, and it still pays the §7 pipeline tax.

---

## 7. The treetune pipeline tax (inherited from VinePPO)

Both SPO arms, like VinePPO, run a **serialized** loop and tear down/rebuild between phases:

```
per iteration:  start 8 per-rank vLLM (~45s) → generate/expand → MC/tree credit
                → kill_vllm_server() + release_memory()  (chain:136, tree:228)
                → rebuild DeepSpeed actor (~50s) → train (2 ep chain / 1 ep tree) → save
```

VoI's loop is **colocated and overlapped** — vLLM (TP=2 ×4) and the FSDP2 actor stay resident
across all 1000 steps; no restart, no rebuild, no FSDP↔HF conversion:

```
per step:  gen (7.6s) → VoI probe (26.6s) → train (5.4s) → repeat   [vLLM/actor always on]
```

From the VinePPO measurement, this tax is **~95s/iter** of pure overhead (≈30% of a VinePPO
iteration). SPO pays it on **both** arms.

---

## 8. Projected wall-time (analytical — to be validated)

Using the VinePPO doc's measured anchors for the *shared* treetune pipeline (vLLM restart 45s,
actor rebuild/other ~110s, train ~33s/epoch for 512 ep) and scaling credit assignment by the
token ratios above:

| Arm | credit-gen / cycle | pipeline tax / cycle | rough cycle time | vs VoI (48.5s) |
|---|---|---|---|---|
| **VoI (measured)** | 0.8M tok (26.6s) | 0s | **48.5s** | 1× |
| **SPO-chain @ c≈10** | ~13.8M tok (≈VinePPO 17.3M→289s, here ~230s) | ~95s + 2ep train | **~7–9 min** | **~9–11×** |
| **SPO-tree** | ~1.0–1.3M tok (≈30–40s) | ~95s + 1ep train | **~3–4 min** | **~4–5×** |

> These are order-of-magnitude projections, **not measurements.** The chain row assumes
> `episodes_metric/num_cutpoints ≈ 10`; it scales ~linearly with that metric (so c≈20 ≈ 13×
> VoI). The tree row assumes the leaf layer isn't heavily shortened by early-stop. **A
> 3-day allocation completes VoI's 1000 steps (~13.5h, measured); SPO-chain at ~8 min/iter ×
> 1000 = ~130h would *not* finish** — same failure mode VinePPO hit.

---

## 9. Pseudocode

### SPO-chain (DeepSeekMath-7B / MATH)
```python
for iteration in range(1000):                       # treetune outer loop
    start_vllm_servers()                             # 8 per-rank, TP=1  (~45s)
    traj = vllm_generate(64 prompts × 8, T=0.6, max_tokens=1024)   # 512 trajectories
    logp = actor_forward(traj)                        # π_old per response token
    score = math_reward(traj)                         # {0,1}

    # ── credit assignment (the expensive part) ──
    for t in traj:                                    # all 512 trajectories
        cut = where(exp(logp_t) < 0.9)[::5]           # c = ⌈N_<0.9 / 5⌉ cutpoints, UNCAPPED
        for c in cut:
            V̂[c] = mean(math_reward(vllm_generate(prefix=t[:c+1], n=9)))   # 9 MC each
        A_seg = segment_deltas(V̂, score)             # Â_k = V̂(τ_k) − V̂(τ_{k−1})
    # MC rollouts this iter = 512 × c × 9   (≈46k at c=10 → ~14M tokens)

    kill_vllm_server(); rebuild_actor()               # serialized tax (~95s)

    # ── train: prob-masked PPO, 2 epochs ──
    for epoch in range(2):
        for mb in batches(512, target=128):           # 512/128 = 4 mb × 2 = 8 optim steps
            mask = exp(old_logp) < 0.9                 # update only "critical" tokens
            ppo_step(mb, A_seg, mask, kl=1e-4_control_variate, clip=0.2)
```

### SPO-tree (DeepSeekMath-7B / MATH)
```python
for iteration in range(1000):
    start_vllm_servers()                              # 8 per-rank, TP=1  (~45s)
    # ── tree IS the generation: 16 questions, 6-6-6, M=66, depth 3 ──
    for q in sample(16):                              # only 16 unique questions/iter
        tree = dfs_expand(q, branch=[6,6,6], M=66)    # depth 0,1 ≤66 tok; depth 2 free
        #   258 node-gens/q (6 + 36 + 216 leaves); early-stop → leaf when answered
        V̂(node) = reward(node) if leaf else mean(children)        # bottom-up
        edges += [(parent→child, Â = V̂_child − V̂_parent)]        # rloo; only Â>0 kept
    # ~4,128 node generations/iter ≈ 1.0–1.3M tokens

    replay.add(edges, iteration)                       # expire >8 iters
    train_edges = replay.get(≤32 per question)         # → ~1024 episodes (REUSED, stale)

    kill_vllm_server(); rebuild_actor()               # serialized tax (~95s)

    for mb in batches(1024, target=128):              # 1024/128 = 8 optim steps × 1 epoch
        mask = exp(old_logp) < 0.9
        ppo_step(mb, A=edge.advantage, mask, kl=1e-4_control_variate, clip=0.2)
```

### VoI (DeepSeekMath-7B / MATH) — measured
```python
# vLLM (TP=2 ×4) and FSDP2 actor stay resident for all 1000 steps — no restart/rebuild
for step in range(1000):
    resp = vllm_generate(64 prompts × 8, T=0.8, top_p=0.95, max_tokens=1024)   # 512
    reward = compute_score(resp)                      # {0,1}

    # ── VoI credit: sparse, information-selected pivots (no per-token MC blowup) ──
    for group in 64_groups:                           # 8 trajs each
        b = detect_boundaries(group); s = stage1_signal(group, b)   # forward-pass only
        pivots = top_k_across_group(s, k≈5)           # ~5 pivots/GROUP (not per traj)
    cont = vllm_generate(all ~320 pivots, n=8, max_tokens=1024)      # 2,560 ≈ 0.8M tok
    A_tok = td_sparse_pivot(cont, reward)             # anchor μ_g + Bayes-shrunk residual

    for mb in batches(64 prompts, ppo_mini_batch=8):  # 8 optim steps × 1 epoch
        ppo_step(mb, A_tok, kl=1e-3_low_var, clip=0.2)
```

---

## 10. wandb keys to watch (analytical → measured)

Submit `sbatch scripts/sbatch_spo_tree_deepseek7b_MATH.sh` (tree) + an `spo_chain` analog,
then read these off wandb (`spo-math`). VoI keys are live today (cross-checked vs `voi_history.csv`
+ `wandb-summary.json`); the map validates each analytical SPO row against the same concept in
treetune. ⭐ = decides the comparison.

| Concept | verl / VoI (measured) | treetune / SPO | Validates |
|---|---|---|---|
| Time / cycle | `timing_s/step`, `perf/time_per_step` | Σ `timing/total/{episode_generation,training_step,init_policy_iteration}` | §8 |
| Main rollout | `timing_s/gen` | `timing/episode_generation/traj_inference` | §4 |
| ⭐ Credit-assign time | `timing_s/voi_probe` | chain `timing/episode_generation/value_estimation`; tree `…/inference` | §5/§6 |
| ⭐ vLLM restart tax | *(none — colocated)* | `timing/episode_generation/vllm_start` | §7 |
| ⭐ Actor rebuild tax | *(none — persistent)* | `timing/actor/deepspeed_init` + `/construct` | §7 |
| Train step | `timing_s/update_actor` | `timing/total/training_step` | §4 |
| Throughput / tokens | `perf/throughput`, `perf/total_num_tokens` | *(derive: `response_lengths/mean` × counts)* | §5/§6 |
| Response length | `response_length/{mean,max,clip_ratio}` | `response_lengths/{mean,std,dist}` | §5/§6 |
| Train reward | `critic/score/mean` | `scores/mean`, `rewards/mean` | signal |
| Value est. V̂ | `voi/v_hat_within_traj_std/mean` | `mc_values/{mean,dist}` | MC quality |
| pg_loss / entropy | `actor/pg_loss` / `actor/entropy` | `actor/loss` / `actor/logit_entropy` | optim |
| grad / clipfrac / kl | `actor/grad_norm` `/pg_clipfrac` `/ppo_kl` | `actor/grad_norm` `/clip_frac` `/approx_kl` | stability |
| KL-penalty / coef | `actor/kl_loss`, `actor/kl_coef` | `kls/mean`, `kls/crtl_var__mean` | KL regime |
| ⭐ Credit-unit count | `voi/probes_fired_per_step` (≈2,560) | chain `episodes_metric/num_cutpoints` (=`c`); tree `replay_buffer/{samples,discard_cnt}` | §5/§6 driver |
| Val accuracy | `val-core/math_v2/reward/mean@4` | in-loop greedy MATH-test + offline lighteval `extractive_match` | time-to-acc |

**ms per credit-unit** (cleanest efficiency number): VoI `timing_s/voi_probe ÷ voi/probes_fired_per_step`;
SPO-chain `timing/episode_generation/value_estimation ÷ (512 × episodes_metric/num_cutpoints)`;
SPO-tree `timing/episode_generation/inference ÷ replay_buffer/samples`.

**Caveats:** (1) the restart/rebuild keys exist **only** on the SPO side — VoI never restarts, so the
*absence* is §7. (2) treetune logs no `perf/throughput`; derive SPO tokens from `response_lengths/mean`.
(3) different x-axis (`training/global_step` vs `train/global_iteration`) and a VoI step ≠ SPO iter →
**plot accuracy vs wall-clock, not step.** (4) accuracy isn't the same measurement (VoI sampled
@T=0.35,n=4 vs SPO greedy/lighteval). (5) mind `response_length` (verl) vs `response_lengths` (treetune),
`pg_clipfrac` vs `clip_frac`.

**3-min triage, priority order:** (1) `val-core/math_v2/reward/mean@4` vs **wall-clock** — the only
headline; (2) `episodes_metric/num_cutpoints` — pins chain's §5 cost; (3) `timing/episode_generation/value_estimation`
— confirms the credit-gen bottleneck; (4) `vllm_start` + `deepspeed_init` — the §7 tax; (5)
`actor/logit_entropy` — entropy collapse.

Then redo §8 as a measured table + add accuracy-vs-wall-time (VoI reference: 34.5%@25 → 41.7%@400 →
44.5%@500 → 46.5%@1000, 13.5h total).

---

## 11. Conclusion

- **Matched** where it should be: base model, 1024-token responses, 8×H200, critic-free,
  KL-in-loss, clip 0.2, γ=1, **8 optimizer steps/cycle**, 1000 iters, seed 42.
- **The speed gap is entirely upstream of training.** VoI spends **~0.8M tok/step** on sparse,
  group-level, information-selected probes. **SPO-chain** spends an **uncapped** `512·c·9` MC
  budget that is **VinePPO-class (~14–28M tok) at any realistic cutpoint count**, i.e.
  **~17–35× VoI**. **SPO-tree** is lighter on raw tokens (~1.0–1.3M) but covers only **16
  prompts/iter** and carries **stale, reused** advantages.
- **Both SPO arms pay VinePPO's serialized pipeline tax** (~95s/iter vLLM-restart + actor-
  rebuild) that VoI's colocated design eliminates.
- **Therefore: compare on time-to-accuracy, not iterations.** Prediction: chain ≈ VinePPO
  speed (won't finish 1000 iters in 3 days); tree ~4–5× VoI/cycle. The decisive unknown is
  `episodes_metric/num_cutpoints` for chain — measure it on iteration 1 before committing a
  multi-day run.

---

## 12. Sources / provenance

- **SPO configs (resolved):** `configs/polIter_deepseekSft2_spo_chain_MATH.jsonnet`
  (+ `episode_generators/{9rolls,interval5}.jsonnet`, base
  `polIter_deepseekSft2_ppo_MATH.jsonnet`, `trainers/{ppo_MATH,lam1,refKl0.0001,klLoss,no_critic}.jsonnet`);
  `configs/polIter_deepseekSft2_spo_tree_MATH.jsonnet` (import-fixed → chain base).
- **SPO source:** `src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py`
  (cutpoints `:919-937`, kill-vllm `:136`); `…/hybrid_episode_generator.py` (replay buffer
  `:26-62`, flow `:180-245`, kill-vllm `:228`); `src/treetune/inference_strategies/hybrid_inference_strategy.py`
  (DFS expand `:354-371`).
- **Provenance:** `git diff upstream/master` (tree config import-fix = +6/−1); upstream
  `AIFrameResearch/SPO`, fork `suesie/SPO`.
- **VoI (measured):** `grpocredit/.../grpo_voi_v041_p4_mb8_T08unifAclip2_deepseekmath_math_v2_seed42`
  + `VinePPO-grpo/COMPARE_vineppo_vs_voi_speed.md` (shared treetune-pipeline timing anchors).
- **Config-level companion:** `SPO/COMPARE_spo_tree_vs_grpocredit_voi.md`.