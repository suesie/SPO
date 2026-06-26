# SPO-tree (deepseekmath-7B) — Experiment Tracking

> **Last verified: 2026-06-12 17:10 UTC** (via `squeue` / `sacct` / live log).
> Scope: the **SPO-tree** full fine-tune of `deepseekmath-7b-sft-MATH-v2` on MATH
> (seed 42), the SPO side of the apples-to-apples vs VoI / VinePPO study. The 1.5B
> sibling (`spo-tree-qwen1b-MATH-8gpu-s42`) is tracked here only where it shares a
> root cause; its own card is `RUN_spo_tree_qwen1b_MATH_seed42.md`.

## Status — prompt_512 rerun (1414137) crashed @ iter 659 (HF shuffle-cache race, fix #3) → **RESUMED** as job 1421588 from iter 650

| Job ID | Name | Config | `max_question_length` | Status | Elapsed / End (UTC) |
|--------|------|--------|---|--------|------|
| **1406496** | `spo-tree-deepseek7b-MATH-s42` | `polIter_deepseekSft2_spo_tree_MATH.jsonnet` (6-6-6, M=66) | 1512 (inherited default) | ✅ COMPLETED 1000 iters, exit 0 | 2d05h, ended 2026-06-15 00:13 |
| **1406636** | `spo-tree-qwen1b-MATH-8gpu-s42` | `polIter_qwen1b_spo_tree_MATH.jsonnet` (6-6-6, M=600) | **200 — SPO bug** | ✅ COMPLETED 1000 iters, exit 0 | 2d09h, ended 2026-06-15 06:28 |
| **1414137** | `spo-tree-qwen1b-MATH-8gpu-s42` (`…-prompt_512`) | same config, `APP_EXPERIMENT_NAME=qwen1.5b-spo-tree-666-s42-prompt_512` | **512 — fixed ✅ verified** | ❌ FAILED at **iter 659/1000** (HF `datasets` shuffle-cache race — see fix #3) | 1d13h, ended 2026-06-17 18:08 |
| **1421588** | `spo-tree-qwen1b-MATH-8gpu-s42` (`…-prompt_512`, **resume**) | same, **resumed from `ckpt--iter_0650`** after fix #3 | **512 ✅** | ✅ RUNNING (h200-037-231; resumed @ iter 650; **passed 659 & 675**; health 0) | started 2026-06-18 00:18 |

### Prompt-length bugfix rerun — `max_question_length` 200 → 512 (qwen 1.5B)

**What was wrong.** The completed qwen run **1406636** trained with `max_question_length=200`
(verified in its resolved `config.json`). That 200 is an **SPO upstream bug**: it is the GSM8K
value (`polIter_qwen05b_spo_chain_gsm8k.jsonnet:45`) carried over into the MATH config. treetune's
`_filter_init_dataset` (`src/treetune/episode_generators/on_policy_episode_generator.py:635-639`)
drops any question whose **templated** prompt (`[MATH_TASK] Problem:\n{query}\n\nSolution:`)
tokenizes to `> max_question_length`; the ~85-token instruction eats into the 200 budget so this
**silently dropped ~12% of MATH train**. Every other MATH config is larger — base default **1512**
(`episode_generators/math_episode_generator.jsonnet:37`), the 7B run inherits **1512**, and
`polIter_qwen1_5b_base_spo_chain_MATH.jsonnet:50` uses **512**.

**Fix.** `configs/polIter_qwen1b_spo_chain_MATH.jsonnet:59` now sets `max_question_length: 512`
(local, **uncommitted** edit — git HEAD `1e64f0c` still has 200; that is why 1406636, launched
before the edit, resolved 200). 512 matches both the base qwen-1.5B MATH config and the VoI
baseline `data.max_prompt_length=512`.

**Rerun.** Job **1414137** — 8×H200 exclusive (compute-matched to 1406636), seed 42,
`APP_EXPERIMENT_NAME=qwen1.5b-spo-tree-666-s42-prompt_512`:
- output dir (fresh, no collision): `…/spo_tree_qwen1b_MATH_seed42/qwen1.5b-spo-tree-666-s42-prompt_512/`
- wandb run `qwen1.5b-spo-tree-666-s42-prompt_512` in project `GRPO_mcts-vineppo`
- **❌ FAILED at iter 659/1000** on h200-133-210 (started 2026-06-16 04:19, ended 2026-06-17 18:08:30,
  1d13h; pended only ~13 min on `QOSGrpCpuLimit`). The prompt-length fix itself was sound — the crash
  was an unrelated HF `datasets` shuffle-cache race (see "Root cause & fix #3").
  **Verified at startup:** resolved `config.json` has `max_question_length=512`; guidance cache on
  `/dev/shm/guidance_cache_1414137`; **cleared iteration 0** (04:26:11); health greps
  (offline / SQLite ×2 / rc=1) all **0**.

The completed **200-run (1406636) is kept** as the buggy baseline for the 200-vs-512 comparison.
For the 7B side the analogous gap (1512 vs VoI's 512) was measured at only **0.8% of MATH train**
(93/11,500 questions > 512 tok), so the 7B was **not** rerun.

> **qwen walltime fix.** The first qwen resubmit (**1406501**, 2-day cap) measured ~4.6 min/iter →
> would have TIMED OUT at ~iter 650/1000. Per user request it was **cancelled, its 28 GB
> output/checkpoints deleted**, and **relaunched as 1406636 with `--time=7-00:00:00`** (= partition max;
> QOS has no MaxWall) — enough for the full 1000 iters (~3.2 days). `sbatch_spo_tree_qwen1b_MATH_8gpu.sh`
> `--time` is now 7d (durable). Both runs use their own per-job `/dev/shm` guidance cache — the isolation
> fix is confirmed live (0 SQLite / offline / rc=1 errors); both previously cleared the iter-0
> cache-write phase that killed the original attempts.

- **wandb:** treetune **hardcodes** the project to `GRPO_mcts-vineppo` (overrides `WANDB_PROJECT=spo-math`);
  the new runs' wandb URLs appear there once they start.
- **Output dirs:** 7B `…/spo_tree_deepseek7b_MATH_seed42/`, qwen `…/spo_tree_qwen1b_MATH_seed42/`.
- **Deletions (2026-06-12, user-requested; slurm logs kept):** failed **1405148** (7B, 203 GB incl.
  iter-200 ckpt); cancelled **1406501** (qwen, 28 GB — superseded by the 7-day relaunch 1406636). All
  current runs **start from scratch**.
- **Launch:** `sbatch scripts/sbatch_spo_tree_deepseek7b_MATH.sh` → `launch_server2_spo_multi.sh spo_tree`
- **Env:** conda `vineppo` (torch2.1.2 / vllm0.4.0.post1 / ds0.14.1), NCCL 2.18.1.

### Resources (whole node, per request)
- `--gres=gpu:h200:8`, `--cpus-per-task=96`, `--exclusive`, `--mem=0` (≈2TB node RAM),
  `--time=7-00:00:00`, account/qos `h200_mrs_2` / `h200_mrs_2_high`.
- `--mem=0 --exclusive` deliberately grabs the **whole node** (VinePPO OOM'd at `--mem=512G`;
  8 per-rank vLLM × swap_space 32G = 256G + ZeRO-2 staging + tree/MC episode data).

### Key config knobs (resolved via `_jsonnet`)
| knob | value | | knob | value |
|---|---|---|---|---|
| base model | `deepseekmath-7b-sft-MATH-v2` (local snapshot) | | response len (`max_tokens`) | 1024 |
| estimator | SPO-tree segment-MC, `use_prob_mask=true` | | tree | 6-6-6, `max_depth=3`, `M=66` |
| episodes/iter | 1024 (16 questions × 64) | | optim batch | target 128, per-dev 8 |
| learning rate | 1e-6 (warmup_ratio 0.03) | | KL | in-loss, `init_kl_coef=1e-4` |
| train temp / top_p | 0.6 / 0.9 | | total iters | 1000 |
| eval cadence | every 25 iters (MATH-validation, n=16) | | save / keep | 25 / 25 (modulus) |
| critic | none (`critic_model=null`) | | seed | 42 |

## Launch history (this run)

| Job ID | Started (UTC) | State | Elapsed | Node | Outcome |
|--------|---------------|-------|---------|------|---------|
| 1404435 | 06:51:23 | ❌ FAILED (1:0) | 16m24s | h200-233-072 | iter-0 `OfflineModeIsEnabled` — vLLM got repo-ID |
| 1405130 | 18:36:13 | ⏹ CANCELLED | 11m50s | h200-228-229 | same crash; first fix attempt (guidance_llm only) was insufficient — cancelled |
| 1405148 | 18:53:36 | ❌ FAILED (1:0) | 11h44m45s | h200-225-064 | local-snapshot fix worked; trained to **iter 209/1000**, then guidance SQLite cache corruption killed all ranks (see "Run-1405148 failure" below) |
| **1406496** | 2026-06-12 18:47 | ✅ RUNNING | — | h200-027-239 | resubmit **with cache-isolation fix** (cache on `/dev/shm` ✅, 0 sqlite errors); pended ~13 min on `QOSGrpGRES`; fresh start |

### Root cause & fix #1 — iter-0 `OfflineModeIsEnabled` (jobs 1404435 / 1405130)
- **Symptom:** all 8 per-rank vLLM servers crash-loop at iter-0 model load with
  `huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/realtreetune/deepseekmath-7b-sft-MATH-v2`.
- **Cause:** the sbatch sets `HF_HUB_OFFLINE=1` (model is cached, avoids proxy flakiness).
  vLLM 0.4.0 resolves a **repo-ID** via `HfFileSystem.ls()` (a Hub API call) which raises
  under offline mode. SPO's `on_policy_episode_generator.py` passes
  `initial_model_name_or_path` (a repo-ID) **straight to vLLM** at iter 0
  (`hf_ckpt_path_or_model = self.initial_model_name_or_path` when `latest_policy_path is None`).
  VinePPO's fork resolves the repo-ID to the local HF-cache snapshot first; SPO's fork does not.
  (A local dir → vLLM `is_local=True` → no Hub call. transformers loads are offline-cache-safe
  either way, which is why only vLLM weight loading broke.)
- **Fix:** point every model/tokenizer reference at the local snapshot dir
  `/lustre-storage/datasets/zengh/huggingface/hub/models--realtreetune--deepseekmath-7b-sft-MATH-v2/snapshots/8b387c255b3bfaaaef2e650d56fecfde1c56ea96`:
  - `configs/polIter_deepseekSft2_ppo_MATH.jsonnet` — `local hf_model_name` (drives
    `initial_model_name_or_path`, actor/reference, main tokenizer). **This is the one that fixes the crash.**
  - `configs/sft_deepseekmath_for_MATH_eval.jsonnet` — eval `tokenizer.hf_model_name` (+ mean_at_k).
  - `configs/guidance_llms/deepseekmath7b-sft-MATH-v2.jsonnet`, `deepseekmath7b_base.jsonnet` —
    `model` (+ `tokenizer_name`); defensive (runtime-overridden for episode gen, but covers eval).
  - Verified: resolved config has **zero** bare repo-ID references.
- Failed experiment dirs `*.failed-1404435/` and `.failed-1405130/` were **deleted 2026-06-12**
  (user-requested cleanup); their slurm logs (`sbatch_slurm-1404435.log`, `-1405130.log`) are kept.

### Root cause & fix #2 — guidance SQLite cache corruption (jobs 1405148 + 1405981)
- **Symptom (7B, 1405148):** trained cleanly to **iter 209/1000** (~11h45m), then every
  generation program threw `sqlite3.DatabaseError: database disk image is malformed` on a
  cache **write** (`src/guidance/llms/_openai_vllm.py:896` → `caches/_diskcache.py:22`).
  Hundreds of `Error in program: database disk image is malformed` cascaded into async
  teardown noise (`RuntimeError: coroutine ignored GeneratorExit`,
  `AttributeError: 'NoneType' object has no attribute 'create_future'`); DeepSpeed killed all
  ranks, rc=1 @ 06:38:21.
- **Sibling (1.5B, 1405981 `spo-tree-qwen1b-MATH-8gpu-s42`):** started 06:34:22 on
  h200-016-059, died rc=1 @ 06:38:26 (~4 min, **iter 0**) with `sqlite3.DatabaseError: file is
  not a database` on cache **open** (`_openai_vllm.py:803` → `_diskcache.py:12`), via
  `EfficientIIDExpander._sample_node` — despite `no_cache: true` in the config (that flag does
  not disable the guidance-level diskcache). The two jobs died **within 5 s of each other**.
- **Cause:** the guidance diskcache is keyed only by the LLM **class** name, so it resolves to a
  single shared SQLite db `~/.cache/guidance/_openai_vllm.diskcache/cache.db` (had grown to
  **1.1 GB**) on the **network home FS** (FSx). The 7B run's 8 ranks (one node) were stable on it
  for 11.75 h; corruption hit the instant the qwen run's 8 ranks (a **second node**) opened the
  same file → 16 **cross-node** concurrent SQLite writers, whose POSIX advisory locking is
  unreliable over a network FS → DB image corrupted, taking down both jobs.
- **Ruled out:** disk-full (FSx 87 % used, 4.8 TB free) and the offline bug (0 `OfflineModeIsEnabled` in both logs).
- **Fix applied (this change):**
  1. `src/guidance/llms/caches/_diskcache.py` now honors a `GUIDANCE_CACHE_DIR` env var
     (falls back to the original `platformdirs` path when unset — backward-compatible).
  2. All three SPO launchers (`sbatch_spo_tree_deepseek7b_MATH.sh`,
     `sbatch_spo_tree_qwen1b_MATH_8gpu.sh`, `sbatch_spo_tree_qwen1b_MATH.sh`) export a
     **per-job, node-local** cache:
     `GUIDANCE_CACHE_DIR=/dev/shm/guidance_cache_$SLURM_JOB_ID` (RAM-backed tmpfs; falls back to
     a per-job `$TMPDIR` if `/dev/shm` is unavailable), with an `EXIT` trap that removes it.
     → each job gets its own cache on local storage; no cross-job/cross-node shared SQLite.
  3. Cleared the corrupt shared cache: `rm -rf ~/.cache/guidance/_openai_vllm.diskcache`.
- **Cleanup (2026-06-12, user-requested):** all failed-run output dirs deleted — 1405148 (203 GB,
  incl. the iter-200 ckpt), `.failed-1404435`, `.failed-1405130`, and qwen `qwen1.5b-spo-tree-666-s42`
  (1405981). Slurm logs retained in each seed dir.
- **Not yet done:** no relaunch submitted (awaiting user); with the checkpoints deleted a relaunch
  **starts from scratch**. The fix lives in the working tree (the `_diskcache.py` edit + the sbatch
  `GUIDANCE_CACHE_DIR` exports — the sbatch scripts are untracked scaffolding, run directly from the
  checkout), so just verify the run's log prints a per-job `GUIDANCE_CACHE_DIR=/dev/shm/...` line and
  that the health greps stay 0.

### Root cause & fix #3 — HF `datasets` shuffle-cache race (job 1414137, iter 659)
- **Symptom:** the prompt-512 rerun trained cleanly through **iter 658/1000** (~1d13h), then died
  rc=1 @ 2026-06-17 18:08:30 at the start of iter 659 episode generation with a single
  `FileNotFoundError: [Errno 2] No such file or directory: '.../data/math/train/cache-edf7937b847f91db.arrow'`
  raised by `os.chmod(indices_cache_file_name, ...)` in `datasets==2.17.1`
  `_select_with_indices_mapping` (`arrow_dataset.py:3991`), reached via
  `on_policy_episode_generator.py:291` → `dataset.shuffle(seed=self.seed + iteration)`.
- **Cause:** all 8 DeepSpeed ranks run `generate()` and call `.shuffle()` with the **same** seed
  (`42 + iteration`) on the **same** dataset, so HF derives one **deterministic** indices-cache
  filename (`cache-<fingerprint+seed>.arrow`) and all 8 ranks write it into the **shared**
  `data/math/train/` dir — a symlink to **lustre** (network FS). HF writes a temp file → atomic
  `shutil.move` → `os.chmod`. Under 8-way concurrent move+chmod on lustre, one rank's `os.chmod`
  hit a window where the name was momentarily absent (superseded by another rank's move / metadata
  lag) → `FileNotFoundError`; that rank died, DeepSpeed killed all 8 → rc=1. Same **family** as
  fix #2 (many ranks sharing one cache file on a network FS), here the HF `datasets` shuffle cache
  rather than the guidance SQLite cache.
- **Why it survived 658 iters / why 1406636 never hit it:** rare probabilistic race — occurred
  exactly **once** in the whole run (1× `FileNotFoundError`). The earlier identical 200-run
  **1406636** completed all 1000 iters with **0** occurrences; it just got lucky on timing.
- **Ruled out:** disk-full (lustre 54 % used, 104 TB free), fix #1 (0 `OfflineModeIsEnabled`),
  fix #2 (0 SQLite errors).
- **Fix applied (this change):** `on_policy_episode_generator.py:291` now calls
  `dataset.shuffle(seed=self.seed + iteration, keep_in_memory=True, load_from_cache_file=False)`.
  With `keep_in_memory=True`, `shuffle` passes `indices_cache_file_name=None` and
  `_select_with_indices_mapping` writes to an in-memory `BufferOutputStream` (`arrow_dataset.py:3945`),
  so the `shutil.move`+`os.chmod` block (`:3986-3991`) is **skipped entirely** — no shared file, no
  race; `load_from_cache_file=False` also ignores the ~2.7k pre-existing `cache-*.arrow` files. The
  shuffle is seeded identically on every rank, so the permutation (hence the per-iteration question
  subset) stays **identical across ranks** — verified by a standalone repro on `datasets==2.17.1`
  (OLD = 1 cache file + 1 chmod; NEW = 0/0; same permutation). Other `.shuffle()` calls in this file
  are not at risk: 159/173/194/206 run once under `main_process_first()`; 424/436 run only on the
  main process; the `.select(range(...))` at 283/293 use the contiguous fast path (no indices cache).
- **Resolved:** resumed as job **1421588** (see below); the ~2.7k stale `data/math/train/cache-*.arrow`
  files (203 MB) were pruned 2026-06-18 (source `data-00000-of-00001.arrow` + the two JSONs kept).

### Resume after fix #3 — job 1421588 (resumed from iter 650)
Submitted 2026-06-18 00:18 UTC, **8×H200** on h200-037-231, seed 42,
`APP_EXPERIMENT_NAME=qwen1.5b-spo-tree-666-s42-prompt_512` (⚠️ **must pass this explicitly** — the sbatch
defaults `APP_EXPERIMENT_NAME` to the non-suffixed `…-s42`, which would target the wrong experiment dir).
Runs from the same working tree as 1414137 **plus the fix #3 edit** (executed directly from the checkout;
HEAD `6c5f947` + uncommitted `on_policy_episode_generator.py`).
- **Resume contract (verified in code):** `run_iteration_loop` →
  `trainer.get_last_checkpoint(return_resumable_only=True)` scans only `checkpoints/` for `ckpt--*`
  whose `actor/` holds DeepSpeed optimizer state (`ppo_trainer.py:is_checkpoint_resumable`), picking
  **iter 650** (`step_2600`, last cadence-25 ckpt; complete — 8 ZeRO optim shards + model state +
  `hf_pretrained`, written 17:36, 30 min pre-crash). The suspect in-flight
  `temp_ppo_checkpoints/ckpt--iter_0659` is in a different dir → ignored, and was auto-removed by
  `_clean_old_temp_checkpoints` on resume. `state.iteration` restored from `custom_checkpoint_0.pkl`
  → `starting_iteration=650`; iters 651–659 are recomputed (≤10 iters, ~30 min — expected for
  cadence-25 checkpointing).
- **Live verification:** log shows `**** Resuming from iteration 650 ****`, per-job
  `GUIDANCE_CACHE_DIR=/dev/shm/guidance_cache_1421588`, wandb `GRPO_mcts-vineppo/runs/hkrnmusd`.
  **Passed the original crash point (iter 659) and the iter-675 in-training eval**, reached iter ~688+
  with **all health greps 0** (offline / SQLite ×2 / `FileNotFoundError` / `os.chmod` / rc=1 / Traceback);
  new `ckpt--iter_0675` written → post-resume checkpointing + eval both healthy.

## Monitoring commands
```bash
JOB=<jobid>   # e.g. the next relaunch
# status
squeue -j "$JOB" --format="%.10i %.10T %.12M %.12L %R"
sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList

# live progress
LOG=/lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed42/sbatch_slurm-${JOB}.log
grep -aE "Running iteration|PPO training step|val-core|reward/mean" "$LOG" | tail
# health (all must stay 0): offline bug, SQLite cache corruption (x2), hard exit
for p in "OfflineModeIsEnabled" "database disk image is malformed" "file is not a database" "os.chmod(indices_cache_file_name" "FileNotFoundError" "exits with return code = 1"; do
  printf "%-34s " "$p"; grep -ac "$p" "$LOG"; done
# confirm the cache fix is active (must print a per-job /dev/shm path, NOT ~/.cache/guidance)
grep -a "GUIDANCE_CACHE_DIR=" "$LOG" | head -1
```

## Comparison context (tracked elsewhere)
- **VoI** (verl, completed reference): see `COMPARE_spo_tree_vs_grpocredit_voi.md`.
- **VinePPO** baselines (same treetune framework, separate repo/tracker):
  `1404430` epoch2/official, `1404431` epoch1/fair — `VinePPO-grpo/TRAINING_RUNS_TRACKING.md`.
  SPO-tree uses `num_epochs_per_iteration=1`, so the apples-to-apples VinePPO is **epoch1 (1404431)**.
- Compare by **time-/step-to-accuracy** (`val-core/math_v2/reward/mean@k`), not raw iterations.

---

# GRPO baseline (R1-Distill-Qwen-1.5B) — Experiment Tracking

> **Last verified: 2026-06-23 04:48 UTC** (via `squeue` + live log + resolved `config.json`).
> Scope: the **official upstream GRPO** baseline on MATH (seed 42), the GRPO column of the
> SPO paper's qwen-1.5B table. Same treetune framework / cluster ops as the SPO-tree runs above;
> shares root causes & fixes #1–#3 (offline snapshot, guidance SQLite cache, HF shuffle-cache).

## Status — job 1458990 launched 2026-06-23 04:43 UTC → ✅ RUNNING, iter-0 healthy

| Job ID | Name | Config | `max_question_length` | Status | Node / Start (UTC) |
|--------|------|--------|---|--------|------|
| **1458990** | `grpo-qwen1b-MATH-official-8gpu-s42` | `polIter_qwen1b_grpo_MATH.jsonnet` **+ `episode_generators/max_question_length_200.jsonnet`** | **200 — official ✅ (reset from local 512)** | ✅ RUNNING (h200-062-093; vLLM ×8 up, iter-0 episodes generating, health 0) | started 2026-06-23 04:43:12 |

### What this run is — the OFFICIAL GRPO setting (not the local 512 tree)

The committed code is upstream-clean (`HEAD 6c5f947` == `origin/master` == `upstream/master`; `polIter_qwen1b_grpo_MATH.jsonnet` itself unmodified). The GRPO config imports the shared parent
`polIter_qwen1b_spo_chain_MATH.jsonnet`, which carries the **local uncommitted** `max_question_length 200→512`
edit (the same bugfix discussed in the SPO-tree section). To reproduce the **upstream** GRPO faithfully
**without** reverting that working-tree edit, this run layers a last-wins override:

- **New override:** `configs/episode_generators/max_question_length_200.jsonnet` (sets `episode_generator.max_question_length: 200`).
- treetune merges `--configs` with jsonnet `+`, so the override (layered last via `EXTRA_CONFIGS`) wins.
- **Verified live:** resolved `…/qwen1.5b-grpo-official-8gpu-s42/config.json` has `max_question_length=200`,
  `type=math_episode_generator_w_group_advantages`, `adv_method=grpo`.
- Only `max_question_length` is touched. Other working-tree deltas are **training-equivalent**: model path
  is a content-identical local snapshot (HF `refs/main == ad9f0ae0`, offline-safe), the in-memory
  `dataset.shuffle` uses the same seed (same permutation), and `save/keep=25/25` is checkpoint-cadence only.

### Multi-GPU (8) correctness — same setting, not bit-identical

Same official GRPO **setting** as a 1-GPU run, statistically equivalent, ~Nx faster episode gen:
- **Global batch held constant:** `target_train_batch_size=128`, `per_device=2` ⇒ grad-accum auto
  `128//2//world` = **8** on 8 GPUs (64 on 1 GPU); global = `2×8×8` = **128** (exact) ⇒ **4 optim steps/iter**.
  (`restem_trainer.py:162-179`; `ppo_trainer.py:505-510`.)
- **Same prompts/iter:** 64 questions selected globally via `shuffle(seed+iteration)` **before** the per-rank shard.
- **GRPO groups intact:** questions are sharded **contiguously**, so all 8 rollouts of a prompt stay on one
  rank; group mean/std is per-prompt (`math_episode_generator_with_group_advantages.py:271-280`) ⇒ no bias.
- **NOT bit-identical:** rollout seed is per-rank `seed = base + rank*100 + iter`
  (`on_policy_episode_generator.py:339`) — **confirmed live** in the log: rank0=42, rank4=442, rank5=542,
  rank7=742 — and 8 vLLM servers batch differently than 1, so the *sampled* trajectories differ
  (run-to-run / seed-level variance, not a systematic difference). The paper used 1× A100, so the *literal*
  reproduction is the single-GPU `sbatch_grpo_qwen1b_MATH_official.sh`; this 8-GPU variant reproduces the
  **setting**.

### Resources / launch / env
- `--gres=gpu:h200:8`, `--cpus-per-task=96`, `--exclusive`, `--mem=0`, `--time=3-00:00:00` (GRPO has no
  tree/MC value rollouts, so episode gen is much lighter than SPO-tree → 1000 iters well under 3 days),
  account/qos `h200_mrs_2` / `h200_mrs_2_high`. Scheduled immediately (0 min pending; 17 idle nodes).
- **Launch:** `sbatch scripts/sbatch_grpo_qwen1b_MATH_official_8gpu.sh` →
  `launch_server2_spo_multi.sh grpo` (`MODEL_FAMILY=qwen1b`, `EXTRA_CONFIGS=…/max_question_length_200.jsonnet`).
  Single-GPU sibling: `scripts/sbatch_grpo_qwen1b_MATH_official.sh`.
- **Env:** conda `vineppo` (torch 2.1.2+cu121 / vllm 0.4.0.post1 / ds 0.14.1), NCCL 2.18.1; `HEAD 6c5f947`.
- **Output dir:** `/lustre-storage/checkpoints/zengh/spo/grpo_qwen1b_MATH_official_seed42/`
  (sub-dir `qwen1.5b-grpo-official-8gpu-s42/`). Per-job cache `GUIDANCE_CACHE_DIR=/dev/shm/guidance_cache_1458990`.
- **wandb:** treetune **hardcodes** project `GRPO_mcts-vineppo` (overrides `WANDB_PROJECT=spo-math`);
  entity `suesie`, run name `qwen1.5b-grpo-official-8gpu-s42`.

### Key config knobs (resolved via `config.json`)
| knob | value | | knob | value |
|---|---|---|---|---|
| base model | `DeepSeek-R1-Distill-Qwen-1.5B` (local snapshot `ad9f0ae0`) | | response len (`max_tokens`) | 4096 |
| estimator | GRPO group adv, `use_prob_mask=false` | | model_context_size | 2048 (2K) |
| episodes/iter | 512 (64 questions × 8 rollouts) | | optim batch | target 128, per-dev 2 |
| learning rate | 1e-6 (warmup_ratio 0.03) | | KL | in-loss, `init_kl_coef=1e-4` (control_variate) |
| train temp / top_p | 0.6 / 1.0 | | total iters | 1000 |
| `max_question_length` | **200 (official, via override)** | | save / keep | 25 / 25 (modulus) |
| critic | none | | seed | 42 |

### Live verification (iter 0, 04:48 UTC)
- resolved `config.json`: `max_question_length=200`, `adv_method=grpo` ✅
- all 8 per-rank vLLM servers started on the **local snapshot** (`ad9f0ae0`) — 0 `OfflineModeIsEnabled` ✅
- iter-0 episode generation underway (~34k tok/s aggregate; `reward: 1.0` logged) ✅
- per-job `GUIDANCE_CACHE_DIR=/dev/shm/guidance_cache_1458990` ✅
- health greps all **0**: offline / `database disk image is malformed` / `file is not a database` /
  `FileNotFoundError` / `Traceback` / `return code = 1` ✅
- ⏳ pending next check: first `ckpt--iter_0025` written; in-training MATH eval; reward/accuracy trend.

### Monitoring (this run)
```bash
JOB=1458990
squeue -j "$JOB" --format="%.10i %.10T %.12M %.12L %R"
sacct -X -j "$JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
LOG=/lustre-storage/checkpoints/zengh/spo/grpo_qwen1b_MATH_official_seed42/sbatch_slurm-8gpu-${JOB}.log
grep -aE "Running iteration|PPO training step|val-core|reward/mean|ckpt--iter" "$LOG" | tail
for p in "OfflineModeIsEnabled" "database disk image is malformed" "file is not a database" "FileNotFoundError" "return code = 1"; do printf "%-34s " "$p"; grep -ac "$p" "$LOG"; done
grep -a "GUIDANCE_CACHE_DIR=" "$LOG" | head -1   # must be a per-job /dev/shm path
```

### Files created for this run (all untracked scaffolding; run from the checkout)
- `configs/episode_generators/max_question_length_200.jsonnet` — the upstream-200 override.
- `scripts/sbatch_grpo_qwen1b_MATH_official_8gpu.sh` — 8×H200 launch (this run).
- `scripts/sbatch_grpo_qwen1b_MATH_official.sh` — single-GPU sibling (literal paper reproduction).


# Why best@16 increasing but for voi and my implemented grpo, it's decreasing?
