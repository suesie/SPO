# SPO-tree (6-6-6) reproduction — DeepSeek-R1-Distill-Qwen-1.5B / MATH → MATH500 (seed 42)

**Goal:** reproduce the headline SPO-tree row of the SPO paper (NeurIPS 2025,
arXiv 2505.23564) on **MATH500**, training from scratch on the H200 cluster with
the same `treetune` framework you used for VinePPO-grpo, then evaluating with the
authors' `lighteval`+`vLLM` long-CoT script.

**Target numbers** (upstream README table; verified against `results/SPO-tree-666/*.json`,
metric = `extractive_match` on `custom|math_500|0`):

| Context | Base (R1-Distill-1.5B) | GRPO | **SPO-tree (this run)** |
|---|---|---|---|
| 2K (`max_new_tokens=2048`)  | 0.566 | 0.62  | **0.736** |
| 4K (`max_new_tokens=4096`)  | 0.740 | 0.752 | **0.828** |
| 32K (`max_new_tokens=32768`)| 0.838 | 0.84  | **0.848** |

> The published checkpoint `gyr66/spo-tree-666-qwen1.5B-math` corresponds to the
> JSON `model_name = Qwen1.5B-MATH-C4096-400` → trained up to **4K context**,
> **iteration ~400** (not the full 1000). Expect the headline 0.736/0.828 around
> that checkpoint, not necessarily at iter 1000.

> **Status:** ⬜ not started. This machine has **no GPU** — every numbered step
> below runs on the H200 cluster (or a login node with internet for downloads).

> 🔧 **Local additions (suesie fork) vs upstream `AIFrameResearch/SPO@1e64f0c`.**
> Unlike the 7B-deepseek workflow, the qwen1b SPO-tree path is **upstream-clean**: every
> `configs/polIter_qwen1b_*.jsonnet`, `configs/qwen1b_for_MATH_eval.jsonnet`, and the
> training entrypoint `src/treetune/main.py` are used **unmodified**. The only local
> additions are two scaffolding shell scripts (and a `data/` symlink) — marked 🔧 inline:
> 1. **`scripts/sbatch_spo_tree_qwen1b_MATH.sh`** (new) — SLURM submit wrapper that
>    pins our cluster's account/qos, lustre HF caches, NCCL 2.18.1 guard, and 1×H200.
> 2. **`scripts/launch_server2_spo.sh`** (new) — single-GPU deepspeed entrypoint that
>    mirrors the upstream README invocation; injects **no** hyperparameters.
> 3. **`data/` → `/lustre-storage/datasets/zengh/spo/data` symlink** — keeps the
>    treetune-expected repo-local `data/` lookup while storing the actual arrow files
>    on lustre.

---

## 0. Repo / remotes (already done)

```
~/projects/SPO   origin=https://github.com/suesie/SPO   upstream=https://github.com/AIFrameResearch/SPO   branch=master
```

Keep your fork synced later with: `git fetch upstream && git merge upstream/master`.

---

## 1. What this reproduces (config provenance)

- **Config:** `configs/polIter_qwen1b_spo_tree_MATH.jsonnet` (unmodified upstream)
  → imports `configs/polIter_qwen1b_spo_chain_MATH.jsonnet`. Resolved knobs:
  - **base model** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - **tree** branch factors **6-6-6**, `max_depth=3`, `M=600` (global node budget; 6-6-6
    tops out at ~258 leaves so M=600 is a safety cap, not the binding constraint);
    **4 MC rollouts per leaf** (`value_estimation_inference_strategy.samples=4`,
    inherited from `polIter_qwen1b_spo_chain_MATH.jsonnet:14`)
  - 512 episodes/iter; 8 trajectory rollouts/sample; **1000 iterations**
  - rollout `T=0.6`, `max_tokens=4096`, `model_context_size=2048` (hard ceiling on
    total prompt+response; the README's "2K→4K context schedule" is **not** encoded in
    this config — see §10 pitfall #7)
  - **`use_prob_mask=true`** (the SPO probability-mask policy update), `lam=1`,
    `init_kl_coef=1e-4` (KL **loss**, not reward), critic-less
  - **DeepSpeed ZeRO-0** (no offload — 1.5B fits one 80GB+ GPU)
  - `target_train_batch_size=128`, `per_device_train_batch_size=2`,
    `save_steps=5`, `checkpoint_keep_steps=10`, 1 epoch/iter
- **In-training eval:** `configs/qwen1b_for_MATH_eval.jsonnet` runs treetune's own
  full MATH-**test** pipeline (`T=0`, `n=1` greedy, `max_tokens=4096` but capped by the
  same `model_context_size=2048` as training, so effective new-tokens ≤ 2048−prompt)
  periodically. Useful as a live signal, but the **published table comes from the
  separate `lighteval` eval in §7.**

---

## 2. Cluster / resource spec — SINGLE GPU (important)

SPO-tree is a **single-GPU** method: the paper used **1× A100 80GB**, and the
README launches with `deepspeed --include localhost:0`. The episode-generation
vLLM server and the trainer **share one GPU** (`wait_until_memory_release=true`
frees vLLM VRAM before the optimizer step). So, unlike the VinePPO-grpo 8×H200
runs, request **one** H200:

| SBATCH | value | note |
|---|---|---|
| `gres` | `gpu:h200:1` | 141GB ≫ the paper's 80GB → comfortable |
| `cpus-per-task` | 24 | |
| `mem` | 256G | 1 vLLM `swap_space=32G` + ZeRO-0 staging + episodes |
| `exclusive` | **no** | leaves the node's other 7 GPUs free |
| `time` | 3-00:00:00 | 1000 iters single-GPU is multi-day; checkpoints as it goes |

Submit script: 🔧 **`scripts/sbatch_spo_tree_qwen1b_MATH.sh`** (new, ours; mirrors
your VinePPO sbatch: lustre caches, `HF_HUB_OFFLINE`, `DS_SKIP_CUDA_CHECK`, NCCL guard).
Launch wrapper: 🔧 **`scripts/launch_server2_spo.sh`** (new, ours; single-GPU,
`gpu_0.jsonnet`; no hyperparameter injection).

---

## 3. Environment

Training uses the **VinePPO stack** (`requirements.txt`: torch 2.1.2+cu121, the
McGill `vllm-0.4.0.post1` wheel, flash-attn 2.5.5, deepspeed 0.14.1,
transformers 4.38.1). Your existing **`vineppo`** env already matches this.

> ⚠️ **Train vs eval dependency split — do NOT mix.** `requirements.txt` appends
> an **unpinned `lighteval`** at the end. `lighteval` (and the `custom|math_500`
> open-r1 task) wants a **much newer** `transformers`/`vllm` than the pinned
> training stack; installing it into the training env will upgrade those and
> break training. **Use two envs:**
> - **`spo`** (or reuse **`vineppo`**) for TRAINING — pinned stack, *without* lighteval.
> - **`spo-eval`** for EVALUATION — fresh, recent `lighteval`+`vllm` (§6).

### 3a. Training env

**Option A — reuse your existing `vineppo` env (fastest):**
```bash
conda activate vineppo
python -c "import torch,vllm,deepspeed; print(torch.__version__, vllm.__version__, deepspeed.__version__)"
# expect: 2.1.2+cu121 0.4.0.post1 0.14.1
```

**Option B — fresh `spo` env (matches the README; excludes lighteval to protect pins):**
```bash
conda create -n spo python=3.10 -y && conda activate spo
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate && conda activate spo
grep -v '^lighteval' requirements.txt > /tmp/spo_train_reqs.txt   # keep training pins clean
pip install -r /tmp/spo_train_reqs.txt
```

**NCCL overlay guard (same pitfall as VinePPO §1.3).** torch 2.1.2 bundles NCCL
2.18.1+cuda12.1; if it got overlaid the first collective crashes. Verify / fix:
```bash
strings "$(python -c 'import os,nvidia.nccl as n;print(os.path.dirname(n.__file__)+"/lib/libnccl.so.2")')" | grep "NCCL version"
# want: NCCL version 2.18.1+cuda12.1 ; if not:
pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.18.1
```

### 3b. Eval env (`spo-eval`) — see §6
```bash
conda create -n spo-eval python=3.11 -y && conda activate spo-eval
pip install lighteval[vllm] math-verify
# (open-r1's math_500 custom task — see §6 for the --custom-tasks file)
```

---

## 4. Data + base model  ✅ DONE (staged on lustre)

Both are downloaded and verified on lustre (grpocredit scheme). Nothing to re-run
unless the cache is cleared.

### 4a. Dataset (treetune format) — on lustre, symlinked into the repo
Downloaded via `scripts/download_and_prepare_dataset.sh` (upstream) run from
`/lustre-storage/datasets/zengh/spo`, then 🔧 symlinked locally so treetune's
repo-local `data/` lookup resolves to lustre:
```
/lustre-storage/datasets/zengh/spo/data/   ->   ~/projects/SPO/data  (symlink)
  math/{train,test,validation}/*.arrow + dataset_dict.json   # what SPO-tree MATH reads
  gsm8k/  collegeMath/  olympiadbench/  point24/  point24-train_test_valid/
```
(Source bundle: `wandb_export_root.zip`, 14M, Aliyun OSS Hong Kong. Verified:
`~/projects/SPO/data/math` reachable; train/test/validation splits present.)

### 4b. Base model — in the lustre HF cache
```
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   (9 files, model.safetensors = 3.4G)
HF_HOME=/lustre-storage/datasets/zengh/huggingface
snapshot: …/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
```
With the model cached + `HF_HOME` set + `HF_HUB_OFFLINE=1` (both in the sbatch),
treetune/vLLM resolve the hub name to this local snapshot — no `local_model_path`
config needed. To re-download later:
```bash
HF_HOME=/lustre-storage/datasets/zengh/huggingface \
  huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

---

## 5. Smoke test (≈10 min, 1 GPU) — do this before the multi-day run

Grab a short interactive GPU and run a couple of iterations to validate
env + NCCL + data + config end-to-end:
```bash
srun --account=h200_mrs_2 --qos=h200_mrs_2_high --gres=gpu:h200:1 \
     --cpus-per-task=24 --mem=128G --time=00:30:00 --pty bash
# inside the allocation:
source /home/zengh/miniconda3/etc/profile.d/conda.sh && conda activate vineppo   # or spo
cd ~/projects/SPO
export HF_HOME=/lustre-storage/datasets/zengh/huggingface HF_HUB_OFFLINE=1 DS_SKIP_CUDA_CHECK=1
APP_DIRECTORY=experiments/spo_tree_smoke \
EXTRA_CONFIGS=configs/episode_generators/branch_factor_444.jsonnet \
  bash scripts/launch_server2_spo.sh spo_tree MATH
```
Watch for: vLLM server comes up, iter-0 episodes generate, a `ckpt--iter_*` dir
appears, no NCCL/CUDA errors. Ctrl-C after iter 1. (`branch_factor_444` reduces the
tree to 4-4-4 **and** lowers `num_episodes_per_iteration` from 512 → 384 — both make
the smoke cheaper; the real run uses the config's built-in 6-6-6 / 512 episodes.)

---

## 6. Full training launch

```bash
mkdir -p /lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42
cd ~/projects/SPO
sbatch scripts/sbatch_spo_tree_qwen1b_MATH.sh
# optional: SEED=43  SPO_ENV=spo  WANDB_PROJECT=spo-math  WANDB_MODE=offline  sbatch ...
```
Chain: `sbatch_spo_tree_qwen1b_MATH.sh` → `scripts/launch_server2_spo.sh spo_tree MATH`
→ `deepspeed --include localhost:0 src/treetune/main.py --configs
"configs/polIter_qwen1b_spo_tree_MATH.jsonnet,configs/gpus/gpu_0.jsonnet" run_iteration_loop`.

Checkpoints land in `…/spo_tree_qwen1b_MATH_seed42/` as `ckpt--iter_NNNN--*`
(HF-format dump under `hf_pretrained/`). `save_steps=5` writes a checkpoint every
5 iterations; `checkpoint_keep_steps=10` **permanently retains every checkpoint whose
iteration is a multiple of 10** (others are pruned by the trainer's `clean_checkpoints`
logic: `ppo_trainer.py:2560-2565`). So iters 10, 20, 30, …, 400, …, 1000 survive —
this means **iter 400 IS retained automatically** for §7's eval; only iters 5, 15,
25, …, 995 get cleaned. (Disk is the only concern, not retention.)

**(Optional) GRPO baseline for the comparison column** — same wrapper:
```bash
APP_DIRECTORY=/lustre-storage/checkpoints/zengh/spo/grpo_qwen1b_MATH_seed42 \
  bash scripts/launch_server2_spo.sh grpo MATH
```

---

## 7. Evaluation on MATH500 (`lighteval` + vLLM) — the headline numbers

Run in the **`spo-eval`** env (§3b), pointing at an HF-format checkpoint from §6
(or first reproduce against the authors' `gyr66/spo-tree-666-qwen1.5B-math` to
sanity-check the harness). Template: `scripts/evaluate_long_cot.sh`.

> ⚠️ **`custom|math_500` requires `--custom-tasks` from open-r1.** Upstream
> `scripts/evaluate_long_cot.sh` calls `custom|math_500|0|0` **without
> `--custom-tasks`** — it only works from inside an open-r1 checkout that defines
> that task. The SPO repo does **not** ship `src/open_r1/evaluate.py`. Clone
> open-r1 first:
> ```bash
> git clone https://github.com/huggingface/open-r1 /tmp/open-r1
> ```
> (Newer `lighteval` may also expose `lighteval|math_500|0|0` built-in — try that
> if the custom-tasks path drifts. Match the open-r1 / lighteval versions the
> authors used if numbers are off.)

```bash
conda activate spo-eval
cd ~/projects/SPO
MODEL=/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42/ckpt--iter_0400--*/hf_pretrained
NUM_GPUS=1
OUTPUT_DIR=data/evals/spo_tree_seed42

# pick ONE context size per eval (these reproduce the three table rows):
CTX=2048    # 2K -> target 0.736   (use 4096 for 0.828, 32768 for 0.848)
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$CTX,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:$CTX,temperature:0.6,top_p:0.95}"

lighteval vllm "$MODEL_ARGS" "custom|math_500|0|0" \
    --custom-tasks /tmp/open-r1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"
```

**Reference outputs** to diff against live in the repo: `results/SPO-tree-666/*.json`
(`extractive_match` 0.736 @2K, 0.828 @4K, 0.848 @32K — verified). Compare your
`data/evals/.../results_*.json` `results["custom|math_500|0"]["extractive_match"]`
to the table.

---

## 8. Paths & wandb

| What | Path |
|---|---|
| Output dir | `/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42/` |
| Checkpoints | `…/ckpt--iter_NNNN--*/hf_pretrained/` |
| Slurm log | `…/sbatch_slurm-<jobid>.log` |
| Run log (wrapper tee) | `…/run_<UTC>.log` |
| Eval JSONs | `~/projects/SPO/data/evals/…/results_*.json` |
| Reference eval JSONs | `~/projects/SPO/results/SPO-tree-666/*.json` |

**wandb:** entity **`suesie`** (from the env's login), project `spo-math` (override
with `WANDB_PROJECT`). As in the VinePPO runbook, treetune may set the project at
`wandb.init` from config/defaults — confirm the project after the run starts;
recover dropped sync with `wandb sync <wandb/run-… dir>`.

---

## 9. Monitoring

```bash
JOB=<jobid>
LOG=/lustre-storage/checkpoints/zengh/spo/spo_tree_qwen1b_MATH_seed42/sbatch_slurm-$JOB.log
squeue -j $JOB -O "jobid,state,timeused,nodelist"
tail -n 40 "$LOG"
grep -E "Iteration [0-9]+|Saving|ckpt--iter|eval|reward" "$LOG" | tail
sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS -X
```
Success = the iteration loop advances, `ckpt--iter_*` dirs appear, and the
in-training MATH-test accuracy (and later the §7 lighteval number) climb toward
the targets.

---

## 10. Known pitfalls (carried over + SPO-specific)

1. **Two envs.** Never `pip install -r requirements.txt` (with the trailing
   `lighteval`) into the training env — it upgrades transformers/vllm and breaks
   the pinned stack. Train in `spo`/`vineppo`; eval in `spo-eval`. (§3)
2. **`custom|math_500` needs `--custom-tasks`** from open-r1 (§7).
3. **NCCL 2.18.1 overlay** → first-collective crash; force-reinstall the wheel (§3a).
4. **DeepSpeed CUDA check** — set `DS_SKIP_CUDA_CHECK=1` (nvcc 12.8 vs torch cu121);
   already in the sbatch.
5. **Single GPU, not 8.** Don't add `--num_gpus`; the wrapper pins `localhost:0`.
   vLLM + trainer share the GPU via `wait_until_memory_release`. (§2)
6. **Dataset host** (Aliyun OSS) may be slow/blocked on compute nodes — fetch on a
   login node first (§4a).
7. **Context: shipped config ≠ published checkpoint (verified).** The published
   model is `Qwen1.5B-MATH-C4096-400` = trained at **context 4096**, iter 400. The
   shipped `polIter_qwen1b_spo_tree_MATH.jsonnet` has `model_context_size=2048`
   (a hard ceiling on total prompt+response), so even with `max_tokens=4096`
   generation is bounded by `min(4096, 2048 − prompt_tokens)`. The repo ships **no
   2K→4K schedule config** (the only `model_context_size:4096` configs are for the
   *point24* task, not MATH). So as-shipped you get a 2K-context training run; to
   match the **0.828@4K** headline you must locally set `model_context_size:4096`
   (keep `max_tokens:4096`), mirroring the README's "2K→4K". The 2K/4K/32K **table
   rows are eval contexts** of one model (set via `max_model_length`/`max_new_tokens`
   in §7).
8. **iter-400 checkpoint is retained automatically.** Headline SPO-tree =
   **iter 400** (GRPO ≈ iter 680 per the published checkpoint naming), both <
   the config's 1000. `checkpoint_keep_steps=10` means **"permanently retain every
   checkpoint whose iteration is a multiple of 10"** (NOT "keep the last 10 saves" —
   common misreading; see `ppo_trainer.py:2560-2565`). Since 400 % 10 == 0, iter 400
   survives the cleanup; same for 410, 420, … so you can eval multiple nearby
   checkpoints and report the best.

---

## 11. Step checklist

- [ ] §3 env(s) ready (`vineppo`/`spo` for train, `spo-eval` for eval); NCCL 2.18.1 verified
- [x] §4 `data/` staged on lustre + symlinked; `DeepSeek-R1-Distill-Qwen-1.5B` cached on lustre ✅
- [ ] §5 smoke test green (iter-0 episodes + a `ckpt--iter_*` dir)
- [ ] §6 full SPO-tree run submitted; checkpoints appearing
- [ ] (opt) GRPO baseline submitted
- [ ] §7 MATH500 eval @2K/4K/32K; compare to 0.736/0.828/0.848
- [ ] record final numbers + wandb links here
