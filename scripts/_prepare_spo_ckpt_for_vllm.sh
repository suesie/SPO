#!/usr/bin/env bash
# Symlink tokenizer files from a base-model HF snapshot into a SPO checkpoint's
# hf_pretrained/ dir, so vLLM can load it standalone.
#
# SPO writes pytorch_model.bin + config.json + generation_config.json to
# <ckpt>/hf_pretrained/ but NOT tokenizer files (treetune keeps the tokenizer
# separate). vLLM crashes without them. This helper makes the dir
# self-contained by symlinking tokenizer.json / tokenizer_config.json /
# special_tokens_map.json / *.jinja from the base-model snapshot.
#
# Idempotent: skips files that are already present (real file or symlink).
#
# Usage:
#   bash _prepare_spo_ckpt_for_vllm.sh <hf_pretrained_dir> <base_model_snapshot_dir>
#
# Example:
#   bash _prepare_spo_ckpt_for_vllm.sh \
#       /lustre-storage/checkpoints/zengh/spo/spo_tree_deepseek7b_MATH_seed42/.../ckpt--iter_0025*/hf_pretrained \
#       /lustre-storage/datasets/zengh/huggingface/hub/models--realtreetune--deepseekmath-7b-sft-MATH-v2/snapshots/8b387c255b3bfaaaef2e650d56fecfde1c56ea96

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <hf_pretrained_dir> <base_model_snapshot_dir>" >&2
    exit 64
fi

HF_DIR="$1"
BASE_DIR="$2"

if [ ! -d "$HF_DIR" ]; then
    echo "[prep] FATAL: not a dir: $HF_DIR" >&2
    exit 3
fi
if [ ! -d "$BASE_DIR" ]; then
    echo "[prep] FATAL: base snapshot not a dir: $BASE_DIR" >&2
    exit 3
fi

# Required: config.json + at least one weight file already in HF_DIR.
if [ ! -f "$HF_DIR/config.json" ]; then
    echo "[prep] FATAL: $HF_DIR missing config.json (not a HF dir?)" >&2
    exit 3
fi
HAS_WEIGHTS=0
shopt -s nullglob
for w in "$HF_DIR"/*.safetensors "$HF_DIR"/pytorch_model.bin "$HF_DIR"/pytorch_model.bin.index.json; do
    [ -e "$w" ] && HAS_WEIGHTS=1 && break
done
shopt -u nullglob
if [ "$HAS_WEIGHTS" -eq 0 ]; then
    echo "[prep] FATAL: $HF_DIR has no model weights" >&2
    exit 3
fi

# Symlink tokenizer files (only those that exist in BASE_DIR; only those NOT
# already in HF_DIR).
LINKED=0
for fname in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json; do
    if [ -f "$BASE_DIR/$fname" ] && [ ! -e "$HF_DIR/$fname" ]; then
        ln -s "$BASE_DIR/$fname" "$HF_DIR/$fname"
        LINKED=$((LINKED + 1))
    fi
done
# chat templates (Jinja)
shopt -s nullglob
for jf in "$BASE_DIR"/*.jinja; do
    target="$HF_DIR/$(basename "$jf")"
    if [ ! -e "$target" ]; then
        ln -s "$jf" "$target"
        LINKED=$((LINKED + 1))
    fi
done
shopt -u nullglob

# Verify the result satisfies vLLM's tokenizer requirement.
if ! ls "$HF_DIR"/tokenizer*.json >/dev/null 2>&1; then
    echo "[prep] FATAL: $HF_DIR still missing tokenizer*.json after linking from $BASE_DIR" >&2
    echo "[prep]   BASE_DIR listing:" >&2
    ls -la "$BASE_DIR" >&2
    exit 3
fi

echo "[prep] $HF_DIR ready (linked $LINKED files from base)"
