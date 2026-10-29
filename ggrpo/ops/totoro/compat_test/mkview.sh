#!/usr/bin/env bash
# mkview.sh CHECKPOINT_DIR VIEW_DIR: base snapshot config/generation/tokenizer files + symlink to the checkpoint weights.
set -euo pipefail
ck=$(realpath "$1"); view=$2
base=$HOME/.cache/huggingface/hub/models--ahmedheakl--gg-armv8-O2/snapshots/0444a378f384300187c7f43914fc986a81ff051a
test -f "$ck/model.safetensors"
mkdir -p "$view"
cp -L "$base"/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json} "$view"/
ln -sfn "$ck/model.safetensors" "$view/model.safetensors"
