# Upstream note: transformers 5 checkpoints load with the wrong RoPE in transformers 4.x

Findings from Oct 23, 2026, kept for a possible transformers PR. Not in scope for the GRPO study.

## Problem
- `save_pretrained` in transformers 5.x writes RoPE as `"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}`
  and drops the top-level `rope_theta`.
- transformers 4.x (checked: 4.50.0, 4.56.2, 4.57.1) only reads the top-level `rope_theta`. When it's absent it falls back
  to the class default (10000 for Qwen2), with no warning. The rotary frequencies are wrong and so are the logits.
- `rope_parameters` was introduced in 5.0.0 (`models/qwen2/configuration_qwen2.py:135` in 5.0.0); no 4.x release reads it.

## Evidence (GG model, Qwen2 1.5B, probe checkpoint saved by TRL 1.16 / transformers 5.17)
| Load | `rope_theta` | `inv_freq[1]` | fp32 logits vs 5.17 (1802 positions) |
|---|---|---|---|
| 4.50 on the 5.17-saved checkpoint | 10000 | 0.86596 | max abs diff 16.5, argmax agrees 1626/1802 |
| 4.50 with top-level `rope_theta` (and `torch_dtype`) added | 1e6 | 0.80584 | bit-identical (diff 0.0, 1802/1802) |
| 5.17 on either config | 1e6 | 0.80584 | reference |

In 5.17 an extra top-level `rope_theta` is harmless: `rope_parameters` wins (`modeling_rope_utils.py:748-754`,
`setdefault`). Scripts and raw outputs: `ggrpo/ops/totoro/compat_test/` (`probe.py`, `compare.py`, `fix_research/cfg.py`,
`r*.json`, `b*.json`).

## Candidate PR (transformers 5.x)
Back-compat on save: also write `rope_theta` (and `torch_dtype`) at the top level of `config.json` when `rope_parameters`
holds a plain default RoPE, so 4.x readers get the right base. Alternatively, warn on load in 4.x when a config has an
unknown `rope_parameters` key, but that needs a 4.x patch release.

## Related reports (same family, none is this exact case in transformers)
- vLLM PR #44158 (v5 checkpoints loaded with v4): https://github.com/vllm-project/vllm/pull/44158
- vLLM issue #58675 (silent fallback to 10000): https://github.com/vllm-project/vllm/issues/58675
- SGLang PR #41327: https://github.com/sgl-project/sglang/pull/41327
- AMD Primus #1193: https://github.com/AMD-AGI/Primus/issues/1193
- NVIDIA Model-Optimizer PR #2221: https://github.com/NVIDIA/Model-Optimizer/pull/2221
- transformers PR #49515 (rope_scaling setter drops rope_theta): https://github.com/huggingface/transformers/pull/49515

## What this project does about it
A load-time assert on `rotary_emb.inv_freq[1]` in `src/ggrpo/generate.py` and the cluster's `gen_bringup.py`, and a
"view" dir (base config and tokenizer files plus a symlink to the trained weights) for 4.50 evals. See
`ops/lanes/relay_cluster.sh`.
