# ggrpo design notes

## Principles
- Use library entry points over custom code. `trl grpo --config grpo.yaml` is the trainer, so there is no train.py.
  LoRA and QLoRA differ only in `--load_in_4bit true --bnb_4bit_quant_type nf4`.
- Our settings and paths live in `config.toml` (pydantic-settings, with CLI overrides). Library configs keep their own
  format (TRL: YAML).
- Keep the paper's exact behavior unless the paper or the data says otherwise. Values with no source are `TODO` and stop
  the run until someone decides them.
- One extraction point: the reward server applies the `config.toml` regex to raw completions, for training and for eval.
- Tracking is Trackio in local mode: SQLite under `runs/trackio`, no server, no docker, no Hub Space. Event logs are
  structlog JSON lines.

## Facts the code relies on (checked by `ggrpo.data` and the tests)
- HumanEval-C eval = GG's bench (`gg-bench-armv8-O2` @ 32a82dd9) joined on `file` with the authors' own tests
  (`asmbench_source` @ 89bca970). `eval/problem{n}/code.c` is `humaneval/{n-1}`. References pass 164/164.
- Every task has exactly one C test driver and no extra compiler flags (MBPP-C `cflags == []`, only system headers).
- The checkpoint's `generation_config.json` sets `do_sample=true, temperature=0.7, top_k=20, top_p=0.8,
  repetition_penalty=1.1`. In transformers 5.x, a field left as `None` in a passed `GenerationConfig` is taken from this
  file. So every decoding value that matters is explicit in `config.toml` and `grpo.yaml`.
- TRL 1.16 supports vLLM ≤ 0.31.0. vLLM 0.31 requires fastapi < 0.137, so both extras share fastapi 0.136.3.
  The lock resolves transformers to 5.17.0 and torch to 2.13.0 (CUDA 13 wheels).

## Reproduction and package evidence (running on ponyo, one RTX 4070 Ti SUPER)
- Paper environment = the model card: transformers 4.50.0, torch 2.6.0+cu124, datasets 3.4.1, tokenizers 0.21.0
  (+ flash-attn 2.7.4.post1, accelerate 1.5.2, not documented). It can't run on Blackwell (sm_120 needs CUDA ≥ 12.8).
- Run A: released `eval/eval_huamaneval.py` after `set_seed(42)` (beam sampling inherited). Run B: same with
  `do_sample=False` (peer session's `greedy_beam`, same env).
- Modern stack (train extra) vs paper env: log-probs of the beam-8, 16k-cap completions (paper+FA2, paper+sdpa, modern+sdpa)
  and a full greedy-beam generation compared output-by-output with Run B.

## Open questions
1. **Paper decoding (for the authors).** `eval/eval_huamaneval.py` passes only `num_beams=8`. It therefore inherited
   `do_sample=true` (beam sampling with temperature 0.7, top-k 20, top-p 0.8) and `repetition_penalty=1.1`, but the paper
   says sampling was disabled. Which one produced 45.12%? `config.toml [generate.beam_search]` currently follows the
   paper text and the beam-8, 16k-cap run: no sampling, with `repetition_penalty` 1.1.
2. **Attention kernel.** The paper's eval used `flash_attention_2`; we use `sdpa`. Beam outputs may differ slightly.
   Check whether FA2 is available on sm_120.
3. **The `TODO` values in `grpo.yaml`:** loss type (grpo vs dapo), LoRA r, alpha and dropout, learning rate, steps,
   batch sizes, and `max_completion_length`.
4. **Training subset:** `data.train_ids` stays unset until the contradicting samples are sorted out.
