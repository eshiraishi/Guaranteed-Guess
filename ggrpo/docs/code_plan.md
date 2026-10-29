# ggrpo design notes

## Principles
- Use library entry points over custom code. `trl grpo --config grpo.yaml` is the trainer, so there is no train.py.
  It trains all weights, as GG's own SFT did (model card tag `full`). QLoRA was dropped (Oct 22): TRL merges the adapter
  into the 4-bit weights for vLLM, which rounds the update away, and generating without vLLM was 6-9x slower.
- Our settings and paths live in `config.toml` (pydantic-settings, with CLI overrides). Library configs keep their own
  format (TRL: YAML).
- Keep the paper's exact behavior unless the paper or the data says otherwise. Values with no source are `TODO` and stop
  the run until someone decides them.
- Splits and reporting (decided Oct 22): MBPP-C is 90% train and 10% dev (`config.toml` `dev_fraction`, `split_seed`;
  identical x86 stays on one side). HumanEval-C and BringUp are held out. The reported model is the final checkpoint.
  Dev is scored offline on every saved checkpoint with the HumanEval decoding, as a learning curve and for later tuning,
  never to pick the reported checkpoint. The 952-prompt run that ran out of memory at step 19 is the pilot.
- Metrics (Enzo, Oct 23): report the paper's metric, pass@1 with beam 8, for reference. Sampled pass@1 under the
  training distribution (`ops/lanes/sample_eval.py`) was run on dev only as a diagnostic: GRPO raised it (45.1% to 48.3%,
  sign test p = 0.016) while beam 8 stayed flat (56 to 54 of 94). It is not a reported metric. Enzo (Oct 23, later):
  it is also scored on HumanEval-C (base and final checkpoint) as a diagnostic, always after report-bound work.
- W&B run names (Enzo, Oct 23): the name is the recipe. `GRPO` is `grpo.yaml` as is, for every seed (the seed is in the
  run config); a variant is `GRPO (key: value)` with TRL's keys. The baseline arms are `GRPO (DeepSeekMath)`,
  `Dr. GRPO` and `SFT`. The base model is its own run, `GG`, at step 0. Run ids are fixed by the launchers
  (`grpo-seedN`, `dsm-seedN`, `drgrpo-seedN`, `sft-seedN`), so a relaunch continues its run. Runs from before the
  switch to W&B (seeds 0 and 1, trained as `baseline, seed=N`) were copied from Trackio by
  `ops/migrate/trackio_to_wandb.py`; the Trackio DB backups are kept.
- One extraction point: the reward server applies the `config.toml` regex to raw completions, for training and for eval.
- Tracking is W&B (Oct 23), chosen over Trackio because TRL's W&B callback needs no workarounds: repeated names with
  unique ids, every argument as run config, `train/global_step` as the x-axis. Since Oct 24 (Enzo), all data goes to
  W&B and every run, totoro included, logs straight to it; Trackio is no longer used or installed. `WANDB_API_KEY` is
  in the gitignored `.env`, read by `Settings` and by the launcher. Evals are scored offline and logged into the
  training run at their checkpoint step by `python -m ggrpo.track` (files listed in `config.toml` `[track]`);
  `python -m ggrpo.report` exports every eval row to one CSV, the source for paper tables and figures. Event logs are
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
3. **Training subset:** `data.train_ids` stays unset until the contradicting samples are sorted out.
