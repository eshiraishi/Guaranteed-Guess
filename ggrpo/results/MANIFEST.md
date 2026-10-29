# Experiment file manifest

Copied 2026-10-23 01:10-01:30 Mac local time (UTC+4). Copy only: no source was modified, moved or deleted. Every copy was checked against its source by sha256.
Scripts and configs that ran the experiments are under `ggrpo/ops/<origin>/`; outputs are under `ggrpo/results/<topic>/`.
Original filenames are kept. Directories under `ops/` mirror the source path relative to the host's home, except that a
source directory named `runs/` is stored as `ggrpo_runs/`, because `ggrpo/.gitignore` ignores every `runs/` directory.

`ggrpo/ops/lanes/` is not part of this manifest: another session created it during this copy (2026-10-23 01:09-01:23 local)
as the master copies of the deployed lane scripts. Its `eval_gpu1.sh` and `eval_ponyo.sh` are byte-identical to the host
copies in `ops/totoro/ggrpo_runs/` and `ops/ponyo/ggrpo_runs/`; its `job.sh` and `gen_bringup.py` are newer than the
cluster copies in `ops/cluster/cs8001-bringup/`, which record what was on the cluster at copy time.

## Source labels

| Label | Meaning |
|---|---|
| `mac-scratch:<id>/<path>` | `/private/tmp/claude-501/-Users-eshiraishi-git-Guaranteed-Guess-feature-env/<session>/scratchpad/<path>` (sessions: 643399c8-a9fa-407f-8537-f428823102e9, 31015bbb-b67c-4890-812b-48505da33507, 33ca3c48-fab5-43c5-ab02-a7a58c388e55, 9da06990-0722-4726-8bf3-51c0b2111421, d37ed3f6-dbb4-49d0-bb9f-f8fbe48eadd9, c34c21bc-d409-4960-8fcf-b8813ffa399a) |
| `mac-scratch-main:d3bd2ab3/<path>` | `/private/tmp/claude-501/-Users-eshiraishi-git-Guaranteed-Guess/d3bd2ab3-c157-4524-9b54-5f107742f9c6/scratchpad/<path>` |
| `mac:~/...` | This Mac (main checkout `~/git/Guaranteed-Guess`, branch qlora-grpo-study; this worktree) |
| `vm:~/...` | Tart VM gg-ventura, `admin@192.168.64.2` |
| `totoro:~/...` | `totoro.mbzu.ae`, `/home/eshiraishi` |
| `ponyo:~/...` | `ponyo`, `/home/eshiraishi` |
| `cluster:~/...` | `student-lab` login node, home of `enzo.shiraishi` |

Markers in descriptions: **LIVE** = the source was still being written when copied (this is a snapshot);
**PARTIAL** = the run had not finished. Remote clocks were about 2026-10-22 21:10-21:20 UTC during the copy.

## Git

The column *Ignored by* shows the rule from `git check-ignore -v`. 36 of 262 files are ignored, all by the
repository root rule `.gitignore:62:*.log`. Nothing has been staged or committed; force-add (`git add -f`) any ignored
file you want to keep in git.

## Totals per origin

| Origin | Files | MB |
|---|---:|---:|
| Mac scratchpads (/private/tmp) | 68 | 2.88 |
| Mac checkouts (main checkout, this worktree's gitignored runs/) | 11 | 2.29 |
| VM gg-ventura | 27 | 9.98 |
| totoro | 105 | 86.48 |
| ponyo | 37 | 13.52 |
| cluster (student-lab) | 14 | 0.09 |
| **Total** | **262** | **115.23** |

Remote transfer used `rsync -az --bwlimit=1500`. With compression about 8.2 MB crossed the network for 89.4 MB of remote
files (totoro 75.8 MB including the 24.9 MB Trackio snapshot, ponyo 13.5 MB, cluster 0.09 MB). The VM (about 45 MB) was
staged over the local VM link. Three files labelled totoro or ponyo (`training/pilot/train952.jsonl`, `training/stress/train858.jsonl`,
`ops/ponyo/cs8001-step0/gen_v2.py`) were copied from byte-identical local files after checking their sha256 on the host.

## Files

### `ops/cluster/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `cluster:~/cs8001-bringup/gen_bringup.py` | `ops/cluster/cs8001-bringup/gen_bringup.py` | 3571 | `1a404cb8b57f5e49428e6e64b1636048911d1388065469e10d88682dee38c3be` | - | Cluster variant of gen_bringup.py (adds --gpu guard and --files shard selection). Identical copies (same sha256): `mac-scratch:643399c8/cluster/gen_bringup.py`. |
| `cluster:~/cs8001-bringup/job.sh` | `ops/cluster/cs8001-bringup/job.sh` | 623 | `3070bba563d324b6bd8b1b7f5e07da0218a8f286d7c9feb0c2ddf066ae657b70` | - | Slurm job: one gen_bringup.py run on an RTX 5000 Ada (partition ws-ia, 48G, 2h). Identical copies (same sha256): `mac-scratch:643399c8/cluster/job.sh`. |
| `cluster:~/cs8001-bringup/shard_a.txt` | `ops/cluster/cs8001-bringup/shard_a.txt` | 359 | `1f256c448e5ae2bd5a963c235a65edffd2b7dece0b4ee8ac58c091f8467bc64e` | - | BringUp file list for beam8 shard a. Identical copies (same sha256): `mac-scratch:643399c8/cluster/shard_a.txt`. |
| `cluster:~/cs8001-bringup/shard_b.txt` | `ops/cluster/cs8001-bringup/shard_b.txt` | 325 | `fc8e9c545daa784e73972afd6be41c2b9a2594bd96a0d3405c053e39eb9ace39` | - | BringUp file list for beam8 shard b. Identical copies (same sha256): `mac-scratch:643399c8/cluster/shard_b.txt`. |
| `cluster:~/cs8001-bringup/greedy_a.txt` | `ops/cluster/cs8001-bringup/greedy_a.txt` | 404 | `47931e5318b6c470301e0cec46154768b7fd33104e1605b046a5d3078f8ae11e` | - | BringUp file list for a greedy shard a (created 2026-10-23 01:08 local, likely for a job not yet run). |
| `cluster:~/cs8001-bringup/greedy_b.txt` | `ops/cluster/cs8001-bringup/greedy_b.txt` | 388 | `8e8675860ebb277b19161c1f159beb647fb369f3e8edbde4837ebcd4f34bd79f` | - | BringUp file list for a greedy shard b (created 2026-10-23 01:08 local, likely for a job not yet run). |
| `cluster:~/cs8001-bringup/requirements.txt` | `ops/cluster/cs8001-bringup/requirements.txt` | 1081 | `4bbcb0ef1a91981e0b682032073a0a70a1cc076b19a6aa87eb12c6d239a9310e` | - | pip freeze of the cluster BringUp venv (flash-attn 2.8.3, torch cu12). Identical copies (same sha256): `mac-scratch:643399c8/cluster/requirements.txt`; `mac-scratch:643399c8/bringup_freeze.txt`. |
| `cluster:~/cs8001-bringup/DATA_SOURCE.txt` | `ops/cluster/cs8001-bringup/DATA_SOURCE.txt` | 70 | `2e0037c979f1aba70251dcbee636920230210f8a2084f6142f5f1a2a5aaf0f6b` | - | Records the HF snapshot of gg-bench-bringup-O2 used (ae6623a3...), same as totoro's. Identical copies (same sha256): `mac-scratch:643399c8/cluster/DATA_SOURCE.txt`. |

### `ops/mac-scratch/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `mac-scratch:643399c8/bringup_to_preds.py` | `ops/mac-scratch/643399c8/bringup_to_preds.py` | 754 | `4eac441fd648947279ce5eb535e9e85954a589ecdd3e80a9bd55f00b91602753` | - | Converts gen_bringup.py JSONL into save_preds.py-style {files, pred} JSON for the BringUp scorer (rows with an error get an empty pred). |
| `mac-scratch:643399c8/compare_gpu.py` | `ops/mac-scratch/643399c8/compare_gpu.py` | 986 | `b8ef4f44d8c00908f843bf91c439177dc8e8aa8a00cfdc0d8f5a0f689b7426b3` | - | Per-row comparison of a cluster RTX 5000 Ada BringUp beam8 run against the Blackwell run (seconds, ratio, peak memory, identical output). |
| `mac-scratch:643399c8/tf450/mode.py` | `ops/mac-scratch/643399c8/tf450/mode.py` | 1564 | `8512210eb92c0598774065f439017806ba4bacc0cdd6e648f1030e44e4aeae34` | - | Checks what eval_huamaneval.py's generate() call resolves to under transformers 4.50.0 (tiny random Qwen2 with the real generation_config): beam search or beam sampling. |
| `mac-scratch:643399c8/xcheck/totoro_generate.py` | `ops/mac-scratch/643399c8/xcheck/totoro_generate.py` | 2246 | `2dfda96dfd7169488a56e8f247733d6bacef5667d30a06b55ccb22b4b5b1ce00` | - | ggrpo.generate as deployed on totoro for the cross-machine generation check (no PEFT adapter, tolerates an empty resume file). |
| `mac-scratch:643399c8/xcheck/ponyo_config.toml` | `ops/mac-scratch/643399c8/xcheck/ponyo_config.toml` | 3769 | `2719d186c01a038ef3025f3871e94aadff1cc08a0f8e56a93c4781688b7511aa` | - | config.toml as deployed on ponyo for the cross-machine check (older data keys: raw ventura MBPP parquet, step0_predictions). |
| `mac-scratch:31015bbb/contam_probe.py` | `ops/mac-scratch/31015bbb/contam_probe.py` | 2279 | `7cfc44ae5f313aca95fb97d0c80e6192e33568c2b53295fcff281d4d21e4fd4c` | - | Feasibility probe: exact and 10-instruction-window overlap between benchmark x86 and one GG SFT shard. |
| `mac-scratch:31015bbb/contam_probe_norm.py` | `ops/mac-scratch/31015bbb/contam_probe_norm.py` | 653 | `298d31318b7cf871cdc04d83c2680ae249311aa563170ab5a1ac5db4eeee3492` | - | Normalized-operand variant of contam_probe.py (same overlap probe). |
| `mac-scratch:31015bbb/overlap_probe.py` | `ops/mac-scratch/31015bbb/overlap_probe.py` | 880 | `14ee591915f55becd04f65db1ab96917ad5cbbcb605f92062bf2584a1d69ff72` | - | Exact normalized x86 overlap inside valid MBPP-C and between MBPP-C and HumanEval-C. |
| `mac-scratch:33ca3c48/check_drivers.py` | `ops/mac-scratch/33ca3c48/check_drivers.py` | 1198 | `dda3c64ca42876e1abedbf34ef913bbdb89af186a287b360f974cf00dfaeac88` | - | Lists helper definitions shared between MBPP-C func and test sources (driver-extraction check for ggrpo.build). |
| `mac-scratch:33ca3c48/grpo_filled.yaml` | `ops/mac-scratch/33ca3c48/grpo_filled.yaml` | 1626 | `14a3f6b0936fbc73c7c180071c4d75ad491b3095212dc1013360d72de1044177` | - | Earlier QLoRA-era grpo.yaml draft with the TODOs filled for a 2-step test (lora_r 16, dapo, LR 1e-6, completion length 64). |
| `mac-scratch:9da06990/scripts/cf_count.py` | `ops/mac-scratch/9da06990/scripts/cf_count.py` | 467 | `f295289cc19079b89eef1d1cd6b6cd8c3f0daad7bf381d59116e5dfb64006882` | - | Dataset-survey/decontamination session script (LIVE session snapshot): % ",".join("'https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/main/data/train-%05d-of-00036.parquet'" % i for i in range(36)). |
| `mac-scratch:9da06990/scripts/cf_count2.py` | `ops/mac-scratch/9da06990/scripts/cf_count2.py` | 1081 | `2b09bedfef1e8e2eb742c5f2a49c20c6fe7096ef127f312c3d74abf4d6f10b5e` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/compile_test.py` | `ops/mac-scratch/9da06990/scripts/compile_test.py` | 3840 | `54099be56f05286fddfa067f479d8d40224839d98e77eac72451cbcfa37d9098` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Compile-only portability probe of ExeBench rows on macOS clang. No execution of dataset code.. |
| `mac-scratch:9da06990/scripts/compile_test2.py` | `ops/mac-scratch/9da06990/scripts/compile_test2.py` | 4026 | `4604cd8ebdcba62561f59f8c1dd7f68d063e0551ea5daaea4481e4a03a71c9d5` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Compile-only portability probe of ExeBench rows on macOS clang. No execution of dataset code.. |
| `mac-scratch:9da06990/scripts/compile_test3.py` | `ops/mac-scratch/9da06990/scripts/compile_test3.py` | 4155 | `f658aec4ef311e4a0b94f7f0ff365368fd4b05ba98d8a11da47e52feb7fa96fb` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Compile-only portability probe of ExeBench rows on macOS clang. No execution of dataset code.. |
| `mac-scratch:9da06990/scripts/dedup.py` | `ops/mac-scratch/9da06990/scripts/dedup.py` | 1563 | `154baa95bf01449b6c1dc9e3e13a1ff04507510e647ac4fb07de33034c799c69` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/embed.py` | `ops/mac-scratch/9da06990/scripts/embed.py` | 1623 | `22ce77ec99ff4e522fa0f0d57cb7c5f333a986f19b52ef30138962e48797fffc` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Embedding similarity of MBPP (train/dev) vs HumanEval problem statements; top-3 MBPP per HumanEval.. |
| `mac-scratch:9da06990/scripts/mbppplus.py` | `ops/mac-scratch/9da06990/scripts/mbppplus.py` | 1236 | `fa82e244049d1274c352acd7f1c38da0d88a12c76a5d49ce90b8d827b1bbeab2` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/neardup.py` | `ops/mac-scratch/9da06990/scripts/neardup.py` | 7916 | `e5f45946275863e0f14a11b536301075f14bc39f1464882b2596796b87ee8b36` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Near-duplicate probe of HumanEval-C, BringUp and MBPP-C against GG's SFT data.. |
| `mac-scratch:9da06990/scripts/neardup_lib.py` | `ops/mac-scratch/9da06990/scripts/neardup_lib.py` | 2956 | `8be7366f1547d2349d02efd270da9793a91f0d233d38b0ff76276b36987054f9` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/overlap.py` | `ops/mac-scratch/9da06990/scripts/overlap.py` | 599 | `0a386433f5eaa1677617197e869b505c70b7b37ce5dbb23eec4d13a310acf4d5` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/peek.py` | `ops/mac-scratch/9da06990/scripts/peek.py` | 993 | `a386e5c0f7719f34424c3383cdb71b837e1564b16c21e453196544cdf60113b8` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/peek2.py` | `ops/mac-scratch/9da06990/scripts/peek2.py` | 923 | `e8eb8d1b115ed8369b46265c793b185bbfbb5dda7e0f3a0150bea544d1b696ed` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/runs.py` | `ops/mac-scratch/9da06990/scripts/runs.py` | 2165 | `871aa6cbb755efa98c3c8d9225d18119153b5bdd5cb0f72c400ce7e86d52de59` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Longest contiguous run of shared canonical 13-grams between each BringUp/HumanEval probe and its top-3 Stack files.. |
| `mac-scratch:9da06990/scripts/runshow.py` | `ops/mac-scratch/9da06990/scripts/runshow.py` | 1841 | `accc1cb73f0e3e78312087633515a587ccd4acfcf014f34ebee6f1b55e8c2518` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Print the longest shared canonical-13-gram run between probe and file. usage: runshow.py SCRATCH GGRPO probe [probe...]. |
| `mac-scratch:9da06990/scripts/sameprob.py` | `ops/mac-scratch/9da06990/scripts/sameprob.py` | 4094 | `990f883c66790aed3736975010dffd79ad73e9592c93d62c989ec7498e0e5c9a` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Candidate same-problem pairs between MBPP-C (train/dev) and HumanEval-C (eval), for manual review.. |
| `mac-scratch:9da06990/scripts/show.py` | `ops/mac-scratch/9da06990/scripts/show.py` | 1798 | `28e81c087cef4a3dcaa751d266a289b7c0f72293e0a724e1b45faecee6c5bc76` | - | Dataset-survey/decontamination session script (LIVE session snapshot): Show the region of a Stack file that shares canonical 13-grams with a probe. usage: show.py SCRATCH GGRPO probe file [probe file ...]. |
| `mac-scratch:9da06990/scripts/stats.py` | `ops/mac-scratch/9da06990/scripts/stats.py` | 1807 | `88ad0e58b5f3ea6943bde9afa25c02e4554dbcabd8eca9a11779d1495f1b34a8` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/synth.py` | `ops/mac-scratch/9da06990/scripts/synth.py` | 1705 | `4790d7f41242074f1616214120975815296d4cb369d7fbe2361beb0bdaef8682` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/tc_ccompat.py` | `ops/mac-scratch/9da06990/scripts/tc_ccompat.py` | 1223 | `faf31ac100c6ca861754d91417e0d88628f8da3b049bc5d398046cdbd24de7db` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/scripts/triv.py` | `ops/mac-scratch/9da06990/scripts/triv.py` | 999 | `7b853756a8692a6680a5367040adcfe1ea04c9814dc639f84ce05261c918d0bf` | - | Dataset-survey/decontamination session script (LIVE session snapshot): ad-hoc analysis helper (no docstring). |
| `mac-scratch:9da06990/dl.cfg` | `ops/mac-scratch/9da06990/dl.cfg` | 4092 | `1537543b106b38799b77966d1264219ad6c4cbcf2503d0b01bc354cd60fb53a5` | - | curl config pinning the gg-armv8-O2-instruct SFT shard URLs (HF revision 5c98f387) downloaded for the survey. |
| `mac-scratch-main:d3bd2ab3/prep.py` | `ops/mac-scratch/d3bd2ab3/prep.py` | 792 | `2d92e853c4c25b0611339a1f88b5aba8b7d9e4aab4500fb67da3f36cce2c5279` | - | Extracts HumanEval-C source and official x86/arm asm (gg-official rows) into cmp/src and cmp/off for the local recompile comparison. |
| `mac-scratch-main:d3bd2ab3/q.py` | `ops/mac-scratch/d3bd2ab3/q.py` | 886 | `549b759095b3f09fe5880d1789caac1fbed973d2f2c0e6382450ed8cf3563c6f` | - | Counts .build_version/.ident directives by x86_origin in rl/artifacts parquets (toolchain fingerprint of official vs rebuilt asm). |

### `ops/ponyo/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `ponyo:~/gg/ggrpo/runs/queue.sh` | `ops/ponyo/ggrpo_runs/queue.sh` | 1111 | `0da94151d7a37bd2a49454c85b1b17d47a35fd8935d2f623e2dc783259b038b5` | - | ponyo GPU queue: paper-as-released run (ggrpo.paper), then the three package-equivalence runs (paper flash/sdpa, modern sdpa). |
| `ponyo:~/gg/ggrpo/runs/queue.sh.bak` | `ops/ponyo/ggrpo_runs/queue.sh.bak` | 1153 | `ab484e972f0aa155bc7842628d94287eb360305b7c368ee087f1497341115ae5` | - | Earlier queue.sh that also ran the modern-stack beam generation (later moved to totoro GPU1). |
| `ponyo:~/gg/ggrpo/runs/eval_ponyo.sh` | `ops/ponyo/ggrpo_runs/eval_ponyo.sh` | 2013 | `7552510ad64f80a5dbd318a485414d13788b2fe0f4c6430670ae7e8691acec19` | - | Current ponyo eval lane: after paper seed 1, cross-machine dev8 identity check, then dev on relayed checkpoints 81 and 106. |
| `ponyo:~/cs8001-step0/run_bench.sh` | `ops/ponyo/cs8001-step0/run_bench.sh` | 517 | `06740df9fdb3a08cd49b94759f739ed283cdaaa8de00b2cea86495e7fa2de82d` | - | Step-0 speed bench on ponyo: 3 HumanEval prompts, max_new_tokens 4096. |
| `ponyo:~/cs8001-step0/run_full.sh` | `ops/ponyo/cs8001-step0/run_full.sh` | 241 | `07c62f2caef15613d6087e74c99e921ceffdea40341719c4974a9edf8944f0c3` | - | Step-0 full HumanEval-C O2 generation on ponyo (beam 8, 16000-token cap) -> preds_full_raw.jsonl. |
| `ponyo:~/cs8001-step0/run_rep1.sh` | `ops/ponyo/cs8001-step0/run_rep1.sh` | 469 | `18902c32c2601a235bc9f9b7e11a017274e31ac222e89522161e7caabd4bd762` | - | Step-0 rerun with repetition_penalty 1.0 (gen_v2.py --rep 1.0) -> preds_rep1_raw.jsonl. |
| `ponyo:~/cs8001-step0/gen_v2.py (copied from an identical local file, sha256 checked on ponyo)` | `ops/ponyo/cs8001-step0/gen_v2.py` | 2501 | `ff9af15b5827466988283f49b062ba7ae880055ee5e6ff01f4d6ec52256f4b6b` | - | Step-0 generator with the --rep override for repetition_penalty (the checkpoint's generation_config sets 1.1). Identical copies (same sha256): `mac:~/git/Guaranteed-Guess/rl_simple/gen.py`. |
| `ponyo:~/cs8001-step0/v450/gen_authors.py` | `ops/ponyo/cs8001-step0/v450/gen_authors.py` | 3159 | `614591729b1ac6f1dc59ac65ec7f2b5e3662e519025eab339244bfc71fd61fea` | - | eval_huamaneval.py's inference() verbatim under transformers 4.50.0 (authors = beam sampling; greedy_beam = do_sample False). Identical copies (same sha256): `mac-scratch:643399c8/ponyo/gen_authors.py`. |
| `ponyo:~/cs8001-step0/v450/run_v450.sh` | `ops/ponyo/cs8001-step0/v450/run_v450.sh` | 655 | `555cb9a2129af597d253d45fb0a740255d6619fd0a2675f93ab64a6a902b4213` | - | Full HumanEval-C O2 runs with transformers 4.50.0 + torch 2.6.0 + FA2 (greedy_beam, authors seeds 0 and 1). Identical copies (same sha256): `mac-scratch:643399c8/ponyo/run_v450.sh`. |

### `ops/totoro/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `totoro:~/ggrpo/runs/after_smoke.sh` | `ops/totoro/ggrpo_runs/after_smoke.sh` | 730 | `cdc50bf8761a3c8d4b986e9c47cde4742c85ef23f6030554677285a55eae5ef2` | - | After the no-vLLM smokes: chunked-prefill equivalence check, then resumes BringUp beam8 on GPU0. |
| `totoro:~/ggrpo/runs/dev_curve.sh` | `ops/totoro/ggrpo_runs/dev_curve.sh` | 2150 | `09b26609c985496dea6889a25b6f703135ff633793a4b5603c1ce21b7bda86f7` | - | Dev-curve watcher on GPU1: scores base dev, then generates+scores dev for checkpoints 27/54/81/106 of runs/full (HumanEval on 106). |
| `totoro:~/ggrpo/runs/eval_gpu1.sh` | `ops/totoro/ggrpo_runs/eval_gpu1.sh` | 1913 | `5eb0bac7ec23956cde06d0add1160cbc0ce3bc38eb5bc61da7e84ae3ca0a6f51` | - | Current GPU1 eval lane: dev on checkpoints 27 and 54, HumanEval-C on 106; generation only, scoring from the Mac against the VM. |
| `totoro:~/ggrpo/runs/full.sh` | `ops/totoro/ggrpo_runs/full.sh` | 539 | `536e292657bebb2073f6422033acffd9bfc096d4ea2d207a9ca9d62af27e76b4` | - | Launches the live full-parameter GRPO run (trl grpo --config grpo.yaml) on GPU0 with nvidia-smi logging; restart after the pilot OOM. |
| `totoro:~/ggrpo/runs/probe_full.sh` | `ops/totoro/ggrpo_runs/probe_full.sh` | 757 | `32195ef45ab7bee9f09643e2afc1c883a29065edf13070ea2e8baff115ad70b2` | - | 2-step full-FT probe with the real grpo.yaml (GPU0 memory every 15 s), output runs/probe_full. |
| `totoro:~/ggrpo/runs/smoke_hf.sh` | `ops/totoro/ggrpo_runs/smoke_hf.sh` | 969 | `34aa2dfcca80b7b8ed5b76a1046d5eb3ddc8ca3ae2e4acd4c4a9184cd1a2706a` | - | 2-step no-vLLM smokes: LoRA then QLoRA (NF4, nested quant) with runs/smoke_hf.yaml, to measure step time and memory. |
| `totoro:~/ggrpo/runs/smoke_hf.yaml` | `ops/totoro/ggrpo_runs/smoke_hf.yaml` | 2703 | `a6e998187f2277f4f71dab1a699ff03f442a437ccc5a54283db673d9822294c5` | - | TRL GRPO config used by the LoRA/QLoRA no-vLLM smokes (values agreed Oct 22, LoRA era). |
| `totoro:~/ggrpo/runs/peft_merge_check.py` | `ops/totoro/ggrpo_runs/peft_merge_check.py` | 2914 | `149d5213ffeab00fe4cec0addec123b8e9dbc837799db2f1d4c401069c4f1622` | - | Measures PEFT merge/unmerge on the QLoRA model (what TRL's vLLM sync pushes): fraction of 4-bit codes changed. Identical copies (same sha256): `mac-scratch:643399c8/totoro/peft_merge_check.py`. |
| `totoro:~/ggrpo/runs/peft_merge_check_bf16.py` | `ops/totoro/ggrpo_runs/peft_merge_check_bf16.py` | 1771 | `b445a7644b4ce3af8e34e9764e701d4eadd271af9622be91243f42cf71f23f94` | - | Same as peft_merge_check.py for the LoRA arm merged into the bf16 base. Identical copies (same sha256): `mac-scratch:643399c8/totoro/peft_merge_check_bf16.py`. |
| `totoro:~/ggrpo/runs/prefill_chunk_check.py` | `ops/totoro/ggrpo_runs/prefill_chunk_check.py` | 2673 | `f8d14a70655bc909ef4d629f1c962448ad84ec039d0ff6a6a6402036b5470bb6` | - | Does transformers' chunked prefill (512) change generation? Compares logits and 256 greedy tokens, bf16 and NF4, with peak memory. Identical copies (same sha256): `mac-scratch:643399c8/totoro/prefill_chunk_check.py`. |
| `totoro:~/ggrpo/runs/requant_merge.py` | `ops/totoro/ggrpo_runs/requant_merge.py` | 2755 | `ca36e22ebfb91a2d6b8a46b8efc680963e26879d262a542b89bcd0411298468a` | - | How much of a LoRA delta survives TRL's QLoRA sync path (dequantize, add, requantize); uses the real smoke adapter. Identical copies (same sha256): `mac-scratch:643399c8/totoro/requant_merge.py`. |
| `totoro:~/ggrpo/runs/pilot/full.sh` | `ops/totoro/ggrpo_runs/pilot/full.sh` | 754 | `2a0e7ec5a7c8886760cda174dcee77ed62464c9fc4e7a73d6e6291f37afd74b8` | - | Pilot full-parameter GRPO launcher (then resume BringUp beam8); the pilot OOMed at step 19. |
| `totoro:~/ggrpo/runs/stress/stress.sh` | `ops/totoro/ggrpo_runs/stress/stress.sh` | 791 | `c948896f8aff8e0a6de6ea31eeb87eb178ba96272fd8d766db05da2b1425f704` | - | Worst-case memory test: one GRPO step on the 16 longest train prompts (micro-batch 4 x 32), waits for the reward tunnel. |
| `totoro:~/ggrpo/runs/stress/grpo.yaml` | `ops/totoro/ggrpo_runs/stress/grpo.yaml` | 2388 | `f468b4e8ddc3368e5f24fcc7145c6c1e42fe593914b82a772a1dee24ecce23d1` | - | grpo.yaml variant used by the stress test (full-parameter GRPO, train set = stress/longest16.jsonl). |
| `totoro:~/cs8001-bringup/gen_bringup.py` | `ops/totoro/cs8001-bringup/gen_bringup.py` | 3269 | `73d4c878a7004c93bd2c5f05dabb9b478610c6b1f6a8d31d973ead25a04ffb90` | - | BringUp O2 generation with gg-armv8-O2 (released = eval_bringupbench.py greedy 32768; beam8 = paper beam 8, cap 32000-prompt); resumable JSONL. Identical copies (same sha256): `mac-scratch:643399c8/totoro/gen_bringup.py`. |
| `totoro:~/cs8001-bringup/run_bringup.sh` | `ops/totoro/cs8001-bringup/run_bringup.sh` | 795 | `bbeaaf532bf6c19aec22a9ad1a71057c2a66d0c1c6a865ed7f008681170972cb` | - | Runs both BringUp O2 modes side by side on the RTX PRO 6000 Blackwell after an FA2 check. Identical copies (same sha256): `mac-scratch:643399c8/totoro/run_bringup.sh`. |
| `totoro:~/cs8001-bringup/bringup_eta.py` | `ops/totoro/cs8001-bringup/bringup_eta.py` | 789 | `7e12d480d8b54b6341513301203847cd5e66c65bc583435f508cc00da3ca56d8` | - | Progress/ETA of the two BringUp runs (rows done, hours, remaining prompt tokens). Identical copies (same sha256): `mac-scratch:643399c8/totoro/bringup_eta.py`. |
| `totoro:~/cs8001-bringup/DATA_SOURCE.txt` | `ops/totoro/cs8001-bringup/DATA_SOURCE.txt` | 70 | `2e0037c979f1aba70251dcbee636920230210f8a2084f6142f5f1a2a5aaf0f6b` | - | Records the HF snapshot of gg-bench-bringup-O2 used (ae6623a3...). Identical copies (same sha256): `mac-scratch:643399c8/totoro/DATA_SOURCE.txt`. |

### `ops/vm/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `vm:~/scratch/authors_tests.py` | `ops/vm/scratch/authors_tests.py` | 2927 | `536ccee5b359abc7b3e35856bcab5a2893504fe28ae3c4d828b28a7359c1cb8b` | - | Authors' asmbench_source vs official gg-bench rows in the VM toolchain: recompiles O0/O2 byte-for-byte, runs reference/empty/step-0 preds against THEIR tests. Identical copies (same sha256): `mac-scratch:643399c8/vm/authors_tests.py`. |
| `vm:~/scratch/bringup_refs.py` | `ops/vm/scratch/bringup_refs.py` | 1105 | `ac63294e81ba031e9f9a160ce1d855f4756c722f83f5a2f25a80257052f00d20` | - | Upstream BringUp eval (make run-tests TARGET=host) on the official reference arm (writes <b>/<b>.s per row). Identical copies (same sha256): `mac-scratch:643399c8/vm/bringup_refs.py`. |
| `vm:~/scratch/bringup_score.py` | `ops/vm/scratch/bringup_score.py` | 2540 | `034d58df0cce6700cb55aee179d58b6ef56e7680e371bc7e3ded58debde15f46` | - | Upstream BringUp eval on a predictions file; fresh bench copy per call, kills runaway programs, missing preds count as failures. Identical copies (same sha256): `mac-scratch:643399c8/vm/bringup_score.py`. |
| `vm:~/scratch/score_runs.py` | `ops/vm/scratch/score_runs.py` | 2567 | `e07340fe8e66140d6518246a7f6ab9602ca066c993bafe3de6973c8eefed9b04` | - | Scores generation runs in the VM two ways: ours (LLM4Decompile driver, data/eval.jsonl) and authors (asmbench_source tests). Identical copies (same sha256): `mac-scratch:643399c8/vm/score_runs.py`. |
| `vm:~/scratch/shifted.py` | `ops/vm/scratch/shifted.py` | 988 | `277c246239640fa9f44c3063357e9bb9c311351e041d8047f4fef66085f2e13b` | - | Scores the 5 shifted HumanEval slots (slot k holds problem k+1's function) against problem k+1's tests. Identical copies (same sha256): `mac-scratch:643399c8/vm/shifted.py`. |
| `vm:~/ggrpo/runs/compare_build.py` | `ops/vm/ggrpo_runs/compare_build.py` | 501 | `f5f04ba1db6cf20ac9a0fa208ff9c47fb44950bf1e053ca6f118168e6ce51938` | - | Compares the ggrpo.build MBPP parquet with the older ventura parquet column by column (x86, arm_ref, driver, ref_verdict). Identical copies (same sha256): `mac-scratch:33ca3c48/compare_build.py`. |

### `results/bringup/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `totoro:~/cs8001-bringup/bringup_O2_released.jsonl` | `results/bringup/totoro/bringup_O2_released.jsonl` | 1191514 | `ffa98745c361892598eaab8c377145ecfe89a8a6bada2d24d0e7451f731f3c17` | - | BringUp O2 predictions, released mode (greedy, 32768 new tokens), RTX PRO 6000 Blackwell, 66 rows. Identical copies (same sha256): `mac-scratch:643399c8/totoro_out/bringup_O2_released.jsonl`. |
| `totoro:~/cs8001-bringup/bringup_O2_beam8.jsonl` | `results/bringup/totoro/bringup_O2_beam8.jsonl` | 591579 | `e5ed2f03e5471b8f86100ba6f3400384b587274452b717f0d7bad3f172048c67` | - | PARTIAL: BringUp O2 beam8 predictions on the Blackwell, 44/66 rows (paused 2026-10-22 20:24 UTC per run.log). |
| `totoro:~/cs8001-bringup/run.log` | `results/bringup/totoro/run.log` | 849 | `7b42ae647f0842d8bc0c107b7feaf7bba45e28c7c61d7ea5b03e074392933c88` | `.gitignore:62:*.log` | Event log of the Blackwell BringUp runs (FA2 check, pauses, resumes). |
| `totoro:~/cs8001-bringup/released.log` | `results/bringup/totoro/released.log` | 2553 | `45e88167c440eaec9c8534d6e4839bce98662e5bec679f08d1c1fbf977ec981b` | `.gitignore:62:*.log` | Per-row log of the released-mode run. |
| `totoro:~/cs8001-bringup/beam8.log` | `results/bringup/totoro/beam8.log` | 1299 | `dc8589209a6e23b82a8d1760d8df2aeaa8e62a4a6b3129eb3574122677e17145` | `.gitignore:62:*.log` | Per-row log of the beam8 run (first session). |
| `totoro:~/cs8001-bringup/beam8_resume.log` | `results/bringup/totoro/beam8_resume.log` | 2100 | `674d949331e5c9cce6b6228f4e95c53511f830d1eeb2f6ed50b1b4f95d20e0b0` | `.gitignore:62:*.log` | Per-row log of the beam8 run after resumes. |
| `vm:~/scratch/bringup_O2_released.preds.json` | `results/bringup/vm/bringup_O2_released.preds.json` | 1069014 | `0450c72e1726415b846b916b6d3c7a0fbd666c070a7a7d2cae20804232d2f21f` | - | save_preds-style JSON of the released-mode Blackwell run (bringup_to_preds.py). Identical copies (same sha256): `mac-scratch:643399c8/totoro_out/bringup_O2_released.preds.json`. |
| `vm:~/scratch/bringup_O2_released.scored.jsonl` | `results/bringup/vm/bringup_O2_released.scored.jsonl` | 4289 | `27286bf1cebada8ed5cb812882d78a1faf10b349a8ef1c13787fb86ea0e71001` | - | Upstream BringUp verdicts for the released-mode run (bringup_score.py). |
| `vm:~/scratch/bringup_O2_beam8_partial41.preds.json` | `results/bringup/vm/bringup_O2_beam8_partial41.preds.json` | 511517 | `dfabf2ee7d3f23064e7273dbd3745903739b3b969a70df911b1e00d2247aa1d5` | - | save_preds-style JSON of the first 41 beam8 rows. Identical copies (same sha256): `mac-scratch:643399c8/totoro_out/bringup_O2_beam8_partial41.preds.json`. |
| `vm:~/scratch/bringup_O2_beam8_partial41.scored.jsonl` | `results/bringup/vm/bringup_O2_beam8_partial41.scored.jsonl` | 4364 | `f60c971d0140ccb4a09a1c15357faaa505f7885a4cf2f54fa6c8a13d5191a8ca` | - | Upstream BringUp verdicts for the first 41 beam8 rows. |
| `vm:~/scratch/bringup_O2_refs.preds.json` | `results/bringup/vm/bringup_O2_refs.preds.json` | 1648787 | `42b1c589789b2794236fd6f2ab092cba0b485e5db08772d2eaf8179b3b926dc3` | - | Official reference arm as a predictions file (bringup_refs.py). |
| `vm:~/scratch/bringup_O2_refs.scored.jsonl` | `results/bringup/vm/bringup_O2_refs.scored.jsonl` | 4290 | `5788aba5d5a4ccd3cb9a137298d8ea2cf0dcf9ea9dff77bd8d7dfa3a9cb1e2f0` | - | Upstream BringUp verdicts for the official O2 references. |
| `vm:~/scratch/bringup_sample_O0.scored.jsonl` | `results/bringup/vm/bringup_sample_O0.scored.jsonl` | 2787 | `9bf35636a4f736af4161e2cb8898f1c21ee86f79113c3d1350036d42a2b7e6b0` | - | Upstream BringUp verdicts for the authors' released O0 sample results. Identical copies (same sha256): `mac-scratch:643399c8/bringup_sample_O0.scored.jsonl`. |
| `cluster:~/cs8001-bringup/test/long.jsonl` | `results/bringup/cluster/test/long.jsonl` | 47753 | `2ec2cb37518f698f3aab875ed56f584838064d1135d29306b020e14c459124da` | - | Cluster RTX 5000 Ada test run on long BringUp prompts (beam8), for GPU comparison with the Blackwell. Identical copies (same sha256): `mac-scratch:643399c8/cluster_out/long.jsonl`. |
| `cluster:~/cs8001-bringup/test/short.jsonl` | `results/bringup/cluster/test/short.jsonl` | 27534 | `ae70d93f81fd514c96582708bc1f5700b0c2da331968c6efce64ca3b4e4f1fb5` | - | Cluster RTX 5000 Ada test run on short BringUp prompts. Identical copies (same sha256): `mac-scratch:643399c8/cluster_out/short.jsonl`. |
| `cluster:~/cs8001-bringup/logs/b8-a-243364.log` | `results/bringup/cluster/logs/b8-a-243364.log` | 1857 | `f0d851b7f6d1a5555d0dfa3c47311bc96f7e68741ca25547d0dbbb675fc4036e` | `.gitignore:62:*.log` | Slurm log of cluster job b8-a-243364 (per-row file, prompt tokens, new tokens, seconds). |
| `cluster:~/cs8001-bringup/logs/b8-b-243365.log` | `results/bringup/cluster/logs/b8-b-243365.log` | 1762 | `708ea0b799ef110c831276677a0cd33ff7daf695e542bbd1e6fa5e6cecd4a429` | `.gitignore:62:*.log` | Slurm log of cluster job b8-b-243365 (per-row file, prompt tokens, new tokens, seconds). |
| `cluster:~/cs8001-bringup/logs/t-long-243358.log` | `results/bringup/cluster/logs/t-long-243358.log` | 1139 | `0c65f0feca67f35de52880c8761aa9308a79790aca6461ae37a388d640b16cb6` | `.gitignore:62:*.log` | Slurm log of cluster job t-long-243358 (per-row file, prompt tokens, new tokens, seconds). |
| `cluster:~/cs8001-bringup/logs/t-short-243357.log` | `results/bringup/cluster/logs/t-short-243357.log` | 1272 | `4f43aff28303056fb31999bcdb27b1d8e93d993e9535bab8aae5b2e2d4ae8598` | `.gitignore:62:*.log` | Slurm log of cluster job t-short-243357 (per-row file, prompt tokens, new tokens, seconds). |

### `results/data-survey/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `mac-scratch:9da06990/decontam.json` | `results/data-survey/decontam.json` | 1032 | `85e2c723c8858e1b89da273a9d77ef61640aaed5344d9dcb05a7fe476bf03e64` | - | LIVE session snapshot. Decontamination verdicts: BringUp programs copied in GG's SFT Stack files, partial matches, MBPP/HumanEval twins. |
| `mac-scratch:9da06990/neardup.log` | `results/data-survey/neardup.log` | 3248 | `427e41f6d4f5197693d9a5be88b48aa284ed56b0f6fb78d05cf09d438cb15a7e` | `.gitignore:62:*.log` | LIVE session snapshot. Log of neardup.py (C 13-gram containment and asm function matching against the SFT data). |
| `mac-scratch:9da06990/sameprob.txt` | `results/data-survey/sameprob.txt` | 19893 | `ede26341898856745596bd8a9e2eccd01c1ce3a8813e5af5a6a34f9be01e8769` | - | LIVE session snapshot. Candidate same-problem MBPP/HumanEval pairs for manual review (sameprob.py output). |
| `mac-scratch:9da06990/angha_top.txt` | `results/data-survey/angha_top.txt` | 1225 | `f2ad17ca307c6e704a5f8262247649f4b7ab1ad3c70101c8cd0a91de55133e2c` | - | LIVE session snapshot. Top AnghaBench repository names used by synth.py/overlap.py. |
| `mac-scratch:9da06990/ct/results.txt` | `results/data-survey/ct/results.txt` | 1880 | `dbdb84f0b166d32e05607747845ea29b2d61f22b87ef4b092912918f958a5392` | - | LIVE session snapshot. ExeBench test_real compile portability on macOS clang (arm/x86 ok rates, top errors). |
| `mac-scratch:9da06990/scripts/cf_count2.log` | `results/data-survey/scripts/cf_count2.log` | 1523 | `4fa56a062a5e074888c7ec4563cd7d6dc6f781f3be43917d1d877d6be92bde30` | `.gitignore:62:*.log` | LIVE session snapshot. Progress log of cf_count2.py (row counts over the SFT shards). |
| `mac-scratch:9da06990/fn_matches.parquet` | `results/data-survey/fn_matches.parquet` | 5561 | `b46362c397397e5b9e1e543c876a51719666b945e06cfca6b6d0c1699e030478` | - | LIVE session snapshot. Function-level matches from the overlap probe. |
| `mac-scratch:9da06990/unit_matches.parquet` | `results/data-survey/unit_matches.parquet` | 23590 | `53725731843aeec8bb96d499867ca4e509bb51bd29f1a8610bace595eab63f40` | - | LIVE session snapshot. Unit-level matches from the overlap probe. |
| `mac-scratch:9da06990/neardup_c.parquet` | `results/data-survey/neardup_c.parquet` | 42929 | `ae41d2d1013d1c6b792f465e409f13ffe13c86065712f9f75c99efac2125726a` | - | LIVE session snapshot. C-level near-duplicate hits per probe. |
| `mac-scratch:9da06990/neardup_c_size.parquet` | `results/data-survey/neardup_c_size.parquet` | 7199 | `da5da61459edd89de1a28157ef74c40fa294acadef4aef018dad4f984d3daf43` | - | LIVE session snapshot. C-level near-duplicate hit sizes. |
| `mac-scratch:9da06990/neardup_asm.parquet` | `results/data-survey/neardup_asm.parquet` | 46689 | `bc079155e6e03e766135914d7f02c8519bf18b1fc6a65cf0ecf52afc29754e5c` | - | LIVE session snapshot. Asm-level exact function matches per probe. |
| `mac-scratch:9da06990/neardup_asm_size.parquet` | `results/data-survey/neardup_asm_size.parquet` | 13435 | `c1df5b0b4dd5ac6a8011c7294c8c6bb2776aecbf02dae9a7093789af946cc210` | - | LIVE session snapshot. Asm-level match sizes. |
| `mac-scratch:9da06990/neardup_runs.parquet` | `results/data-survey/neardup_runs.parquet` | 4889 | `2e07e128e058edc8dc43add8d57b8087ea16b2e2c66507397d81d1fef62c9db8` | - | LIVE session snapshot. Longest shared canonical-13-gram runs per probe (runs.py). |
| `mac-scratch:9da06990/sameprob_candidates.parquet` | `results/data-survey/sameprob_candidates.parquet` | 8533 | `7ff7628d8971d1562f4a3dcd32eca8640c9d1ffb878020ed104d32626ac3a675` | - | LIVE session snapshot. Shortlisted same-problem candidates. |
| `mac-scratch:9da06990/sameprob_embed.parquet` | `results/data-survey/sameprob_embed.parquet` | 13843 | `4b0940fc641f074884ee1315ae0e93cad615203b699e0eb6c74f5aace945ee4e` | - | LIVE session snapshot. Embedding-similarity top-3 MBPP per HumanEval (embed.py). |
| `mac-scratch:9da06990/sameprob_pairs.parquet` | `results/data-survey/sameprob_pairs.parquet` | 895887 | `689c68b9148ae671ba225f4becab76e888fab59d551a44767651e03b6686cdf3` | - | LIVE session snapshot. All scored MBPP/HumanEval pairs (TF-IDF cosine, 5-gram Jaccard). |

### `results/dev/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `totoro:~/ggrpo/runs/modern/dev_beam_search.jsonl` | `results/dev/modern/dev_beam_search.jsonl` | 349726 | `111d670faa1b912b6abef9651151bd50d21399e6401449aa5e56f54e87016579` | - | PARTIAL (75 rows at copy time, still being written): base model on data/dev.jsonl, modern stack, totoro GPU1, beam 8. |
| `totoro:~/ggrpo/runs/modern_dev_beam_search.log` | `results/dev/modern/modern_dev_beam_search.log` | 23458 | `1ae2ce27457d5af14e7bb2fb2e2ab0bcd9677ca927315671a5125fad5dca827f` | `.gitignore:62:*.log` | Log of the base dev beam generation on totoro GPU1 (in progress at copy time). |

### `results/equivalence/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `ponyo:~/gg/ggrpo/runs/equivalence/paper_flash.jsonl` | `results/equivalence/paper_flash.jsonl` | 3016597 | `54d36a6a92c7e81122720b4a4709afb0bedd08bc3112b8a540213d5d5ad37eba` | - | Package-equivalence reference: paper stack, flash_attention_2; per-task prompt/completion id hashes and logprobs (157 tasks). |
| `ponyo:~/gg/ggrpo/runs/equivalence/paper_sdpa.jsonl` | `results/equivalence/paper_sdpa.jsonl` | 3016771 | `f13a8cc736d09e6190398b01b7351d3ec7dba1ce38aefd9a03cc05ae869d7757` | - | Paper stack with sdpa attention; compared against paper_flash (157/157 same token ids). |
| `ponyo:~/gg/ggrpo/runs/equivalence/modern_sdpa.jsonl` | `results/equivalence/modern_sdpa.jsonl` | 3015617 | `405ebfa971369dd6710d7ab0a155b4e4197e994d29cb91921fbd1ea8a57b2b1f` | - | Modern stack with sdpa attention; compared against paper_flash (157/157 same token ids). |
| `ponyo:~/gg/ggrpo/runs/equivalence/cpu_test_a.jsonl` | `results/equivalence/cpu_test_a.jsonl` | 67729 | `b6598e61964ffe167dfa6633ca2873653b9c37fc78207a89b7d554f8a6fc6302` | - | CPU dry run A of ggrpo.equivalence (most tasks skipped), used to test the comparison. |
| `ponyo:~/gg/ggrpo/runs/equivalence/cpu_test_b.jsonl` | `results/equivalence/cpu_test_b.jsonl` | 68998 | `61953b4dff99e87499010c4151c57740e2add5d340630f8dddf8f3fc546e7fa1` | - | CPU dry run B of ggrpo.equivalence, compared against A. |
| `ponyo:~/gg/ggrpo/runs/equivalence_paper_flash.log` | `results/equivalence/equivalence_paper_flash.log` | 395 | `bb072b21c709faf1f5a5f492b94603fe5182e7130f87b92d6d5d2b798c89a57d` | `.gitignore:62:*.log` | Log of the paper_flash equivalence run, including the comparison summary line. |
| `ponyo:~/gg/ggrpo/runs/equivalence_paper_sdpa.log` | `results/equivalence/equivalence_paper_sdpa.log` | 967 | `fa8de9599b7f98a7e178f3afa57dc6cfce06728a01129fcd9c97b1206a9df91e` | `.gitignore:62:*.log` | Log of the paper_sdpa equivalence run, including the comparison summary line. |
| `ponyo:~/gg/ggrpo/runs/equivalence_modern_sdpa.log` | `results/equivalence/equivalence_modern_sdpa.log` | 1675 | `940093b4eab606f9f66c048dba1c5b56f9f5ed1554f2039de67d7fbc284be7a5` | `.gitignore:62:*.log` | Log of the modern_sdpa equivalence run, including the comparison summary line. |
| `ponyo:~/gg/ggrpo/runs/queue.log` | `results/equivalence/queue.log` | 620513 | `d8409977b959e214fc06b5ada5ecdfd66bed8f653b2e88a09027a02702b71a23` | `.gitignore:62:*.log` | Combined log of the ponyo queue (paper as released, then the three equivalence runs). |
| `ponyo:~/gg/ggrpo/runs/xcheck/eval6.jsonl` | `results/equivalence/xcheck/eval6.jsonl` | 17367 | `1945c28ed94968c9e0565ea2771463d4e7300a594cfeff3f45db973807d9e366` | - | Six HumanEval eval tasks used for the cross-machine generation check. Identical copies (same sha256): `mac-scratch:643399c8/xcheck/ponyo/eval6.jsonl`. |
| `ponyo:~/gg/ggrpo/runs/xcheck/r1.jsonl` | `results/equivalence/xcheck/r1.jsonl` | 6820 | `69b2115328df868016f7796fa6e0fd9cfc9453485b047100e4eace66cde3fb41` | - | Cross-check run 1 generations on ponyo (6 tasks). Identical copies (same sha256): `mac-scratch:643399c8/xcheck/ponyo/r1.jsonl`. |
| `ponyo:~/gg/ggrpo/runs/xcheck/r2.jsonl` | `results/equivalence/xcheck/r2.jsonl` | 6820 | `c428cc75b43de49ed1821d0945d99ac4de1e31580232640ada6a97f0b80f60f2` | - | Cross-check run 2 generations on ponyo (6 tasks). Identical copies (same sha256): `mac-scratch:643399c8/xcheck/ponyo/r2.jsonl`. |
| `ponyo:~/gg/ggrpo/runs/xcheck/r1.log` | `results/equivalence/xcheck/r1.log` | 1833 | `e2698ae9382e18eec9b4cebe875d9e4e77a2c91ea8f5feeb6b9b44a2c0cdd85b` | `.gitignore:62:*.log` | Log of cross-check run 1. Identical copies (same sha256): `mac-scratch:643399c8/xcheck/ponyo/r1.log`. |
| `ponyo:~/gg/ggrpo/runs/xcheck/r2.log` | `results/equivalence/xcheck/r2.log` | 1847 | `5210cd125162d738cd4069c84de212c56562eab23ccdcdca8c78ca0257bd3a06` | `.gitignore:62:*.log` | Log of cross-check run 2. Identical copies (same sha256): `mac-scratch:643399c8/xcheck/ponyo/r2.log`. |

### `results/humaneval/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `totoro:~/ggrpo/runs/modern/beam_search.jsonl` | `results/humaneval/modern/beam_search.jsonl` | 796444 | `59330d006690bb7f4050a381529fda05e0defbf60dc8917f809fb63a3714beac` | - | Base model HumanEval-C O2 predictions, modern stack on totoro GPU1 (RTX 4070 Ti SUPER), beam 8, cap 32000-prompt (72/164 per eval_gpu1.sh). Identical copies (same sha256): `vm:~/scratch/modern_beam_search.jsonl`; `mac-scratch:643399c8/totoro_out/modern_beam_search.jsonl`; `mac-scratch:643399c8/totoro_out/modern_full.jsonl`. |
| `vm:~/scratch/modern_beam_search.scored.jsonl` | `results/humaneval/modern/modern_beam_search.scored.jsonl` | 811256 | `fd4a87c1a9aac4fcb498d48bc6d3956177aa371765197a182f4dc3a1f5c717d1` | - | VM-scored version of modern/beam_search.jsonl (score_runs.py: ours and authors verdicts). |
| `totoro:~/ggrpo/runs/modern_beam_search.log` | `results/humaneval/modern/modern_beam_search.log` | 40235 | `e956e84a55961c0811ee854ad0fc865df907dd08989c1800a9d878ac2f01e47a` | `.gitignore:62:*.log` | Log of the modern-stack HumanEval beam generation on totoro GPU1. |
| `totoro:~/ggrpo/runs/modern_beam_search.failed_no_python_h.log` | `results/humaneval/modern/modern_beam_search.failed_no_python_h.log` | 8545 | `3ec80338eba080effce997d359bd882ea542e5015e0fad34da2256e9cee4c8ab` | `.gitignore:62:*.log` | Failed first attempt of the modern beam generation (missing Python.h for a compiled extension). |
| `totoro:~/ggrpo/runs/modern_beam_search.failed_empty_resume.log` | `results/humaneval/modern/modern_beam_search.failed_empty_resume.log` | 1618 | `d114c33f4bd67849cc1f87d766ee55d9df3fb1ee7d0c892f33e7437b2d6bacc7` | `.gitignore:62:*.log` | Failed attempt of the modern beam generation (empty resume file crash, fixed in generate.py). |
| `ponyo:~/gg/ggrpo/runs/paper/as_released/predictions.jsonl` | `results/humaneval/paper_as_released/predictions.jsonl` | 673525 | `bcf3878820e64c5a2a7bb47a7de712163cadb1cc885721381d9c6bf124b8de3a` | - | Run A: authors' eval_huamaneval.py as released (beam sampling, seed 42) on ponyo, HumanEval-C O2 predictions. Identical copies (same sha256): `vm:~/scratch/runA_predictions.jsonl`; `mac-scratch:643399c8/ponyo_out/runA_predictions.jsonl`. |
| `ponyo:~/gg/ggrpo/runs/paper/as_released/eval_armv8_O2.json` | `results/humaneval/paper_as_released/eval_armv8_O2.json` | 1547093 | `5369710ab321198866901e0b97aa25498e685751c5f757799507093cae5b7165` | - | Run A raw output of the authors' script (save_preds-style JSON with pred list). |
| `vm:~/scratch/runA_predictions.scored.jsonl` | `results/humaneval/paper_as_released/runA_predictions.scored.jsonl` | 696833 | `7611e093f33b3f6f702b49a5aa686e8e75df552a1ad121f89d6933be7bf54aa7` | - | VM-scored Run A predictions (score_runs.py). |
| `ponyo:~/gg/ggrpo/runs/paper_as_released.log` | `results/humaneval/paper_as_released/paper_as_released.log` | 617411 | `21a0bbf725fa85be8a25b1b49c2c99349029238dc1bd1f78d815e4545fa23499` | `.gitignore:62:*.log` | Log of the Run A (paper as released) generation on ponyo. |
| `ponyo:~/cs8001-step0/v450/full_greedy_beam.jsonl` | `results/humaneval/v450/full_greedy_beam.jsonl` | 770536 | `750b3580ceeaaf7fed8533df7be59b2b2ae1361003a40f3ab220566c81bfd2d2` | - | Run B: HumanEval-C O2 with transformers 4.50.0 stack, greedy beam 8 (paper text), cap 32000-prompt, ponyo. Identical copies (same sha256): `vm:~/scratch/full_greedy_beam.jsonl`; `mac-scratch:643399c8/ponyo_out/full_greedy_beam.jsonl`; `mac-scratch:643399c8/ponyo_out/runB_full.jsonl`. |
| `vm:~/scratch/full_greedy_beam.scored.jsonl` | `results/humaneval/v450/full_greedy_beam.scored.jsonl` | 17215 | `5e90aaaeaae9edd97ec6977126b62f01a15375e8ad5e9e681a4532e9fc403fed` | - | VM-scored Run B (ours and authors verdicts). Identical copies (same sha256): `mac-scratch:643399c8/ponyo_out/full_greedy_beam.scored.jsonl`. |
| `ponyo:~/cs8001-step0/v450/full.log` | `results/humaneval/v450/full.log` | 5953 | `f3acab14372342239dabafebcb7979cdd85f158e4be0b507ed3d41dd66808827` | `.gitignore:62:*.log` | Per-row log of the v450 runs (file, prompt tokens, new tokens, seconds). |
| `ponyo:~/cs8001-step0/v450/smoke_greedy.jsonl` | `results/humaneval/v450/smoke_greedy.jsonl` | 4423 | `1def77f9a647bbfe5c29051846c3fbb803d5d48205dc4e680408c1903e7e1a3a` | - | v450 smoke: greedy_beam mode on the first rows. Identical copies (same sha256): `mac-scratch:643399c8/ponyo_out/smoke_greedy.jsonl`. |
| `ponyo:~/cs8001-step0/v450/smoke_authors_s0.jsonl` | `results/humaneval/v450/smoke_authors_s0.jsonl` | 4411 | `c798ce4f77aa9d2595c8287e14435740469b5c972bfd773a37ab4466afbceb28` | - | v450 smoke: authors mode (beam sampling), seed 0. Identical copies (same sha256): `mac-scratch:643399c8/ponyo_out/smoke_authors_s0.jsonl`. |
| `ponyo:~/cs8001-step0/v450/smoke_authors_s1.jsonl` | `results/humaneval/v450/smoke_authors_s1.jsonl` | 4411 | `3c7d06de5c2e6d42c690d367205a8363a93b54dbc6023433d559f6bb721df039` | - | v450 smoke: authors mode (beam sampling), seed 1. Identical copies (same sha256): `mac-scratch:643399c8/ponyo_out/smoke_authors_s1.jsonl`. |
| `vm:~/scratch/authors_tests.jsonl` | `results/humaneval/authors_tests.jsonl` | 39022 | `0d71702292f0f817f39db666c1f052bd0105006096de8cc28557c4d5d47f6e8a` | - | Output of authors_tests.py: per problem/opt, x86/arm byte-identity with official asm and reference/empty/pred verdicts on the authors' tests. Identical copies (same sha256): `mac-scratch:643399c8/authors_tests.jsonl`. |

### `results/logs/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `totoro:~/ggrpo/runs/ggrpo.jsonl` | `results/logs/totoro_ggrpo.jsonl` | 65519 | `9dd60d512de5a7d4240f7710870872ed0e32a15d3468a84ad01b036f2248dc4a` | - | LIVE: structured log of every ggrpo command on totoro (data, generate, reward per training step). |
| `ponyo:~/gg/ggrpo/runs/ggrpo.jsonl` | `results/logs/ponyo_ggrpo.jsonl` | 11165 | `dc0edbd8b919f953ce4e6e9dfb9bf288203e0469f3cc1d9dfdf4d6ed2310c3e3` | - | Structured log of every ggrpo command on ponyo (paper, equivalence, generate). |
| `vm:~/ggrpo/runs/ggrpo.jsonl` | `results/logs/vm_ggrpo.jsonl` | 15387 | `6d24ebc7e911f3e76b1312521ab1c2024eaa0a97f4831aac1fd6b7409b10a308` | - | LIVE: structured log of the VM (data builds, reward server scoring batches for the live run). |
| `mac:~/git/Guaranteed-Guess-feature-env/ggrpo/runs/ggrpo.jsonl` | `results/logs/mac_ggrpo.jsonl` | 4980 | `eb6ea8d90bf3d4413dd7bce5b4e1cb7cf58d45b20e6da9c2f71e50d827a2a332` | - | Structured log of ggrpo commands run from the Mac worktree (gitignored runs/). |
| `vm:~/ggrpo/runs/server.out` | `results/logs/server.out` | 11460 | `b9d2e6a2cf1271bc421f65fdef5d83b8f69d4bb89f07ea28e4735acc77ece72b` | - | LIVE: stdout of the VM reward server (uvicorn on :8765) serving the training run. |

### `results/notes/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `mac-scratch:643399c8/handoff_grpo_parameters.md` | `results/notes/handoff_grpo_parameters.md` | 10213 | `bef5cdee1f9cf0b11259289012f7770a18363173dafc1b86c479b334feaedb21` | - | Handoff brief (Oct 22) for choosing the GRPO training parameters: verified facts, paper values, open questions. |
| `mac:~/git/Guaranteed-Guess/rl_simple/docs/project-knowledge.md` | `results/notes/project-knowledge.md` | 13165 | `dc3d591d9cb63af6f44fdd6380538c33c923705b090364a6dd848e27c738169d` | - | Working notes of everything known as of Oct 22: goal, repos, machines, pinned toolchain, step-0 results. |
| `mac-scratch:d37ed3f6/midpoint/midpoint_report.tex` | `results/notes/midpoint/midpoint_report.tex` | 23410 | `6ad9bda41fc218b5cf7e8ca51254c9b2fd2a3376e0b6e37ff1fe402859ed4bc9` | - | Midpoint report, first revision (diff is against ~/Downloads/midpoint_report.tex). |
| `mac-scratch:d37ed3f6/midpoint/midpoint_report.pdf` | `results/notes/midpoint/midpoint_report.pdf` | 60644 | `cb7bf4ffb787308d4a3ac1977c3b7c530a63e046a63c3be947cb5f9daa623768` | - | Midpoint report, first revision (diff is against ~/Downloads/midpoint_report.tex). |
| `mac-scratch:d37ed3f6/midpoint/midpoint_report.diff` | `results/notes/midpoint/midpoint_report.diff` | 22394 | `5d402d22c4a88391320492947aaa332f3e5ceaf719e113f731cf21af69081f51` | - | Midpoint report, first revision (diff is against ~/Downloads/midpoint_report.tex). |
| `mac-scratch:d37ed3f6/midpoint_v2/midpoint_report.tex` | `results/notes/midpoint_v2/midpoint_report.tex` | 20354 | `47de4a69221297e2293dcf66860f459914fce99911c8c449714bd30b1579184d` | - | Midpoint report v2 with tracked changes. |
| `mac-scratch:d37ed3f6/midpoint_v2/midpoint_report.pdf` | `results/notes/midpoint_v2/midpoint_report.pdf` | 77954 | `70cba4aded8fecc83bc573e9cdd7d78be9dfb03f343ce483c6b33ecda78c3cd1` | - | Midpoint report v2 with tracked changes. |
| `mac-scratch:d37ed3f6/midpoint_v2/midpoint_report_clean.tex` | `results/notes/midpoint_v2/midpoint_report_clean.tex` | 20355 | `db306b35464b5bcb70397aa0d4243a3b85f60e36e979459d1de99b21c28c5854` | - | Midpoint report v2 without tracked changes (clean). |
| `mac-scratch:d37ed3f6/midpoint_v2/midpoint_report_clean.pdf` | `results/notes/midpoint_v2/midpoint_report_clean.pdf` | 49447 | `b449bf33826a43e39ff6b137904fde26cb14818c427d9218ca7fa2bcdaf1e827` | - | Midpoint report v2 without tracked changes (clean). |
| `mac-scratch:d37ed3f6/midpoint_v2/midpoint_report.diff` | `results/notes/midpoint_v2/midpoint_report.diff` | 15112 | `aa721322c528a1508418810a789ca9a3f0bd98b2a5c70e4a44ecb7c31e4a099b` | - | Diff of the v2 report against ~/Downloads/midpoint_report.tex. |
| `mac-scratch:d37ed3f6/midpoint_v2/neurips_2026.sty` | `results/notes/midpoint_v2/neurips_2026.sty` | 13704 | `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a` | - | NeurIPS 2026 style file needed to build the report. Identical copies (same sha256): `mac-scratch:d37ed3f6/midpoint/neurips_2026.sty`. |

### `results/preflight/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `vm:~/ggrpo/runs/preflight/train_ref.jsonl` | `results/preflight/train_ref.jsonl` | 2266877 | `3293ce38495748248169ddb3444643a3bddfeff7c6e49de48f3ea8d9af6f4a23` | - | Reference solutions of the 952-row pilot training set scored on the VM harness (952/952 pass). |
| `vm:~/ggrpo/runs/preflight/eval_ref.jsonl` | `results/preflight/eval_ref.jsonl` | 399941 | `bbbe9fd107f043a1d509de7126f8dd63edcfa4d4a3eb44c755c15228f0b89f1e` | - | Reference solutions of the HumanEval-C eval set scored on the VM harness (164/164 pass). |
| `mac:~/git/Guaranteed-Guess-feature-env/ggrpo/runs/preflight/mac_eval_ref.jsonl` | `results/preflight/mac_eval_ref.jsonl` | 400328 | `22b8ef5570216e0f23f0f9284db27886a702bf60c8fd82a0b0dd460e5f1df872` | - | Same eval reference check run from the Mac (gitignored runs/ in this worktree). |

### `results/step0/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `mac:~/git/Guaranteed-Guess/rl_simple/step0_result.md` | `results/step0/step0_result.md` | 717 | `25ae9ba4cab34e704208731693a3caec094d88541a18cd8eb6696de72bee4d4f` | - | Step-0 baseline summary: unmodified gg-armv8-O2 on HumanEval-C O2, 68/164 = 41.46% (paper 45.12%), verdict breakdown. |
| `mac:~/git/Guaranteed-Guess/rl_simple/step0_scored_O2.jsonl` | `results/step0/step0_scored_O2.jsonl` | 35416 | `12fe9a37a284e8a35b5a1441c098e057f309324d86521e9c62746586e4f67c70` | - | Step-0 per-task verdicts (beam 8, 16000 cap, ponyo) scored in the gg-ventura VM. Identical copies (same sha256): `vm:~/rl_simple/scored_O2.jsonl`; `mac-scratch-main:d3bd2ab3/ponyo/scored_O2.jsonl`. |
| `mac:~/git/Guaranteed-Guess/rl_simple/step0_scored_rep1_O2.jsonl` | `results/step0/step0_scored_rep1_O2.jsonl` | 35075 | `9c97385b433c1de79f1884ce3e470ca23c92ca88924d8f9dc41b65d5b50317d5` | - | Step-0 verdicts for the repetition_penalty 1.0 rerun. Identical copies (same sha256): `vm:~/rl_simple/scored_rep1.jsonl`; `mac-scratch-main:d3bd2ab3/ponyo/scored_rep1.jsonl`. |
| `mac:~/git/Guaranteed-Guess/rl_simple/step0_preds_raw_rep1_ponyo_O2.jsonl` | `results/step0/step0_preds_raw_rep1_ponyo_O2.jsonl` | 1163431 | `c02bfd5aa165efe64cb81b23f2594b6737497b0476af6ceb679c1b6cf18d55d8` | - | Raw completions of the repetition_penalty 1.0 rerun on ponyo. Identical copies (same sha256): `ponyo:~/cs8001-step0/preds_rep1_raw.jsonl`; `mac-scratch-main:d3bd2ab3/ponyo/preds_rep1_raw.jsonl`. |
| `vm:~/rl_simple/preds_O2.jsonl` | `results/step0/preds_O2.jsonl` | 599469 | `f03583dee685e5c5ac81c2a34a39ad158118e53c6f592656ced5080189135b83` | - | Extracted step-0 predictions (beam 8) as scored by vm_check.py. Identical copies (same sha256): `mac-scratch-main:d3bd2ab3/ponyo/preds.jsonl`. |
| `vm:~/rl_simple/preds_rep1.jsonl` | `results/step0/preds_rep1.jsonl` | 578184 | `f7148f167845c9db85fe1c18103dba6f0cb2632131ff9d59b02e64a541529321` | - | Extracted predictions of the repetition_penalty 1.0 rerun as scored. Identical copies (same sha256): `mac-scratch-main:d3bd2ab3/ponyo/preds_rep1.jsonl`. |
| `mac:~/git/Guaranteed-Guess/rl_simple/bench_mps.jsonl` | `results/step0/bench_mps.jsonl` | 12643 | `0596b30279d33a3b1fa0f0aff2f48d6914139809bd754403be09ef0abe8f5bc0` | - | Step-0 speed bench on the Mac (MPS): 3 HumanEval prompts. |
| `mac:~/git/Guaranteed-Guess/rl_simple/bench_mps.log` | `results/step0/bench_mps.log` | 673 | `28e24b13c1192f2a470354d87e2e79a8115db8e7ef5a259c35b9e1d338c34cbc` | `.gitignore:62:*.log` | Log of the first Mac MPS bench. |
| `mac:~/git/Guaranteed-Guess/rl_simple/bench_mps2.log` | `results/step0/bench_mps2.log` | 407 | `b8858da75b1119af7e588e0b502076e6f96b45de997b4b46910acaadec52659c` | `.gitignore:62:*.log` | Log of the second Mac MPS bench. |
| `ponyo:~/cs8001-step0/bench.jsonl` | `results/step0/ponyo/bench.jsonl` | 13077 | `c92a3d740e71cb2a5399ad4c5082a2dffa7b7dcf1ffb476d8bb984cca5b29264` | - | Step-0 speed bench on ponyo (3 prompts, 4096 cap). |
| `ponyo:~/cs8001-step0/bench.log` | `results/step0/ponyo/bench.log` | 502 | `23bb6f8046107673effda267beaec51622364105ffaa5eb0ce78df532e0be0a3` | `.gitignore:62:*.log` | Log of the ponyo bench (torch/transformers versions, seconds per row). |
| `ponyo:~/cs8001-step0/full.log` | `results/step0/ponyo/full.log` | 4990 | `37c94cc6b27dc50405d2ed148417720271768ab543ab7b2e120ed947d5e225d4` | `.gitignore:62:*.log` | Log of the step-0 full run on ponyo. |
| `ponyo:~/cs8001-step0/rep1.log` | `results/step0/ponyo/rep1.log` | 5034 | `e55d08db78b482f28b60830f97a3e34022606f51ec4de3223ff354c6d6c8f683` | `.gitignore:62:*.log` | Log of the repetition_penalty 1.0 rerun on ponyo. |
| `vm:~/scratch/runs_step0.jsonl` | `results/step0/runs_step0.jsonl` | 608296 | `dc42a1aa219550075763d754d00d46e8ebaa9d59926fbf9f18575421806977c2` | - | Step-0 predictions converted to the generation-run format ({file, raw, ...}) for score_runs.py. Identical copies (same sha256): `mac-scratch:643399c8/vm/runs_step0.jsonl`. |
| `vm:~/scratch/runs_step0.scored.jsonl` | `results/step0/runs_step0.scored.jsonl` | 17190 | `b5ecd7e12634f85110d77ef2854fcff9aeef7e083726e2670e867588ac1e4a46` | - | Step-0 scored two ways (ours LLM4Decompile driver vs authors' tests) by score_runs.py. |
| `vm:~/ggrpo/runs/base/step0_beam8_16k.scored.jsonl` | `results/step0/base/step0_beam8_16k.scored.jsonl` | 626121 | `e28f82b6e242897b7fef41b52cdc21ff10e966c0dd8ed9e8bed638a6d94ed294` | - | ggrpo.score output on the VM for the step-0 beam8/16k predictions (per-task verdict, stderr tail). |
| `mac:~/git/Guaranteed-Guess-feature-env/ggrpo/runs/base/mac_step0.scored.jsonl` | `results/step0/base/mac_step0.scored.jsonl` | 626426 | `59e37354d5f998b1243b0261daacad3fe93ee34495cc67b7155651a246821056` | - | Same step-0 scoring run from the Mac against the VM server (gitignored runs/ in this worktree). |
| `vm:~/ggrpo/runs/old_vm_check_step0.jsonl` | `results/step0/old_vm_check_step0.jsonl` | 35411 | `d2330b27ae7918a38b6d34268297f7d68f8f51ef38eb1001cb70bd337d929cc1` | - | Older step-0 verdicts from rl_simple/vm_check.py, kept on the VM for comparison with ggrpo.score. |

### `results/toolchain/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `mac-scratch-main:d3bd2ab3/cmp/out.tgz` | `results/toolchain/cmp/out.tgz` | 238583 | `35d66dc1298367b07ab044bd9b070369a4eaecb070fb3fda227ba5339f1bf275` | - | Local recompile of 164 HumanEval-C sources (O0/O2, x86/arm .s) for comparison with the official asm (input: prep.py). |
| `mac-scratch-main:d3bd2ab3/bb/bbout.tgz` | `results/toolchain/bb/bbout.tgz` | 1159887 | `b8ea38187942e35e00fa44f90168ec98d810ce1a570f0fe927062cea49b1d4a4` | - | Local recompile of the BringUp benchmarks (O0/O2, x86/arm .s) for comparison with the official dataset asm. |
| `mac-scratch-main:d3bd2ab3/sweep2/p0.c` | `results/toolchain/sweep2/p0.c` | 300 | `cdd6a0cb0b42d63c88b0b28e717bc79c43c7bff3a1ce418329cd765600ddc7ff` | - | Toolchain sweep on HumanEval problem 0 variant: C input. |
| `mac-scratch-main:d3bd2ab3/sweep2/p0_ret10.c` | `results/toolchain/sweep2/p0_ret10.c` | 301 | `754398183284ba2ffe0994a134ede0ec83f0dd2df199f728e46f868d1407d565` | - | Toolchain sweep on HumanEval problem 0 variant: C input. |
| `mac-scratch-main:d3bd2ab3/sweep2/r10_O0.arm.s` | `results/toolchain/sweep2/r10_O0.arm.s` | 2001 | `aafdc0a2f62b9f1661eaca0ab69c48fabc5697d3111bdaec59af211a84a97478` | - | Toolchain sweep on HumanEval problem 0 variant: compiled output. |
| `mac-scratch-main:d3bd2ab3/sweep2/r10_O2.arm.s` | `results/toolchain/sweep2/r10_O2.arm.s` | 1308 | `dfbfaa5ab7fe77a32c3956cb2e38c486f4ebf8d0cd0fb67d583d331db4f6a02b` | - | Toolchain sweep on HumanEval problem 0 variant: compiled output. |
| `mac-scratch-main:d3bd2ab3/sweep2/r10_O0.x86.s` | `results/toolchain/sweep2/r10_O0.x86.s` | 2395 | `70f9615ccbd5900eeae426dd29e69389e2b302dfcbcc5808f3913884c05e32b5` | - | Toolchain sweep on HumanEval problem 0 variant: compiled output. |
| `mac-scratch-main:d3bd2ab3/sweep2/r10_O2.x86.s` | `results/toolchain/sweep2/r10_O2.x86.s` | 1939 | `b87ea1d2c2852ce41c60ea293d3461d1e809e620666e457d4222cbafc430b1b8` | - | Toolchain sweep on HumanEval problem 0 variant: compiled output. |

### `results/training/`

| Source | Destination | Bytes | sha256 | Ignored by | Description |
|---|---|---:|---|---|---|
| `totoro:~/ggrpo/runs/trackio/cs8001.db (sqlite backup API snapshot, 2026-10-22 21:15 UTC)` | `results/training/trackio/cs8001_snapshot.db` | 24911872 | `07a7b258ddf09166fd3545b3b4c9d0b22bd1d6f621b5cbe937c38f001c88a222` | - | Consistent snapshot of the live Trackio DB (project cs8001: all runs' metrics, configs, system metrics); integrity_check ok. |
| `totoro:~/ggrpo/runs/full.status` | `results/training/full.status` | 43 | `dcb8c44d91fc8879881329cfe8990cc7326bc1646d709ab4866267ceddb6a363` | - | LIVE: status of the full-parameter GRPO run (FULL_START only at copy time). |
| `totoro:~/ggrpo/runs/full_gpu.csv` | `results/training/full_gpu.csv` | 1665 | `9ce393961534150b8d055cee901c36f9628633ded7fbdb15c3f28ddd5f15f294` | - | LIVE: GPU0 memory/utilization every 60 s for the full run (snapshot). |
| `totoro:~/ggrpo/runs/dev_curve.status` | `results/training/dev_curve.status` | 44 | `fff372a9409870946c680282c55ac843e281962f604e4a999785e66ec0f1100a` | - | Status of dev_curve.sh (WATCH_START only). |
| `totoro:~/ggrpo/runs/eval_gpu1.status` | `results/training/eval_gpu1.status` | 32 | `574a4886d94a620b9a789459f78374781efcfc9d8eed38c2185d2eac6e6cff71` | - | LIVE: status of the GPU1 eval lane (LANE_START only). |
| `totoro:~/ggrpo/runs/probe_full.log` | `results/training/probe_full.log` | 9385440 | `f25aaa79a12c78293b59d6316dd514688862d25d8afe7956a076ebc7a840a01b` | `.gitignore:62:*.log` | Log of the 2-step full-FT probe (TRL completion tables, exit 0). |
| `totoro:~/ggrpo/runs/probe_full_gpu.csv` | `results/training/probe_full_gpu.csv` | 1706 | `888084499b45ad7ae8e782898a5b91b2d71a0cfcf012487832b64e924acf9d3a` | - | GPU0 memory every 15 s during the probe. |
| `totoro:~/ggrpo/runs/probe_full.status` | `results/training/probe_full.status` | 89 | `1ca33fbc88631e93541cd5a3dd8f261f2aebb011a5aee0f639772f31f2876b42` | - | Probe start/exit (exit 0). |
| `totoro:~/ggrpo/runs/smoke_hf_gpu.csv` | `results/training/smoke_hf_gpu.csv` | 30281 | `9c4439d97ac408a7c03cecd4cf32822093e5eaf54fd6eb733fbc54172d14e8bc` | - | GPU memory every 15 s during the no-vLLM LoRA/QLoRA smokes. |
| `totoro:~/ggrpo/runs/smoke_hf.status` | `results/training/smoke_hf.status` | 85 | `62bfca4b0b161bb364a5c6202bc025b00386931c9f0f030df5730c81ddd2bbd1` | - | Smoke status: LoRA exit 0; QLoRA never reported. |
| `totoro:~/ggrpo/runs/smoke_lora.log` | `results/training/smoke_lora.log` | 13151004 | `53c433de5b356d09aecf6a30dd61ed8967cd8e4b2bb5211385ee77b6748d537f` | `.gitignore:62:*.log` | Log of the 3-step LoRA smoke that preceded the no-vLLM smokes (completed). |
| `totoro:~/ggrpo/runs/smoke_lora_hf.log` | `results/training/smoke_lora_hf.log` | 9518397 | `467e0fe3ccf187e02d82410f23c084c1f1d085f5a5cb2b6f43e0c8d108450df5` | `.gitignore:62:*.log` | Log of the 2-step no-vLLM LoRA smoke (completed). |
| `totoro:~/ggrpo/runs/smoke_lora.oom_shared_with_bringup.log` | `results/training/smoke_lora.oom_shared_with_bringup.log` | 15896 | `d54712f5e96ff78bd64561cc8b49505541ee55beddcc8a5ed14607732ee2c56a` | `.gitignore:62:*.log` | LoRA smoke that OOMed while sharing the GPU with BringUp. |
| `totoro:~/ggrpo/runs/smoke_qlora.log` | `results/training/smoke_qlora.log` | 5423 | `2050fe710413fed2d1f4ea429c841d8d90620050d1eefefdb90d6ecba1785a77` | `.gitignore:62:*.log` | Log of the 2-step QLoRA smoke (NF4, nested quant); exited 1. |
| `totoro:~/ggrpo/runs/smoke_qlora_hf.log` | `results/training/smoke_qlora_hf.log` | 1403 | `e92b00151eaf51892c27e232cf3128c598513d30ad48db38bcad945d16ac96f2` | `.gitignore:62:*.log` | Log of the no-vLLM QLoRA smoke (no exit recorded in smoke_hf.status). |
| `ponyo:~/gg/ggrpo/runs/eval_ponyo.status` | `results/training/eval_ponyo.status` | 32 | `37ae3212aa7b4c130414ab46103a10ec094e0adb5553cb10b702db110e5410b4` | - | LIVE: status of the ponyo eval lane (LANE_START only). |
| `totoro:~/ggrpo/runs/pilot/full_gpu.csv` | `results/training/pilot/full_gpu.csv` | 3137 | `47fba80b0132ceaab534ec751ce82a29c91c49e2379790001757882351b7f370` | - | Pilot GPU0 memory every 60 s. |
| `totoro:~/ggrpo/runs/pilot/full.status` | `results/training/pilot/full.status` | 87 | `dc73f7a4415bf2a8b5b2cbd933d4d0fe3cfd59687387f7f0b6373f015b36d5b0` | - | Pilot start/exit (exit 1: OOM at step 19). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00001.parquet` | `results/training/pilot/completions/completions_00001.parquet` | 109538 | `fbec29a279d5b144e4eb43cac63d5dc26a0deb959f8c6bc850aec443daf64e67` | - | TRL completion log of the pilot run, logging step 1 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00002.parquet` | `results/training/pilot/completions/completions_00002.parquet` | 63039 | `ebeabbbe3d562ab0f0f8a44960a446bde1b18bbebb88567457b75129ba0f5546` | - | TRL completion log of the pilot run, logging step 2 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00003.parquet` | `results/training/pilot/completions/completions_00003.parquet` | 82462 | `ae17cd6db874b05159d9f8b26af74386c069ef73be70b5fce106b0938862f251` | - | TRL completion log of the pilot run, logging step 3 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00004.parquet` | `results/training/pilot/completions/completions_00004.parquet` | 47147 | `74b647fcbbc85b3fd71286419b4952675f2e7d6aaabf7c8afe749f97952b3380` | - | TRL completion log of the pilot run, logging step 4 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00005.parquet` | `results/training/pilot/completions/completions_00005.parquet` | 75642 | `1b98979116f714c9d45461a38165720e9b6a62289d89bb4efcb7012deb512d94` | - | TRL completion log of the pilot run, logging step 5 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00006.parquet` | `results/training/pilot/completions/completions_00006.parquet` | 76287 | `552cc7f89dabd5c40258464eee6f385fe59d14ab20b04d941bc408c4a95bb838` | - | TRL completion log of the pilot run, logging step 6 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00007.parquet` | `results/training/pilot/completions/completions_00007.parquet` | 115563 | `e0e8645cdea28effeea0a4b643b6040cb79d7535ca7e7176f80968b04112a69a` | - | TRL completion log of the pilot run, logging step 7 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00008.parquet` | `results/training/pilot/completions/completions_00008.parquet` | 72845 | `75de750e3abf3d72e8939fce55581305b25d4cff80d0279549cd98455afc90f9` | - | TRL completion log of the pilot run, logging step 8 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00009.parquet` | `results/training/pilot/completions/completions_00009.parquet` | 85447 | `c25581ba02610352fabd4f45b1f5e0e50182c1d8c49102055177c9b0232e37e6` | - | TRL completion log of the pilot run, logging step 9 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00010.parquet` | `results/training/pilot/completions/completions_00010.parquet` | 79593 | `7f307bd4e390bc094e12d256e2ec6bbac21420747c7a04c4f9b6f0ef9e975c45` | - | TRL completion log of the pilot run, logging step 10 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00011.parquet` | `results/training/pilot/completions/completions_00011.parquet` | 41434 | `f660434765deb9068add162df61668c9064fddb11070163b392b69a6335a1057` | - | TRL completion log of the pilot run, logging step 11 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00012.parquet` | `results/training/pilot/completions/completions_00012.parquet` | 71233 | `4bb1f11f694bda852f8fe4957547d7ab1be21c1efe8f0b977150fa2da7a9ee92` | - | TRL completion log of the pilot run, logging step 12 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00013.parquet` | `results/training/pilot/completions/completions_00013.parquet` | 66957 | `2551ff6ffa5b811200c164084ecc23cf615b9677f0cbada108c18cf934fcf8dc` | - | TRL completion log of the pilot run, logging step 13 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00014.parquet` | `results/training/pilot/completions/completions_00014.parquet` | 65526 | `c387bba996b0728e9fa67fb7e7e35be29aace37a50603a829a58c826baecbc78` | - | TRL completion log of the pilot run, logging step 14 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00015.parquet` | `results/training/pilot/completions/completions_00015.parquet` | 55362 | `01602b913d94b8ef43d76def7734518f00208847c566843f0ded177319c458b5` | - | TRL completion log of the pilot run, logging step 15 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00016.parquet` | `results/training/pilot/completions/completions_00016.parquet` | 56707 | `8ddaa28e83a3b2df09d0ed9273dbae9bdc423e64bef50dbff41fd00cdf28f050` | - | TRL completion log of the pilot run, logging step 16 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00017.parquet` | `results/training/pilot/completions/completions_00017.parquet` | 66470 | `a76e223fa561bb34e0f7292bdee3f93df621721167955d90fd274eee4e4d69d2` | - | TRL completion log of the pilot run, logging step 17 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00018.parquet` | `results/training/pilot/completions/completions_00018.parquet` | 96751 | `1c2d5102f13da7538700e749b620ea744ae290d437736822945215df58534cbd` | - | TRL completion log of the pilot run, logging step 18 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/completions/completions_00019.parquet` | `results/training/pilot/completions/completions_00019.parquet` | 88183 | `1db8ba831d59ae189592975418919dc9cbd2c5614bcae2b8a4bd664a0f0513a8` | - | TRL completion log of the pilot run, logging step 19 (prompts, completions, rewards). |
| `totoro:~/ggrpo/runs/pilot/train952.jsonl (copied from an identical local file, sha256 checked on totoro)` | `results/training/pilot/train952.jsonl` | 5600413 | `28bff7d2fc4ec724932c14ee24ed73348689cda679ad564009154462e3a07698` | - | Pilot training set, 952 MBPP-C rows (older data build). Identical copies (same sha256): `mac-scratch:31015bbb/train_pilot_952.jsonl`; `vm:~/ggrpo/data/train.jsonl`; `vm:~/ggrpo/runs/train.before.jsonl`. |
| `totoro:~/ggrpo/runs/aborted/save25/full.log` | `results/training/aborted/save25/full.log` | 9995402 | `d01064adfef2525ee5ec67e7f36413d5d21642a1bc6ce330cfbe7276947c99d5` | `.gitignore:62:*.log` | Log of an earlier full GRPO run that was aborted after about 3 logged steps (kept as aborted/save25). |
| `totoro:~/ggrpo/runs/aborted/save25/full_gpu.csv` | `results/training/aborted/save25/full_gpu.csv` | 664 | `dba1e649fecfc3bc87d50d8896655d25925ec8929240747e210ae02a34264951` | - | GPU0 memory of the aborted run. |
| `totoro:~/ggrpo/runs/aborted/save25/full.status` | `results/training/aborted/save25/full.status` | 43 | `f0f12c06e8873e7e9c4b678bbb5e5b37648e42e94fde1052fb2ecfbf8efc6451` | - | Aborted run status (FULL_START only). |
| `totoro:~/ggrpo/runs/aborted/save25/trackio/cs8001.db` | `results/training/aborted/save25/trackio/cs8001.db` | 2895872 | `804e4c51477da2c990022b5aed9c290572104406f70bc5e85d272021ad47cff9` | - | Trackio DB copy saved with the aborted run (static). |
| `totoro:~/ggrpo/runs/aborted/save25/full/completions/completions_00001.parquet` | `results/training/aborted/save25/full/completions/completions_00001.parquet` | 111699 | `00b49bf8c372571a4f3a61183a64b638d655014733a4eb6250015fabcfe28312` | - | TRL completion log of the aborted run, logging step 1. |
| `totoro:~/ggrpo/runs/aborted/save25/full/completions/completions_00002.parquet` | `results/training/aborted/save25/full/completions/completions_00002.parquet` | 62300 | `81a6cf4a923c43ad6bf3523b4cb0e5fe3effba6ad477c9fd1a48610007bdd1dc` | - | TRL completion log of the aborted run, logging step 2. |
| `totoro:~/ggrpo/runs/aborted/save25/full/completions/completions_00003.parquet` | `results/training/aborted/save25/full/completions/completions_00003.parquet` | 79143 | `d08ad248f39f4a3e661cc0c1a5a426cab67fd30362be3fbf16858b22bee39894` | - | TRL completion log of the aborted run, logging step 3. |
| `totoro:~/ggrpo/runs/probe_full/config.json` | `results/training/probe_full/config.json` | 1373 | `e985075d51fc707636cd5df4eb7c60b5cd0ee68478b7f88696156fa339440275` | - | Model config saved by the probe. |
| `totoro:~/ggrpo/runs/probe_full/generation_config.json` | `results/training/probe_full/generation_config.json` | 216 | `2d6e5ec1e9b136f0b3f81d3c8657ef916a5680c45da329f666540fdb5bd041cc` | - | Generation config saved by the probe (stop tokens). |
| `totoro:~/ggrpo/runs/probe_full/README.md` | `results/training/probe_full/README.md` | 1924 | `d5afdd7b294ed351a93de3430cf2e22704eab309857e49955453f5b9c5206dba` | - | TRL auto-generated model card of the probe (framework versions). |
| `totoro:~/ggrpo/runs/probe_full/checkpoint-2/trainer_state.json` | `results/training/probe_full/checkpoint-2/trainer_state.json` | 3916 | `c6b8b854800f7297514dd1488ed526dc5fd098e534ce0dab3b7d85eea88d4a2f` | - | Trainer state at step 2 (log_history: loss, reward, timing). |
| `totoro:~/ggrpo/runs/probe_full/completions/completions_00001.parquet` | `results/training/probe_full/completions/completions_00001.parquet` | 109538 | `fbec29a279d5b144e4eb43cac63d5dc26a0deb959f8c6bc850aec443daf64e67` | - | TRL completion log of the probe, step 1. |
| `totoro:~/ggrpo/runs/probe_full/completions/completions_00002.parquet` | `results/training/probe_full/completions/completions_00002.parquet` | 62084 | `439e0e967c3a433135666476616db9d9b326807df2c4ef862feee62778a01922` | - | TRL completion log of the probe, step 2. |
| `totoro:~/ggrpo/runs/stress/log` | `results/training/stress/log` | 5992 | `d8547a8a0205ad2f2af0732cc9b2f34759dbda41d372b5d20c584bce797984b7` | - | Log of the successful stress step (exit 0). |
| `totoro:~/ggrpo/runs/stress/status` | `results/training/stress/status` | 91 | `6710ee07b0c686fe2d7f264b65abfcd7b9c9c820de9a7ac4eaa3234a4b228950` | - | Stress run start/exit. |
| `totoro:~/ggrpo/runs/stress/gpu.csv` | `results/training/stress/gpu.csv` | 24335 | `9ea1848bc4f94734d4638c5e12856c05aba1e1fdb9573fed811d3252d66cb17b` | - | GPU0 memory every 1 s during the stress step. |
| `totoro:~/ggrpo/runs/stress/longest16.jsonl` | `results/training/stress/longest16.jsonl` | 398263 | `e1534614f1360ad0961e9202561efbc4dce3a40f5457ccda5dace5800c58d5f4` | - | The 16 longest training prompts used for the stress step (10,749 to 4,136 tokens). |
| `totoro:~/ggrpo/runs/stress/no_tunnel_1940/log` | `results/training/stress/no_tunnel_1940/log` | 13406 | `64d7a113e65a1276a77d7e70a4d67c7c9a14de3d17044d68360834a152dfabdc` | - | Failed stress attempt at 19:40 UTC (reward tunnel not up). |
| `totoro:~/ggrpo/runs/stress/no_tunnel_1940/status` | `results/training/stress/no_tunnel_1940/status` | 91 | `49d7bb9c268e46d1346df764e87b4894df299f90f0de04dc2d3f47815ed0de39` | - | Status of the failed attempt (exit 1). |
| `totoro:~/ggrpo/runs/stress/no_tunnel_1940/gpu.csv` | `results/training/stress/no_tunnel_1940/gpu.csv` | 39489 | `fc0a461951b1befe50c6e887b62abd2f4ba7b123074418ede4286928b1989154` | - | GPU0 memory during the failed attempt. |
| `totoro:~/ggrpo/runs/stress/out/config.json` | `results/training/stress/out/config.json` | 1373 | `e985075d51fc707636cd5df4eb7c60b5cd0ee68478b7f88696156fa339440275` | - | Model config saved by the stress run. |
| `totoro:~/ggrpo/runs/stress/out/generation_config.json` | `results/training/stress/out/generation_config.json` | 216 | `2d6e5ec1e9b136f0b3f81d3c8657ef916a5680c45da329f666540fdb5bd041cc` | - | Generation config saved by the stress run. |
| `totoro:~/ggrpo/runs/stress/train858.jsonl (copied from an identical local file, sha256 checked on totoro)` | `results/training/stress/train858.jsonl` | 5049852 | `7d2e1b2395cae9996d40a9d57f0ed8fa30698a5310ad6302ef5d04786ec093ca` | - | Current training set, 858 rows, as used by the stress test and the live full run (totoro data/train.jsonl). Identical copies (same sha256): `totoro:~/ggrpo/data/train.jsonl`; `mac:~/git/Guaranteed-Guess-feature-env/ggrpo/data/train.jsonl (gitignored)`. |
| `totoro:~/ggrpo/runs/smoke_lora/adapter_config.json` | `results/training/smoke_lora/adapter_config.json` | 1176 | `7eb59e88a692bb8a90fefe3a7790afcadbe3dd9bf4627f7550beb495f836cea4` | - | LoRA adapter config of smoke_lora. |
| `totoro:~/ggrpo/runs/smoke_lora/README.md` | `results/training/smoke_lora/README.md` | 2001 | `361b3a5728f195f99cea4e7d7b97b437d1467f18a2732d8cbfc11a203cc5d5be` | - | TRL auto-generated model card of smoke_lora. |
| `totoro:~/ggrpo/runs/smoke_lora/checkpoint-3/trainer_state.json` | `results/training/smoke_lora/checkpoint-3/trainer_state.json` | 5558 | `7451fd7051d66269e6c660c9d6c98dcfc91477e0d78a15b4a15ffd206acf235a` | - | Trainer state of smoke_lora at checkpoint-3 (log_history). |
| `totoro:~/ggrpo/runs/smoke_lora/completions/completions_00001.parquet` | `results/training/smoke_lora/completions/completions_00001.parquet` | 111699 | `00b49bf8c372571a4f3a61183a64b638d655014733a4eb6250015fabcfe28312` | - | TRL completion log of smoke_lora, step 1. |
| `totoro:~/ggrpo/runs/smoke_lora/completions/completions_00002.parquet` | `results/training/smoke_lora/completions/completions_00002.parquet` | 60721 | `3e03da411f17108dc586da1e6432380262d5facb19f7877eb94216e47f871112` | - | TRL completion log of smoke_lora, step 2. |
| `totoro:~/ggrpo/runs/smoke_lora/completions/completions_00003.parquet` | `results/training/smoke_lora/completions/completions_00003.parquet` | 86084 | `281c4e873d5fcaa5cbb6cb482a2381dde222756f83f60d129d82f583fe9b4ddc` | - | TRL completion log of smoke_lora, step 3. |
| `totoro:~/ggrpo/runs/smoke_lora_hf/adapter_config.json` | `results/training/smoke_lora_hf/adapter_config.json` | 1176 | `9b613da8da541a23651f8cce94f12bd7aef5c89f60a9c194d6224867f847e0ac` | - | LoRA adapter config of smoke_lora_hf. |
| `totoro:~/ggrpo/runs/smoke_lora_hf/README.md` | `results/training/smoke_lora_hf/README.md` | 2007 | `87293d602830f1793ac62bfdc4ca8b17dbb0ac2f2757b990d434e2ac26ef5ec3` | - | TRL auto-generated model card of smoke_lora_hf. |
| `totoro:~/ggrpo/runs/smoke_lora_hf/checkpoint-2/trainer_state.json` | `results/training/smoke_lora_hf/checkpoint-2/trainer_state.json` | 3269 | `38a7856aa0c8c9783dcd2a4a4b614825477c5ca2d478e596447c3af69da968c1` | - | Trainer state of smoke_lora_hf at checkpoint-2 (log_history). |
| `totoro:~/ggrpo/runs/smoke_lora_hf/completions/completions_00001.parquet` | `results/training/smoke_lora_hf/completions/completions_00001.parquet` | 111068 | `db40bca03a3b5d47ca3b903d2db1451872c46996c21a69bfa98dead15d054273` | - | TRL completion log of smoke_lora_hf, step 1. |
| `totoro:~/ggrpo/runs/smoke_lora_hf/completions/completions_00002.parquet` | `results/training/smoke_lora_hf/completions/completions_00002.parquet` | 61398 | `5878832b11213bb31a909591a669aa126a81070e873c0b12e014816fda20ec0a` | - | TRL completion log of smoke_lora_hf, step 2. |
| `totoro:~/ggrpo/runs/failed/smoke_hf_gpu.oom_full_prefill.csv` | `results/training/failed/smoke_hf_gpu.oom_full_prefill.csv` | 438 | `aae865498a0d55fdf0c88c01194e350cc3100bcff783905abc6f3bf26d303df4` | - | GPU memory of the smoke attempt that OOMed on full (unchunked) prefill. |
| `totoro:~/ggrpo/runs/failed/smoke_lora_hf.oom_full_prefill.log` | `results/training/failed/smoke_lora_hf.oom_full_prefill.log` | 11470 | `de49382047896b23e27b5cd43c4319c9a38a86df9ad1194593302141021091cd` | `.gitignore:62:*.log` | LoRA no-vLLM smoke OOM on full prefill. |
| `totoro:~/ggrpo/runs/failed/smoke_qlora_hf.oom_full_prefill.log` | `results/training/failed/smoke_qlora_hf.oom_full_prefill.log` | 12412 | `a838e47cee57c5802a74a7edd35ee71f4d2998b9bfc366922a11f7b44ab919ca` | `.gitignore:62:*.log` | QLoRA no-vLLM smoke OOM on full prefill. |

## Not copied (deliberate)

### Live or in progress at copy time (copy these when they finish)
| Source | Size | Why |
|---|---:|---|
| `totoro:~/ggrpo/runs/full/` | 740 KB at listing, grows with checkpoints | The live full-parameter GRPO run (instructions: skip). Its metrics are in `results/training/trackio/cs8001_snapshot.db`, its status and GPU log in `results/training/`. |
| `totoro:~/ggrpo/runs/full.log` | 30.8 MB, growing | Log of the live run. Still being written and on track to pass the 50 MB cap. |
| `totoro:~/compat_test/` | ~75 KB of scripts/JSON + 2 x 1.1 GB `*.logits.pt` | Not in the requested sources. Another session was writing it during the copy (21:11-21:12 UTC): `compare.py`, `make_inputs.py`, `probe.py`, `inputs.json`, `r450_probe.{json,err,out}`, `r517_probe.{json,err}`. Copy the small files when the probe is done. |
| `ponyo:~/gg/ggrpo/runs/paper/seed1/`, `paper_seed1.log` (137 KB) | | Paper seed 1 still running (instructions: skip). |
| `ponyo:~/gg/ggrpo/runs/relay_test/` | 5.8 GB | Not in the requested sources; a relayed checkpoint (`model.safetensors` 6.2 GB) being written during the copy. Small files: `config.json`, `generation_config.json`, `trainer_state.json`. |
| `cluster:~/cs8001-bringup/base/` | 207 KB | Slurm BringUp jobs still writing (instructions: skip). |
| `mac-scratch:9da06990/` (whole session) | | This session was live during the copy. Its scripts and small outputs were copied as a snapshot (`ops/mac-scratch/9da06990/`, `results/data-survey/`); re-copy when it finishes. |

### Over the 50 MB per-file cap
| Source | Size | Why |
|---|---:|---|
| `totoro:~/ggrpo/runs/pilot/full.log` | 59,831,405 B | Pilot GRPO log (OOM at step 19). Over the cap. The pilot's metrics are in the Trackio snapshot and `results/training/pilot/completions/`. |

### Model weights, checkpoints, optimizer state, tokenizer copies (totoro)
| Source | Size | Why |
|---|---:|---|
| `runs/probe_full/model.safetensors`, `runs/probe_full/checkpoint-2/model.safetensors` | 6.17 GB each | Model weights. |
| `runs/probe_full/checkpoint-2/optimizer.pt` | 12.35 GB | Optimizer state. |
| `runs/stress/out/model.safetensors` | 6.17 GB | Model weights. |
| `runs/smoke_lora{,_hf}/adapter_model.safetensors` and their checkpoint copies | 4.67 MB each | LoRA weights (LoRA/QLoRA path dropped Oct 22). |
| `runs/smoke_lora{,_hf}/checkpoint-*/optimizer.pt` | 9.57 MB each | Optimizer state. |
| `rng_state.pth`, `scheduler.pt`, `training_args.bin` in every run/checkpoint dir | < 15 KB each | Binary pickles, not results (the run configs are in `ops/totoro/`). |
| `tokenizer.json` (11.4 MB), `tokenizer_config.json`, `chat_template.jinja` in every run dir | | Copies of the base model's tokenizer. |
| `runs/trackio/cs8001.lock`, `runs/aborted/save25/trackio/cs8001.lock` | 0 B | Lock files. |

### Hugging Face dataset copies
| Source | Size | Why |
|---|---:|---|
| `totoro:~/cs8001-bringup/gg-bench-bringup-O2.parquet`, `cluster:~/cs8001-bringup/gg-bench-bringup-O2.parquet`, `vm:~/scratch/data/bringup_O2.parquet`, `mac-scratch:643399c8/{totoro,cluster}/gg-bench-bringup-O2.parquet`, `mac-scratch:643399c8/vm/data/bringup_O2.parquet`, `mac-scratch:9da06990/bench/bringup_O2.parquet` | 998,544 B | All sha256 `c3acfd94...`: HF gg-bench-bringup-O2 at snapshot ae6623a3 (see `DATA_SOURCE.txt`). |
| `ponyo:~/cs8001-step0/v450/gg-bench-armv8-O2.parquet`, `vm:~/scratch/data/official_O2.parquet`, `mac-scratch:643399c8/ponyo/gg-bench-armv8-O2.parquet`, `mac-scratch:643399c8/vm/data/official_O2.parquet` | 212,020 B | Identical to git's `ggrpo/data/raw/gg-bench-armv8-O2.parquet`. |
| `vm:~/scratch/data/bringup_O0.parquet` (955 KB), `vm:~/scratch/data/official_O0.parquet` (182 KB), and their copies in `mac-scratch:643399c8/vm/data/` | | O0 counterparts of the dataset copies above (named and used the same way by `bringup_refs.py`); not checked byte-for-byte against HF. |
| `vm:~/scratch/data/asmbench_source.parquet`, `mac-scratch:643399c8/vm/data/asmbench_source.parquet`, `mac-scratch:643399c8/dl_asmbench_source/0.parquet` | 68,795 B | Identical to git's `ggrpo/data/raw/asmbench_source.parquet`. |
| `mac-scratch:31015bbb/sft_shard/train-00000.parquet` | 65 MB | One GG SFT shard (gg-armv8-O2-instruct). |
| `mac-scratch:9da06990/`: `codenet_dl/` (1.5 GB), `dl_exebench_main/` (5.8 GB), `sft/` (2.3 GB), `stackc/` (~860 MB), `hf/` (80 MB), `dl_dcbench/` (98 MB), `dl_exebench_clang/` (59 MB), `transcoder_dl/` (37 MB), `mbppplus_dl/`, `probs/`, `bench/`, `sft_paths.parquet` (10 MB) | ~10.8 GB | Dataset downloads for the survey. |

### Already in git (identical blob in some commit)
- `vm:~/ggrpo/` source, docs, configs, `data/raw/*`, `data/mbpp_armv8_O2.parquet`; `vm:~/rl_simple/{vm_build.py, vm_check.py, tasks_O2.jsonl, ref_empty_O2.jsonl, reward_copy/}`.
- `totoro:~/ggrpo/step0/*`, `ponyo:~/gg/ggrpo/step0/*`; `ponyo:~/cs8001-step0/{gen.py, preds_full_raw.jsonl, tasks_O2.jsonl}`; `ponyo:~/gg/eval/eval_huamaneval.py` (upstream file).
- `mac:~/git/Guaranteed-Guess/rl_simple/`: `export_*.py`, `make_mbpp_parquet.py`, `vm_*.py`, `p0_official_variant.c`, `mismatch_findings.md`, `step0_toolchain.txt`, `tasks_O2.jsonl`, `step0_ref_empty_O2.jsonl`, `step0_preds_raw_ponyo_O2.jsonl`, `mbpp_armv8_O2_ventura.parquet`.
- `mac-scratch:643399c8/`: the in-git files of the `ggrpo/` project snapshot, `xcheck/ponyo_generate.py`, `xcheck/totoro_config.toml`, `vm/data/bringup_sample_results_O0.json`, `old_mbpp.parquet`; `mac-scratch-main:d3bd2ab3/ponyo/preds_full_raw.jsonl`.

### Deployments identical to this worktree
- `totoro:~/ggrpo/`: every code, config and doc file is byte-identical to this worktree's current working tree, including its uncommitted changes (`config.toml`, `grpo.yaml`, `pyproject.toml`, `uv.lock`, `src/ggrpo/{data,generate,reward,settings}.py`, `README.md`, `docs/code_plan.md`, `tests/test_reward.py`). Its `step0/` directory (removed from the worktree) is in git history. Committing the worktree pins the code of the live run.
- `ponyo:~/gg/ggrpo/`: `src/` and `config.toml` match the worktree; `grpo.yaml`, `pyproject.toml`, `uv.lock`, `tests/test_extract.py` are older git versions; `README.md` and `docs/code_plan.md` are older uncommitted drafts.

### Regenerable data and superseded intermediates
| Source | Size | Why |
|---|---:|---|
| `vm:~/ggrpo/data/eval.jsonl`, `vm:~/ggrpo/runs/eval.before.jsonl`, `totoro:~/ggrpo/data/eval.jsonl` | 1.0 MB | sha256 `3de05c6a...`, same as this worktree's gitignored `ggrpo/data/eval.jsonl` (built by `ggrpo.data`). |
| `totoro:~/ggrpo/data/dev.jsonl` | 551 KB | Same as this worktree's untracked `ggrpo/data/dev.jsonl`. |
| `ponyo:~/gg/ggrpo/data/{train,eval,dev}.jsonl` | 5.6 MB, 1.0 MB, 551 KB | Built by `ggrpo.data` on ponyo; not hashed (sizes match the pilot train set, the eval set and `dev.jsonl`). |
| `vm:~/ggrpo/data/eval_llm4decompile.jsonl` | 1.0 MB | Older eval-set variant (LLM4Decompile drivers). |
| `mac-scratch:643399c8/ggrpo/data/train.jsonl` (sha `68ae7a2a...`), `ggrpo/data/eval.jsonl` (`24629e5d...`) | 5.6 MB, 1.0 MB | Intermediate data builds inside an old project snapshot. |
| `mac-scratch:643399c8/ggrpo/` other files (`pyproject.toml`, `config.toml`, `docs/code_plan.md`, `src/ggrpo/{settings,data}.py`, `runs/ggrpo.jsonl` 281 B, `.cache/drivers/*.o`) | | Old snapshot of the project, superseded by this worktree. |
| `mac:~/git/Guaranteed-Guess/rl_simple/mbpp_O2_vm.jsonl` (also `vm:~/rl_simple/`) | 5.8 MB | VM build output that `make_mbpp_parquet.py` turned into git's `mbpp_armv8_O2_ventura.parquet`. |
| `mac:~/git/Guaranteed-Guess/rl_simple/mbpp_jobs_O2.jsonl` (also `vm:~/rl_simple/`) | 1.3 MB | VM build input, made by `export_mbpp_jobs.py` from the HF data. |
| `mac:~/git/Guaranteed-Guess/rl_simple/mbpp_armv8_clang21.parquet` | 3.6 MB | Superseded clang-21 build; identical to `rl/artifacts/data/mbpp_armv8.parquet`. |
| `mac-scratch:643399c8/totoro_out/bringup_O2_beam8.jsonl` (41 rows), `totoro_out/modern_partial.jsonl`, `ponyo_out/runB_partial.jsonl` | | Byte-prefixes of files that were copied (`results/bringup/totoro/bringup_O2_beam8.jsonl`, `results/humaneval/modern/beam_search.jsonl`, `results/humaneval/v450/full_greedy_beam.jsonl`). |
| `mac-scratch-main:d3bd2ab3/cmp/out/`, `bb/bbout/` | | Unpacked forms of the copied `out.tgz` and `bbout.tgz`. |
| `mac-scratch-main:d3bd2ab3/cmp/src/`, `cmp/src.tgz`, `cmp/off/`, `bb/off/` | | Extracted from the HF datasets by `prep.py`. |

### Third-party code, downloads, environments
- `mac-scratch:643399c8/vllm-bnb-plugin/` (third-party git clone; QLoRA path dropped), `mac-scratch:31015bbb/wheels/` (vendored transformers 5.17 source, 69 MB), `mac-scratch:33ca3c48/{grpo_trainer.py, vllm_generation.py}` (TRL source copies).
- `mac-scratch:9da06990/`: `bringup_src/repo/` (BringUp-Bench clone), `exebench_src/`, `inc/` (third-party headers), `emb/` (embedding cache), `ct/ebct_*` (443 MB of build trees), `pdfs/` and paper text dumps (`gg.txt`, `gg_emnlp.txt`, `neucomback.txt`, `xcodeeval.txt`, `stack_v1.txt`, `codenet/paper.txt`); `mac-scratch:31015bbb/gg_emnlp.txt` (text of the GG paper).
- `mac-scratch-main:d3bd2ab3/`: `bb/bb.tgz` (BringUp-Bench source, 1 MB), `tartdl/tart.tar.gz` (Tart download, 23 MB), `t/` (trivial toolchain smoke with binaries).
- `vm:~/bringup-bench`, `bringup-bench-clean`, `bb_dbg`, `bb_fb` (BringUp-Bench working copies used by the scorers), `vm:~/brew_coreutils.log`.
- `ponyo:~/cs8001-step0/get-pip.py`; every `.venv*`, `venv/`, `__pycache__/`, `.pytest_cache/`, cache directory.

### Transcripts, build products, empty or trivial files
- `mac-scratch:c34c21bc/{thread.md, thread2.md}`, `mac-scratch:d37ed3f6/{thread.txt, new_643399c8-a9fa-407f-8537-f428823102e9.txt, new_33ca3c48-fab5-43c5-ab02-a7a58c388e55.txt}`: exports of Claude session transcripts (the originals are under `~/.claude/projects/`).
- `mac-scratch:d37ed3f6/midpoint_v2/midpoint_report.log` (LaTeX build log).
- `mac-scratch:643399c8/{empty.jsonl, one.jsonl}` (test stubs), `ponyo:~/cs8001-step0/v450/stop_after_greedy.log` and `vm:~/rl_simple/mbpp_build.log` (empty).
- Every Mac scratchpad `tasks/` directory (tool output logs).

### Outside the requested sources, noticed while looking
- Main checkout `~/git/Guaranteed-Guess` (branch `qlora-grpo-study`, not on GitHub): untracked `rl/analysis/compiler_drift.py`, `rl/data/build_bringup.py`, `rl/eval/e2e_mac.py`; gitignored `rl/artifacts/` (110 MB: ExeBench/HumanEval/MBPP parquets, `compiler_drift.json`, three small logs); uncommitted edits to `rl/data/build_exebench.py`, `rl/reward/harness.py`, `rl/reward/server.py`. These are persistent local files, not under `/private/tmp`, so they were left in place.

