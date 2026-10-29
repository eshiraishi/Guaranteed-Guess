# Evidence: 06-recipes (deep-research output, Oct 24, 2026)

Raw synthesis of the 3-vote verified claims. Fields: confidence | claim, SRC (sources), EV (evidence and vote). Sections after the findings: caveats, open questions, refuted claims, sources, stats.

SUMMARY: Our current RL settings (LR 1e-6, G=8, beta 0, token-level loss, no entropy bonus) match the minimal recipes that published work uses at 1.5-8B: JustRL, Lite PPO, and a GRPO run on Qwen2.5-Coder-3B. The verified gaps are scale and a few cheap switches. We ran 106 steps where published runs use 400-4,380, and 128 completions per step where they use 768-2,048. The switches are FP32 LM-head logits, clip-higher (eps_high 0.28) and no explicit overlong penalty. For the redone SFT, LR 2e-5 for 2 epochs with loss on the output only matches both GG and LLM4Decompile. Batch size (192 vs about 2,048) is the main SFT setting nobody has tested for assembly. Pick the checkpoint that starts RL by held-out validation loss and pass@large-k, not by SFT pass@1. Switch to RL once validation loss plateaus, not earlier and not after heavy overfitting. Keep all unique pairs, because data scale sets the ceiling after RL. Sweeps under a small budget cannot trust early rankings, because three sources show them inverting. Fit ScaleRL sigmoids only after about one epoch of RL prompts, and treat a difference in fitted asymptote below about 0.02 (about 3 of 164 HumanEval-C problems) as noise. No verified evidence covers base-model choice or prior assembly work that uses RL or execution feedback. Nearly all the RL evidence is math, not code.

- high | Keep the core RL recipe: LR 1e-6, G=8, KL beta 0, no entropy regularization, clipped PPO-style loss, no critic. These values recur across independent minimal recipes that match or beat heavier ones. JustRL used one fixed recipe on two 1.5B models with no per-model tuning: constant LR 1e-6, N=8, no KL, no entropy bonus, clip range [0.8, 1.28], temperature 1.0. Lite PPO beat GRPO and DAPO on Qwen3-4B/8B-Base with LR 1e-6, 128 prompts x 8, init_kl_coef 0, token-level loss, group-mean/batch-std advantages. The one code data point, Qwen2.5-Coder-3B GRPO, also used LR 1e-6, g=8, beta 0. One schedule change follows by inference, not from a verified source: JustRL keeps the LR constant through 4k+ steps. Our linear decay to 0 over 106 steps gave an average LR of about 5e-7. Use a constant LR (or decay over a much longer run) in the next arm. Evidence grade B: several primary sources, mostly math, one code run at 3B.
  SRC: https://arxiv.org/abs/2512.16649 https://arxiv.org/html/2508.08221v2 https://arxiv.org/abs/2606.18487
  EV: JustRL Table 2: LR 1e-6 constant, rollout N 8, no KL loss, no entropy regularization, clip [0.8,1.28], T 1.0; 'The same hyperparameters work for both, without per-model tuning'. Lite PPO (ROLL): LR 1e-6, 128x8 = 1,024, init_kl_coef 0.0, T 0.99, 500 steps, 8k response; beats GRPO/DAPO on non-aligned Qwen3-4B/8B. 2606.18487: GRPO LR 1e-6, g=8, beta 0, eps_high 0.28, T 1.0, 400 steps on Qwen2.5-Coder-3B.

- medium | Our 106 steps are far below every published run, so the next RL arm should run at least 400-500 steps and plan for 1-2k if the curve is still rising. Published step counts: 400 (Qwen2.5-Coder-3B code GRPO), 500 (Lite PPO, 4B/8B), and 3,440-4,380 (JustRL at 1.5B, 32 A800 for about 15 days each). JustRL improved smoothly and monotonically with no entropy collapse or plateau, and entropy stayed around 1.0-1.6. ScaleRL starts its scaling fits only after about 1k steps (about one epoch). A 106-step run is therefore in the early phase, where rankings are known to be unreliable (see the sweep-methodology finding). Evidence grade B/C: the numbers are descriptive, and no source verifies how results scale with step count for 1.5B code RL.
  SRC: https://arxiv.org/abs/2512.16649 https://arxiv.org/html/2508.08221v2 https://arxiv.org/abs/2606.18487 https://arxiv.org/html/2510.13786
  EV: JustRL: 4,380 steps (DeepSeek-R1-Distill-Qwen-1.5B) and 3,440 steps (OpenMath-Nemotron-1.5B), 'smooth, monotonic improvement over 4,000+ steps without the collapses or plateaus'. These are single runs with no seeds, the curve is noisy, and the domain is math. ScaleRL App. A.6/A.7: the first phase is about 1k steps, and fits start after about 1.5k GPU-hours, roughly one epoch.

- medium | Raise completions per step. Keep G=8 and add prompts: about 64 prompts x 8 = 512 completions as the next arm, and 128 x 8 = 1,024 if compute allows. Published batches: ScaleRL 768 (48x16), Lite PPO 1,024 (128x8), and JustRL train batch 256 with PPO mini-batch 64. In veRL that most likely means 256 prompts x 8 = 2,048 completions per rollout, taken in 4 optimizer mini-batches; the verifier did not confirm the unit. In ScaleRL, larger batches raised the fitted asymptote and avoided late stagnation, while smaller batches looked better early. The evidence is thin: 512 and 768 were indistinguishable (A=0.605 vs 0.610), and only 2,048 prompts separated (A=0.645), probably from a single run. Our 128 is 4x below the smallest batch tested. Under a fixed small budget, 128 per step may actually be the better choice. Evidence grade C for the asymptote effect.
  SRC: https://arxiv.org/html/2510.13786 https://arxiv.org/html/2508.08221v2 https://arxiv.org/abs/2512.16649
  EV: ScaleRL Sec. 5: 'Larger batches reliably improve the asymptote and avoid the downstream stagnation we observe in smaller-batch runs'. Table 1 (App. A.13): bs512 A=0.605, 768 A=0.610, 2048 A=0.645, against a stated +/-0.02 error margin on A. 8B, math (Polaris-53K), thousands of GPU-hours.

- medium | Turn on FP32 LM-head logits in both the trainer and the rollout engine before anything else, and add CISPO (or GSPO) as one loss arm. In ScaleRL at 8B, the FP32 fix alone raised the asymptotic pass rate from 0.52 to 0.61. The large gain appeared with GRPO/DAPO-style losses like ours, and it mattered little under the full ScaleRL recipe (CISPO). CISPO and GSPO both clearly beat DAPO on asymptote; the paper gives no per-loss number. Loss aggregation, advantage normalization, curriculum and the off-policy algorithm mostly change efficiency, not the ceiling. Three of them gave small asymptote gains and are cheap to add: prompt-average loss aggregation, zero-variance filtering, and No-Positive-Resampling, which permanently drops prompts with pass rate >= 0.9 from later epochs. Evidence grade C: split vote, one source, math at 8B.
  SRC: https://arxiv.org/html/2510.13786
  EV: ScaleRL Sec. 3.2 / Fig. 5b: 'the precision fix dramatically improves the asymptotic performance A from 0.52 to 0.61'. 'both GSPO and CISPO substantially outperform DAPO'. Abstract: details 'primarily modulate compute efficiency without materially shifting the asymptote'. In the leave-one-out study, FP32 makes little difference with ScaleRL's own loss. Advantage normalization gave only a marginal gain.

- high | Use clip-higher: eps_low 0.2, eps_high 0.28. Monitor policy entropy every step. Mechanism: clipping biases entropy independently of the reward. Clip-low (on negative-advantage tokens) raises entropy, and clip-high (on positive-advantage tokens) lowers it. This is shown with first-order theory on a tabular softmax policy and with verl experiments on 1-8B models. Raising eps_high therefore weakens the term that drives entropy down. Clip-higher slowed entropy collapse and helped on aligned (instruct) Qwen3 models, but had negligible or harmful effects on base models, where the clip rate is only about 0.003. Our policy starts from an instruct or SFT'd checkpoint, so the aligned-model result applies. On the best value: 4B improved steadily up to 0.32, 8B was best at 0.28, and JustRL (1.28) and the 3B code run (0.28) used 0.28 stably. Evidence grade B for the mechanism and the 0.28 default, all math except the 3B code run.
  SRC: https://arxiv.org/pdf/2509.26114 https://arxiv.org/html/2508.08221v2 https://arxiv.org/abs/2512.16649 https://arxiv.org/abs/2606.18487
  EV: 2509.26114: 'clip-low increases entropy, while clip-high decreases it' (theory plus verl runs on Qwen2.5, Llama3, OLMo-2). 2508.08221 Sec. 4.2: aligned models gain from a higher upper clip; 4B best at 0.32, 8B best at 0.28; base models are not helped. Skywork-OR1 is noted by a verifier as finding the best eps_high task-dependent.

- low | Fallback arm, only if entropy collapses under eps_high 0.28: turn clip-high off (eps_high = infinity) and set eps_low around 0.15. On GSM8K RLVR (Qwen2.5-3B-Instruct, Llama3-8B-Instruct, LR 5e-7, batch 512, G=8, 4,096-token response), this held entropy steady at about 0.2. eps_low 0.1 exploded entropy on both models, and 0.2 collapsed it toward 0.05. The working window is narrow and depends on the setup, so treat 0.15 as the centre of a small sweep, not a fixed value. Evidence grade C/D: split vote, single runs, one preprint, math.
  SRC: https://arxiv.org/pdf/2509.26114
  EV: Sec. 3.2 / Fig. 5 and App. Fig. 8(b): '(eps_high = inf, eps_low = 0.15) achieves a balance, preventing both entropy collapse and entropy explosion'; 'for this particular setup'.

- low | Do not add an explicit overlong length penalty. Keep plain truncation at the 4,096-token cap. In JustRL ablations at 1.5B over 3,000+ steps, a DAPO-style overlong penalty on the last 4k tokens lowered the AIME24 plateau from about 55% to 50%. It also collapsed entropy to 0.5-0.6, against 1.2-1.4 in the base recipe. Adding a robust verifier on top lowered the plateau to about 45%. Evidence grade C: single runs, a 30-problem benchmark, and the verifier effect is confounded with the penalty. DAPO itself reported that overlong shaping helped.
  SRC: https://arxiv.org/abs/2512.16649
  EV: JustRL Sec. 4.4: 'adding overlong penalty plateaus at 50% AIME 2024 (vs 55% baseline) ... the explicit penalty collapses exploration, driving entropy down to 0.5-0.6'.

- medium | Sampling temperature: keep 1.0 as the default, which JustRL (1.0), the 3B code run (1.0) and Lite PPO (0.99) all use. Then check it against the AceReason heuristic: set temperature so that temperature-adjusted entropy stays around 0.3. At 7B, of 0.6, 0.85 and 1.0, 0.85 scored best on all four benchmarks, but only narrowly over 1.0 (+0.1 AIME25, +0.5 LCBv5). Action: measure entropy on our SFT'd checkpoint at T=1.0. If it is well above 0.3, run a T=0.85 arm. Evidence grade C: one source, math stage-2 RL on a distilled 7B, single runs.
  SRC: https://arxiv.org/abs/2506.13284 https://arxiv.org/abs/2512.16649 https://arxiv.org/abs/2606.18487
  EV: AceReason-Nemotron 1.1 Sec. 4.5.2: T 0.6 starts at entropy about 0.15 (over-exploitation); T 0.85 starts at about 0.26 and gives AIME24/25/LCBv5/v6 of 67.6/56.8/52.1/47.1; T 1.0 gives 65.3/56.7/51.6/45.7.

- high | Do not pick the SFT checkpoint for RL by its SFT pass@1 or beam score. Pick it by generalization loss on a fixed held-out set plus pass@large-k with tests (for example pass@64). Over hundreds of models up to 12B, those two signals predicted post-RL results better than pre-RL scores, improving R^2 and Spearman by up to 0.5 (about 2x, best case). Independently, the minimum SFT validation loss followed the post-RL ceiling on both Qwen2.5-7B (S1K 0.51 -> 77.3 ... SFT889K 0.37 -> 84.0) and Llama3.2-3B (0.71 -> 25.0 ... 0.44 -> 70.4), all on one shared held-out set. On Qwen2.5-Coder-3B, pre-RL pass@1 rose with SFT depth while post-RL pass@10 fell. Pre-RL policy entropy is not a usable substitute: that claim (Spearman +0.69) was refuted 0-3. Evidence grade B: three independent primary sources, math plus one code run.
  SRC: https://arxiv.org/abs/2510.01624 https://arxiv.org/abs/2512.11470 https://arxiv.org/abs/2606.18487
  EV: 2510.01624 abstract: 'improving R^2 coefficient and Spearman's rank correlation coefficient by up to 0.5 (2x)'; 'high SFT scores are not reliably predictive of subsequent RL gains'. 2512.11470 Table 5 and Table 1 (v2). 2606.18487: pre-RL pass@1 0.151 -> 0.187, peak GRPO pass@10 0.806 -> 0.481.

- medium | How much SFT before RL: use sequential SFT-then-RL. Start RL once SFT validation loss has plateaued (the stable regime), not early and not after heavy overfitting. Sequential SFT-then-RL beat unified methods (LUFFY, SRFT, UPT, Prefix-RFT) with 78.1 for SFT-then-DAPO_d on Qwen2.5-7B. Switching early (69.8 exaFLOPs) capped the ceiling at 80.7, against 84.0 at the stable regime. Mild overfitting was best only for one dataset. Deeper SFT can hurt RL when early-RL success is low. Qwen2.5-Coder-3B with LoRA SFT at 1.0 -> 5.8 epochs saw peak pass@10 fall from 0.806 to 0.481. DeepSeek-Coder-6.7B showed only compression (0.861-0.891). RL on an SFT'd model can even end below RL on the base model (Qwen3-4B-base). RL narrows gaps between SFT starts (AIME24 v5 vs v7: 6.6 -> 1.6 points), but large quality gaps remain, and stronger SFT still ends ahead. Action: run RL from the plateau checkpoint, and add one arm from a shallower checkpoint if the early success rate on our RL prompts is low. Evidence grade B/C: sources point in partly different directions and are mostly math.
  SRC: https://arxiv.org/abs/2512.11470 https://arxiv.org/abs/2606.18487 https://arxiv.org/abs/2510.01624 https://arxiv.org/abs/2506.13284
  EV: 2512.11470 abstract: 'Transitioning to RL at the Stable or Mild Overfitting Regime of SFT maximizes the final ceiling'; 'Sequential SFT-then-RL pipeline as the superior standard'. Not compute-matched, and pure-RL is better early. 2606.18487 (ICML 2026 DL4C workshop) finds inversion only below an early-success threshold. 2506.13284 Sec. 4.5.1 and App. B.

- medium | SFT data: keep the full set and add unique pairs rather than repeated samples. Data scale set the post-RL ceiling (S1K 77.3, 102K 79.7, 220K 82.6, 440K 82.8, 889K 84.0), with almost no gain from 220K to 440K. Scaling unique prompts gave larger gains than scaling responses per prompt (log2 coefficients 4.83 vs 2.64, R^2 0.989), although responses also helped. For GG this argues for keeping all 1.31M pairs and adding new unique sources, for example O2 or RISC-V pairs, instead of sampling several translations per source. In long-CoT SFT at 7B, accuracy rose from epoch 1 to epoch 5 and flattened at 5-6. That is weak evidence for testing 1/2/3 epochs on GG, not a recommended value: GG's outputs are direct translations with no CoT. Evidence grade C for transfer: math/long-CoT at 3-7B, and data scale is confounded with SFT compute.
  SRC: https://arxiv.org/abs/2512.11470 https://arxiv.org/abs/2506.13284
  EV: 2512.11470 Table 5: 'Data Scale determines the primary post-training potential'. 2506.13284: 'scaling the number of prompts yields more significant gains'; 'consistent performance gains from the first to the fifth epoch'.

- medium | SFT hyperparameters: keep LR 2e-5 (cosine, warmup 0.1), 2 epochs and loss on the output only. Sweep batch size (192 vs about 1,024-2,048) and add a 1e-5 LR arm. LLM4Decompile-End, the closest verified assembly recipe, fully fine-tuned DeepSeek-Coder 1.3B and 6.7B on asm-to-source pairs. It used batch 2,048, LR 2e-5, 2 epochs (about 15B tokens), with loss only on the output. The 1.3B model took 12 days on 8xA100. A small-LLM SFT guide found larger effective batches with lower LR gave better final results: 4k beat 128 on 4 of 5 benchmarks. 2e-5 was the lowest and best Granite LR tested, while Mistral was best at 1e-6, so the best LR depends on the model family. Large batches win only at the end and need more samples, so they can look worse early. The claim that LLM4Decompile's epoch ablation supports 2 epochs was refuted 0-3, so no verified evidence sets the epoch count for assembly. Evidence grade B/C.
  SRC: https://arxiv.org/abs/2403.05286 https://arxiv.org/abs/2412.13337
  EV: 2403.05286 Sec. 4.1.1: 'batch size=2048 and learning rate=2e-5 and train the models for 2 epochs (15B tokens) ... the loss is calculated only for the output sequence'. 2412.13337: batch sizes 128/3,840/7,680; '2e-5 works well for Granite models, while 1e-6 performs best for Mistral'. Single seed, with schedule confounds.

- medium | Sweep method under a small budget: do not rank arms on early reward. Three sources show early rankings inverting: smaller RL batches look better early and fall behind (ScaleRL), SFT settings with lower final scores dominate early before plateauing (2412.13337), and higher pre-RL pass@1 can predict worse post-RL results (2606.18487). Fit ScaleRL's sigmoid R - R0 = (A - R0)/(1 + (C_mid/C)^B) per arm. ScaleRL found it more stable than a power law, started fitting after about 1.5k GPU-hours (about 1 epoch, about 1k steps at 8B), and extrapolated a fit to 8k GPU-hours out to 16k. Its 100k-hour prediction used data up to 50k, not just early compute. The stated error is about +/-0.02 on A (+/-0.015 over 3 seeds), and B is hard to estimate. Practical rules: (1) run each screening arm for at least one epoch over the RL prompt pool before fitting; (2) treat differences in A below 0.02 (about 3/164 on HumanEval-C) as noise; (3) if using successive halving or ASHA, put the first rung after about one epoch and rank by fitted A, not raw reward; (4) one seed for screening, at least 3 seeds for the final comparison. Most cited studies report single seeds.
  SRC: https://arxiv.org/html/2510.13786 https://arxiv.org/abs/2412.13337 https://arxiv.org/abs/2606.18487
  EV: ScaleRL Eq. 1, App. A.7: 'The 1.5k GPU-hour threshold is a heuristic chosen empirically'; 'scaling curves extrapolated from only the initial stages of training closely match the final observed performance'. ICLR 2026. No independent replication of the extrapolation was found.

== caveats
"Domain transfer is the main weakness. All RL claims except one come from math RLVR (GSM8K, DAPO-Math, DeepMath, Polaris, AIME) at 1.5-8B. The one code RL source (2606.18487) is a workshop paper: LoRA SFT on Qwen2.5-Coder-3B, scored on a frozen 40-problem HumanEval+ subset. The only assembly source (LLM4Decompile) covers SFT only. No verified claim covers 1.5B code RL with a 4k-token cap, and none covers x86-to-ARM/RISC-V translation. Most results are single runs with no seeds. ScaleRL is the exception (3 seeds, +/-0.015).\n\nTwo claims passed on split votes: FP32/loss choice (2-1) and clip-high off with eps_low 0.15 (2-1). For claims 9-18, several verifiers ran out of web-search budget, so the search for contradicting work is incomplete there.\n\nConfidence grading: 'high' means several independent primary sources agree and all votes were unanimous. A finding resting on one primary source that was checked against the paper text with a 3-0 vote got 'medium', not the rubric's 'low', because fidelity to the paper was confirmed. Transfer to our setting is graded separately (B/C) inside each finding.\n\nFour claims were refuted and must not be used:\n- token-level vs sequence-level loss preference by model type (1-2);\n- symmetric clipping lowering entropy under random rewards on 1-1.5B instruct models (0-3);\n- pre-RL entropy predicting the GRPO outcome, rho +0.69 (0-3);\n- LLM4Decompile's epoch ablation supporting 2 epochs (0-3).\nSo our token-level loss choice for an instruct or SFT'd start is still unverified either way.\n\nResearch questions 1 (base-model choice, tokenizer efficiency on assembly, long context) and 5 (prior assembly or decompilation work with RL or execution feedback, and test-time scaling) have essentially no verified claims. About 30 votes in this topic failed when the session limit hit, so a missing claim is not evidence against it.\n\nTime sensitivity: the sources run from Dec 2024 to Jun 2026. 2510.01624 is an unreviewed preprint. 2512.11470 lists ACL-26 in v2, and its Table 5 numbers appear only in v2. ScaleRL and JustRL are ICLR 2026.\n\nScale gap: ScaleRL's batch and FP32 results come from thousands to 100k GPU-hours at 8B. Our 128-completion batch is 4x below its smallest tested batch, so 'too small' is an extrapolation, and at a fixed tiny budget small batches may win. JustRL's batch of 256 most likely counts prompts (veRL convention), giving 2,048 completions, but no verifier confirmed the unit. The inference that our linear decay gave an average LR of about 5e-7 follows from the schedule, not from a source."
== openQuestions
[
 "Which 1-2B base model should the redone SFT start from? Candidates: Qwen2.5-Coder-1.5B base vs instruct, Qwen3-1.7B, DeepSeek-Coder-1.3B, OpenCoder-1.5B and newer 2026 models. No claim on pretraining exposure to assembly, tokenizer efficiency on assembly text or long-context ability survived verification. Tokens per instruction on our own corpus can be measured locally at low cost.",
 "Do ScaleRL's batch-size and FP32-logits effects hold at 1.5B code RL with a 4,096-token cap and only a few hundred steps? A compute-matched comparison (16x8 vs 64x8 at equal total completions, FP32 on/off) would answer this directly for our setting.",
 "Is our RL prompt pool in the low early-success regime where deeper SFT inverts RL results (2606.18487)? HumanEval-C beam-8 was already 72/164 before RL, but harder dev or BringUp prompts may sit below the threshold. Early-RL success rate per SFT checkpoint is the measurement that decides how much SFT to do.",
 "Which prior work on assembly translation or decompilation with RL or execution feedback is the state of the art? Candidates: SuperCoder, D-LiFT, SK2Decompile, ReF Decompile, CRT and others. Do any of them report best-of-n with tests or self-repair from assembler/linker errors? These claims are still waiting for verification."
]
== refuted
[
 {
  "claim": "Which loss aggregation is better depends on the model type. Token-level loss beats sequence-level loss on Base models. On aligned or instruction-tuned models the gain is limited, and response-level aggregation is preferred. This bears directly on our token-level loss choice for an SFT'd or Instruct Qwen2.5-Coder-1.5B.",
  "vote": "1-2",
  "source": "https://arxiv.org/html/2508.08221v2"
 },
 {
  "claim": "With standard symmetric clipping (eps_low = eps_high = 0.2; 0.1 for OLMo-2), the clip-high effect dominates, so entropy falls even under purely random rewards. This holds for Qwen2.5-1.5B-Instruct, Llama3.2-1B-Instruct and OLMo-2-0425-1B-Instruct. It implies that apparent gains from spurious or random rewards come mainly from entropy minimization.",
  "vote": "0-3",
  "source": "https://arxiv.org/pdf/2509.26114"
 },
 {
  "claim": "Choosing the SFT checkpoint with the highest pass@1 as the GRPO starting point is not reliable. Pre-RL policy entropy predicts the GRPO outcome better: it is positively correlated with the outcome (Spearman rho = +0.69).",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2606.18487"
 },
 {
  "claim": "In the epoch ablation (Appendix D), 1 epoch is a strong baseline, 2 epochs give the best results, and more epochs overfit. This supports a 2-epoch SFT budget for assembly-related seq2seq fine-tuning.",
  "vote": "0-3",
  "source": "https://arxiv.org/abs/2403.05286"
 }
]
== unverified
[]
== sources
[
 {
  "url": "https://arxiv.org/html/2510.13786",
  "quality": "primary",
  "angle": "RLVR recipe and hyperparameter ablations (1-7B code)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2508.08221v2",
  "quality": "primary",
  "angle": "RLVR recipe and hyperparameter ablations (1-7B code)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2609.39321v1",
  "quality": "primary",
  "angle": "RLVR recipe and hyperparameter ablations (1-7B code)",
  "claimCount": 5
 },
 {
  "url": "https://together.ai/blog/deepcoder",
  "quality": "blog",
  "angle": "RLVR recipe and hyperparameter ablations (1-7B code)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2512.16649",
  "quality": "primary",
  "angle": "RLVR recipe and hyperparameter ablations (1-7B code)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/pdf/2509.26114",
  "quality": "primary",
  "angle": "RLVR recipe and hyperparameter ablations (1-7B code)",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2606.18487",
  "quality": "primary",
  "angle": "SFT recipe and SFT-to-RL interplay for code translation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2510.01624",
  "quality": "primary",
  "angle": "SFT recipe and SFT-to-RL interplay for code translation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2506.13284",
  "quality": "primary",
  "angle": "SFT recipe and SFT-to-RL interplay for code translation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2512.11470",
  "quality": "primary",
  "angle": "SFT recipe and SFT-to-RL interplay for code translation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2403.05286",
  "quality": "primary",
  "angle": "SFT recipe and SFT-to-RL interplay for code translation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2412.13337",
  "quality": "primary",
  "angle": "SFT recipe and SFT-to-RL interplay for code translation",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2506.14606",
  "quality": "primary",
  "angle": "Base model choice and assembly tokenization",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2511.03825",
  "quality": "primary",
  "angle": "Base model choice and assembly tokenization",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2604.06095",
  "quality": "primary",
  "angle": "Base model choice and assembly tokenization",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2505.11480",
  "quality": "primary",
  "angle": "State of the art: assembly, decompilation and binary translation with LLMs and execution feedback",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2509.22114",
  "quality": "primary",
  "angle": "State of the art: assembly, decompilation and binary translation with LLMs and execution feedback",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2606.16162",
  "quality": "primary",
  "angle": "State of the art: assembly, decompilation and binary translation with LLMs and execution feedback",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2511.01183v1",
  "quality": "primary",
  "angle": "State of the art: assembly, decompilation and binary translation with LLMs and execution feedback",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2510.13786",
  "quality": "primary",
  "angle": "Sweep methodology under compute limits",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/html/2606.03073v1",
  "quality": "primary",
  "angle": "Sweep methodology under compute limits",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2504.07086",
  "quality": "primary",
  "angle": "Sweep methodology under compute limits",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2508.13144",
  "quality": "primary",
  "angle": "Sweep methodology under compute limits",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2504.13837",
  "quality": "primary",
  "angle": "Contrarian: limits of RLVR on small models",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2506.10947",
  "quality": "primary",
  "angle": "Contrarian: limits of RLVR on small models",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2505.24864",
  "quality": "primary",
  "angle": "Contrarian: limits of RLVR on small models",
  "claimCount": 5
 },
 {
  "url": "https://arxiv.org/abs/2507.14843",
  "quality": "primary",
  "angle": "Contrarian: limits of RLVR on small models",
  "claimCount": 5
 }
]
== stats
{
 "angles": 6,
 "sourcesFetched": 27,
 "claimsExtracted": 135,
 "claimsVerified": 25,
 "confirmed": 21,
 "killed": 4,
 "unverified": 0,
 "afterSynthesis": 13,
 "urlDupes": 1,
 "budgetDropped": 8,
 "agentCalls": 110
}
