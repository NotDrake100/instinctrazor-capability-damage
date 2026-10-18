# mathfix Phase 3: self-voiced facts. Pre-registered BEFORE any Phase-3 code or run.

Written 2026-10-18 07:55 IST by the sleep-shield routine driver. Edit only by adding dated "post-hoc" sections.

## What we learned (Phases 1-2)
- Phase 1 (0.5B, arm b, 3 seeds): H1 format overwrite supported. Night-7 GSM8K failures are ~96% `no_steps`, mean output
  ~6 tokens vs base ~218, prefill "Let's solve this step by step." restores +28.3 pts. The math is still there; the model
  stopped talking.
- Phase 2 H1 (0.5B and 1.5B): mixing 20% long-form text restores GSM8K, but GSM8K-train replay (r) is no better than
  digit-free Dolly filler (c): 0.5B r-c 4.7 <= std 7.4, 1.5B r-c 0.7 <= std 4.2. Any long text works, so the damage is
  generic output-format/length, not math. Mixing costs recall: |r-b| 16.0 pts (0.5B), 20.1 pts (1.5B). Both FAIL.
- 1.5B night-7 (3 seeds): base GSM8K strict 73.5; b strict 5.4, tok 4; r 53.5, tok 91, recall 68.0 vs b 88.1; c 52.8.

## Claim (one falsifiable sentence)
On Qwen2.5-0.5B-Instruct, 7 nights x 100 fictional facts (same fact/probe hashes as Phase 2), training every fact with
the base model's OWN full-sentence answer (written by the base model with the fact given in context, kept only if it
contains the gold answer) keeps night-7 GSM8K strict retention >= 50% of base-correct items, beats a token-matched
fixed-template control by more than the combined seed std, and keeps recall within 5 pts of terse arm b, with ZERO
extra (non-fact) training data.

## Arms (same LoRA r16/a32 all 7 modules, lr 2e-4, 120 steps/night, batch 16, 3 seeds 0/1/2, all 16 slots facts)
- `b`: terse targets "X." (identical to Phase 2 b).
- `v` (fix): self-voiced target per fact. Generated ONCE by the frozen base model, greedy, max 64 new tokens, prompt =
  system "Fact: <train_q> <answer>." + user <train_q>. Keep if ans_match(gold) and the generation ended with EOS. Failed
  facts get one retry with system "Fact: ... Use this fact in your answer." Remaining failures fall back to
  "The answer is <a>." Fallback rate is logged. Same training questions as b.
- `t` (control): for each fact, a fixed hand-written template that is NOT model-generated: "The answer is <a>." followed
  by fixed neutral sentences from a constant list, truncated so its supervised token count EXACTLY equals that fact's
  `v` target token count (asserted; the answer sits in the first sentence). Isolates "the model's own voice" from
  "just a longer answer".

## Measurements (each night, base-correct items, as Phase 2)
GSM8K strict (v2) + lenient, mean tokens, % steps, failure categories, mult, ARC, recall on reworded questions,
adapter-off recall at nights 1 and 7. Recall generation max_new_tokens = 64 for ALL arms (was 24, too short for
full-sentence answers). Also log recall@24 for comparability.

## PASS (night 7, mean over 3 seeds; ALL must hold)
- P1: v GSM8K strict retention >= 50% of base-correct items.
- P2: v - t > sqrt(sd_v^2 + sd_t^2) (sample std, ddof=1).
- P3: |recall v - recall b| <= 5 pts.
- Validity: adapter-off recall <= base recall + 5 in every run; v fallback rate <= 10% of facts.

## Kill rule
Any of P1/P2/P3 fails, or validity fails, at 0.5B => the self-voiced idea is DEAD as stated. Record numbers, do not
rerun. Report separately: if v and t both pass P1/P3 but P2 fails, the finding is "answer length alone prevents format
overwrite" (still useful, not the claim).

## If 0.5B PASSES
Run the same config on Qwen2.5-1.5B-Instruct (memory settings that finished the Phase 2 1.5B run) as confirmation, then
tell Archit with real numbers.

## Prior art to cite (scoop risk)
Self-Distillation Fine-Tuning (Yang et al. 2024, "Self-Distillation Bridges Distribution Gap in LM Fine-Tuning") and
"I Learn Better If You Speak My Language" (Ren et al. 2024) already rewrite targets with the model itself to reduce
forgetting in single-task fine-tuning. Our contribution if it passes: nightly sequential fact injection, a measured
mechanism (format overwrite, Phase 1), and the replay-vs-filler-vs-self-voice controls.

## Compute
Vast 16 GB GPU (~$0.09/hr). Estimate ~2-2.5 h for 0.5B (~$0.25). Vast credit ~$0.98 at 07:26 IST 18 Oct.
