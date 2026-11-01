# Plan — H-Net comparable rung

**Status:** active. This file is the reference. Deviations are recorded in the log at the
bottom before they are acted on, not after.

---

## Objective (agreed 2026-11-01)

> A reproducible held-out bpb at a stated FLOPs budget, with the mechanism findings and
> caveats stated honestly.

Not: beating a published number, not reproducing the 10× claim, not a sampler. Those may
follow, but they are not the bar. Success is a number we can defend and re-derive.

## Operating rules

1. **Facts before plans.** Run `tools/facts.sh` in the conversation before stating any plan
   that spends quota. Facts are the agent's job; nothing is left silently assumed.
2. **This file is the reference.** Adaptation is expected and welcome — a plan can go to shit.
   But a deviation is written here, with its reason, *before* it is acted on.
3. **No GPU run without explicit approval.** No fixed ceiling; every run is brought to the
   owner with its estimated cost first.
4. **Dual GPU by default.** 1.96× measured. Single-GPU only by explicit choice.
5. **No kernel optimisation** until the write-up exists. The pipeline is fast enough to
   finish the experiment; more speed is not the constraint on the science.
6. **Local machine is for short correctness checks only** (seconds, one config). Any sweep or
   multi-minute benchmark runs on Kaggle or not at all.
7. **One metric, one sample.** Every bpb figure compared in this project comes from the same
   protocol: the driver's `evaluate()`, 40 batches x 4 documents x 2048 bytes, seed 0, about
   327k bytes scored. Numbers from other samples are labelled as such and never compared
   against each other. The 1.508 above came from a 72k-byte ad-hoc sample and is therefore a
   starting point, not the baseline.

## Verified state (from `tools/facts.sh`, 2026-11-01)

- Kaggle: 2 × Tesla T4, 15.6 GB each; one GPU 10,616 ms/step, two 5,414 ms/step → **1.96×**.
- Quota: **19.65 h remaining** of 30, resets 2026-11-07 (6 days).
- Corpus: 2.35 GB train / 47.9 MB val, 492,363 / 10,048 docs, 589 duplicates dropped before
  the split, source FineWeb-Edu (ODC-By-1.0), split seed 20261031.
- Last run: 39,049 steps, 18.56 M params, 2.97 h, nominal 1.5e16 → **realised 1.92e16**
  (1.28×), realised chunk 3.99 bytes/chunk.
- Result: held-out **bpb 1.508** (72,450 bytes), training split 1.539 — no gap, so no leak and
  no memorisation; the model is under-trained rather than stuck.

## Step 0 — the baseline, on the canonical sample (free, CPU)

Re-score `ckpt-final.pt` from run 1 with the driver's exact evaluation protocol (rule 7).
About five minutes of CPU, no quota. Without this, run 2's number is compared against a
different sample and the comparison means nothing.

**Output:** `baseline_bpb`. Every branch below is judged against it.

## Step 1 — dual-GPU support in the driver (code only, no quota)

`DataParallel` over both T4s, measured at 1.96x. The model's forward returns an `aux` dict of
floats that `DataParallel` cannot gather, so the training step must return tensors only.
Verified by the QC gate plus a three-step smoke before any long run.

## Step 2 — the scaling run (3e16 nominal, ~3.8e16 realised, ~3.1 h dual GPU)

One variable changes from run 1: the budget. Same corpus, geometry, optimiser, schedule shape.

### Pre-registered criteria

The BLT scaling law (bpb proportional to C^-0.206, fitted about 300x above our scale) predicts
roughly a 13% drop for a doubling of compute. Treat that as a direction, not a promise.

| outcome | criterion vs `baseline_bpb` | reading |
|---|---|---|
| **strong** | at least 13% lower | the law holds at our scale; the architecture converts compute into bpb |
| **success** | 7-13% lower | compute still pays, but the curve is shallower than predicted |
| **flat** | within 5% | saturated at 18.6M parameters on this corpus |
| **worse** | more than 5% higher | instability or an invalid comparison, not a result |

### Branch A — success or strong

Compute is the lever, so the remaining question is whether the effect is real or seed noise.

1. **Second seed at 3e16** (~3.1 h). Without it the write-up has one seed per point and cannot
   separate a trend from a lucky draw.
2. **Re-score both seeds on the canonical sample**; report mean and spread, not a single number.
3. **Write up.** Quota then: ~16.6 h spent, ~3 h held in reserve.

Optional only if 1 and 2 are complete and quota allows: a third point at 1e16 (~1 h) to fit
three points rather than two.

### Branch B — flat

Compute is not the lever at this size. Two candidate causes, and choosing between them is a
decision rather than a guess:

- **Capacity-bound.** The main network sees only about a quarter of the bytes (46.9 vs 112
  MFLOPs/byte against a plain transformer). The lever is then the target chunk ratio: N=4
  gives the main net more work per byte. One run, ~3.1 h.
- **Data-bound.** 320 MB seen is 0.14 epochs. If more compute on the same data does nothing,
  the corpus is the constraint, and the lever is more distinct documents — a data problem, not
  a GPU problem.

**The discriminating evidence is already free**: whether train bpb kept falling while val
flattened. Train still falling means capacity or overfitting; train flat too means data or
optimisation. Decide from that before spending.

### Branch C — worse

Not a result. Stop spending and diagnose, in this order:

1. **Is the comparison valid?** Same sample, same protocol, same realised-chunk accounting.
2. **Is the run unstable?** Ratio loss, chunk-length drift and gradient norm are all in the
   history and cost nothing to read.
3. **Is the schedule wrong for a longer run?** Warmup and the cosine decay are fractions of
   `max_steps`, so doubling the budget changes both. This is the most likely cause and it is
   checkable from run 1's history without another run.

Only after 1-3 is there a hypothesis worth a run.

## Kill criteria — when we stop experimenting and write up

Any one of these ends experimentation:

- two consecutive branches show no improvement over `baseline_bpb`;
- fewer than 6 h of quota remain (reserve for one corrective run and the write-up);
- the 2026-11-07 reset arrives.

The write-up then reports what we have, null results included. The objective allows this
explicitly, and a recorded null is worth more than an unrecorded guess.

## Known caveats to carry into the write-up

- Realised FLOPs exceed nominal by 1.28× because the router settles below the target chunk
  length. Published figures must use the realised number.
- 1.508 bpb rests on 72,450 bytes; the full val split is 47.9 MB and should be scored once the
  run is final.
- MFU is ~2.1% of the T4's 16-bit peak. Fast enough to finish the experiment; not a tuned
  kernel, and the write-up should say so rather than imply otherwise.
- The H-Net's per-byte cost is 2.4× lower than a plain transformer of the same parameter count
  (46.9 vs 112 MFLOPs/byte) because the main network runs at chunk resolution. That is the
  architecture's structural advantage, and it is why FLOPs-per-byte figures must be stated
  rather than assumed from parameter count.

## Quota ledger

| item | hours | running total |
|---|---|---|
| spent to date, including tonight's probes | 10.35 | 10.35 |
| step 2 (3e16 on both GPUs) | ~3.1 | ~13.5 |
| branch A1 (second seed) | ~3.1 | ~16.6 |
| reserve: one corrective run and overrun | ~3.0 | ~19.6 |

## Deviation log

*(newest first; each entry states what changed, why, and who approved it)*

- 2026-11-01 — Plan created from the four decisions agreed in conversation. No deviations yet.
