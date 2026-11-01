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

Baseline is measured: **`baseline_bpb` = 1.4900 over 272,093 bytes** (run 1, 1.5e16 nominal,
39,049 steps, scored on the canonical 40-batch sample). Thresholds are now concrete:

| outcome | criterion | absolute bpb | reading |
|---|---|---|---|
| **strong** | at least 13% lower | **<= 1.2963** | the law holds at our scale; the architecture converts compute into bpb |
| **success** | 7-13% lower | 1.2963 - 1.3857 | compute still pays, but the curve is shallower than predicted |
| **flat** | within 5% | 1.4155 - 1.5645 | saturated at 18.6M parameters on this corpus |
| **worse** | more than 5% higher | **> 1.5645** | instability or an invalid comparison, not a result |

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

## Result of step 2 (2026-11-01, kernel mauricew/hnet-eng-3e16)

Both seeds completed, no guard fired, 4.65 h each on its own T4 (two seeds for the quota price
of one). Realised FLOPs came in at 1.00x nominal -- the router settled at 5.97/6.01 bytes per
chunk against the 6.0 target, unlike run 1's 3.99.

| run | realised FLOPs | held-out bpb | bytes/param |
|---|---|---|---|
| run 1 (1.5e16) | 1.92e16 | 1.4900 | 22.1 |
| run 2 (3e16) | 3.01e16 | **1.4173** (1.4170, 1.4177; spread 0.0007) | 34.6 |

**Verdict against the pre-registered bands: FLAT.** 4.88% lower, and the flat band starts at
1.4155 -- the result misses "success" by 0.0018 bpb. Seed spread is 0.0007, so the 4.88% is far
above noise; it is a real but small gain.

**Measured scaling exponent b = 0.111**, against the 0.206 the BLT law reports from fits about
300x larger. At b = 0.111 a 1.57x increase in compute predicts 7.4%; we observed 4.88%.

**Why:** run 2 sits at 34.6 bytes/param. On the ~20 bytes/param rule this model is under-sized
for the data it has seen -- parameter-poor, not data-poor. That rule is a transformer
heuristic and this architecture spends 2.53 FLOPs per parameter per byte rather than 6, so the
optimum is not exactly 20; solving at fixed C = 3e16 gives roughly 24M parameters. Treat that
as a reasoned estimate, not a law.

Consequence for the plan: this is Branch B, and the free evidence (train bpb tracking val, no
overfitting, val still descending at the end) points to size rather than data as the lever.

## Knowns vs guesses (the frontier, after step 2)

**Measured.** Two bpb points on the canonical sample: 1.4900 at 1.92e16 realised FLOPs, and
1.4173 at 3.01e16 (two seeds, spread 0.0007). Exponent along that line: 0.111, against 0.206
reported at ~300x larger scale. Throughput: 1.8e12 FLOP/s per T4 = 2.8% of the 65 TFLOP/s
peak; the scan is 87.5% of the step; the reference implementation feeds 2.1 MB per step
against our 8 KB. The router hit its 6.0 target, so realised FLOPs equal nominal here. Train
bpb tracked val -- no overfitting -- and val was still descending when the run ended. Quota:
14.88 h left.

**Guessed.**
1. The ~20 bytes/param optimum transfers to this architecture. It is a transformer heuristic;
   this model spends 2.53 FLOPs per parameter per byte rather than 6, because the main network
   runs at chunk resolution. There is no evidence either way in our data.
2. Therefore "24M is the right size" -- derived entirely from 1. Two data points cannot locate
   an optimum.
3. A larger batch would fill the GPU now. The evidence for batch not mattering was measured at
   the *untrained* router (chunk 1.9 bytes), where memory, not the scan, was binding. At the
   trained router (5.97) that measurement may not apply.

**The risk this section exists to name:** spending ~4.5 h on a size point whose value rests
on guesses 1 and 2, when guess 3 may make the same point cost ~1 h -- or when the write-up is
the better use of the quota. The cheapest discriminating measurement is the batch/throughput
probe (about 0.15 h); the size question is not answerable from two points at any price.

## Decisions after step 2 (2026-11-01, owner)

1. **Quota discipline:** start with the throughput probe, and do NOT plan to spend the
   remaining 14.9 h -- the reset is 5 days away and the prior pattern of burning quota on
   guesses is what the owner wants stopped.
2. **Evidence standard for size:** literature first. The owner wants the byte-level scaling
   question researched against primary sources before any further size experiment, so the
   24M figure stops being a guess and becomes either supported or replaced.
3. **Pipeline work is welcome:** efficiency improvements and adaptations are always in scope;
   the goal is fewer mistakes, not fewer changes.

## Probe overrun (2026-11-01, recorded as it happens)

The warm-started batch probe was budgeted at ~0.15 h and burned ~2.1 h before being stopped:
no live logs, no completion, 14x over. Probable cause is torch.compile re-tracing for every
batch shape; the re-run drops inductor compile, keeps the fused scan, and carries a hard
720 s in-probe budget so it cannot run away again. Cost of the lesson: ~2.2 h of quota.

## Literature on the size question (docs/RESEARCH_SCALING.md, 12 sources)

For a ~3e16 FLOPs byte-level run the sources bracket the compute-optimal size at **10M-25M
parameters**: ~15.8M by Chinchilla arithmetic in token units (an extrapolation -- their table
stops at 1.84e19 FLOPs, ~600x above this budget); ~24.3M if the 20 bytes/param rule transfers
to bytes unchanged; ~10.3M using BLT's ~111 bytes/param anchor with this project's 2.53
FLOPs/param/byte. The spread is dominated by the assumed bytes/param ratio, and **no published
work fits a scaling law on byte-level models**, so the literature alone cannot choose between
20 and 111 bytes/param -- a 2.3x range in optimal size.

**Our own two points now bear on that choice.** At 34.6 bytes/param, a further 57% more data
bought only 4.9% lower bpb. If the optimum were near 111 bytes/param, that first leg of the
journey should have paid much more. So the measurement leans toward the low end of the
literature range: this 18.56M model is roughly right-sized or modestly under, and the flat
result reflects the architecture's own slow exponent (0.111) more than a size error.

Consequence: a 24M run would sit at ~20 bytes/param -- at the heuristic optimum -- but the
expected gain over 1.4173 is a few percent, not a step change. The price of that run depends
entirely on the throughput probe, which is why the probe comes first.

## Deviation log

*(newest first; each entry states what changed, why, and who approved it)*

- 2026-11-01 — **DataParallel abandoned for step 1.** The dual-GPU smoke (0.08 h) failed in
  replica 1: a parameter stayed on cuda:0 while the activation was on cuda:1, then gradient
  checkpointing tripped a SavedTensorHooks assertion. `torch.compile` inside `DataParallel` is
  the incompatibility -- compiled blocks hold references to the original module's parameters,
  which `DataParallel` never replicates. Read from the traceback, not guessed.
  **Replacement, which suits the purpose better:** two *independent* single-GPU jobs in
  parallel, one per T4. Quota is billed per session, so two seeds in one 5.9 h session cost the
  same as one seed alone -- the second seed is free in quota terms. No distributed training, no
  replication, and it keeps the compiled path already proven at 274 ms/step on a T4. Awaiting
  approval.
- 2026-11-01 — Plan created from the four decisions agreed in conversation.
