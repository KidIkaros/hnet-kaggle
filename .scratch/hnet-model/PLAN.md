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

## Verified state (from `tools/facts.sh`, 2026-11-01)

- Kaggle: 2 × Tesla T4, 15.6 GB each; one GPU 10,616 ms/step, two 5,414 ms/step → **1.96×**.
- Quota: **19.65 h remaining** of 30, resets 2026-11-07 (6 days).
- Corpus: 2.35 GB train / 47.9 MB val, 492,363 / 10,048 docs, 589 duplicates dropped before
  the split, source FineWeb-Edu (ODC-By-1.0), split seed 20261031.
- Last run: 39,049 steps, 18.56 M params, 2.97 h, nominal 1.5e16 → **realised 1.92e16**
  (1.28×), realised chunk 3.99 bytes/chunk.
- Result: held-out **bpb 1.508** (72,450 bytes), training split 1.539 — no gap, so no leak and
  no memorisation; the model is under-trained rather than stuck.

## Remaining work

| # | step | cost | status |
|---|---|---|---|
| 1 | Dual-GPU support in the training driver | code only | **needed before any run** |
| 2 | Scaling run at 3e16 nominal (~3.8e16 realised) | ~3.1 h dual GPU | awaiting approval |
| 3 | Second seed at 1.5e16 for error bars | ~1.5 h dual GPU | optional, after 2 |
| 4 | Re-score checkpoints, assemble write-up | CPU, free | after 2 |

Step 2 is the only one that changes what we know: it turns one point into a scaling pair.
Step 3 only tightens a number we already have.

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

## Deviation log

*(newest first; each entry states what changed, why, and who approved it)*

- 2026-11-01 — Plan created from the four decisions agreed in conversation. No deviations yet.
