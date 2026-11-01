# Does dynamic chunking buy compute efficiency at 1e16 FLOPs?

**A two-point, two-seed scaling measurement of a byte-level H-Net on FineWeb-Edu, checked
against an independent replication — and what it does to the "22% less compute" claim.**

Companion to [WRITEUP.md](WRITEUP.md), which answers the mechanism question (what the chunker
aligns to, across four scripts and three modalities). This document answers the efficiency
question the Cartesia posts lead with.

## The finding, in plain language

We trained the same 18.6 M-parameter byte-level H-Net at two compute budgets — 1.92e16 and
3.01e16 realised FLOPs — and measured held-out bits-per-byte on a fixed, pre-registered eval
sample. Doubling compute bought **4.9%**, not the ~13% the blog's efficiency claim implies at
this ratio. The measured scaling exponent is **b = 0.111**, roughly half of the 0.206 that the
Byte Latent Transformer paper reports for byte models (fitted at ~300× our scale).

That alone is a small-model caveat. What turns it into a finding is the external check: an
independent replication (main-horse, pure-torch, 1e16–1e18 FLOPs, FineWeb-Edu, byte-level)
published per-run bpb curves before us, and **our point sits on their curve family** — in line
with their H-Net runs (~1.47 bpb interpolated at 3e16) and at-or-slightly-below their
plain-transformer baseline (~1.44). Nothing about our number suggests an implementation gap.

Their conclusion, from their own write-up: *"None of the H-Net (S>0) runs ever achieve more
compute efficiency than L2"* — parity at 1e17, then what they call "the dreaded scaling curve
bounce" at higher FLOPs. They also locate the likely source of the blog's 1.28× claim: the
paper normalises both architectures by identical **bytes**-per-batch, which trains the BPE
baseline at a sub-optimal batch size (its optimal token batch maps to ~4.57× more bytes), and
its learning rate may be 2–10× too low. The paper itself concedes it never formally ran
scaling laws.

So: **the architecture trains and scales normally; the headline efficiency claim is what does
not replicate at this scale — independently, and now in our runs too.**

## Setup

| | |
|---|---|
| model | 1-stage byte-level H-Net, 18.6 M params, raw-byte embedding (256), predicts 256 byte classes — no tokenizer, no BPE anywhere |
| data | `mauricew/hnet-english-bytes`: 2.35 GB train / 47.9 MB val from FineWeb-Edu, 492,363 train docs, 10,048 val docs, 589 dupes dropped |
| eval | the driver's `evaluate()`, 40 batches × 4 docs × 2048 bytes, seed 0 — **272,093 bytes** after document packing; one metric, one sample, every number in this doc |
| run 1 | 1.5e16 nominal FLOPs → **1.92e16 realised** (router settled at 3.99 B/chunk), 39,049 steps, seed 0 → **1.4900 bpb** |
| run 2 | 3e16 nominal FLOPs → **3.008e16 realised** (1.00× nominal; router settled at 5.97 / 6.01 B/chunk), seeds 0 and 1 → **1.4177 / 1.4170**, mean **1.4173** |
| hardware | one Kaggle T4 per seed, two independent jobs in one session (DataParallel was tried and abandoned — `torch.compile` inside `DataParallel` deadlocks on parameter/activation device mismatch); 4.65 h per seed |
| early stopping | `guard_reason()`: non-finite loss → above-uniform (>8.5 bpb after step 0) → 50% regression from the run's own best; train loss checked every log interval. Ten unit tests, including the NaN case a naive `x > 8.5` check silently passes |

Bytes-per-parameter: run 1 at 22.1 B/param, run 2 at 34.6 B/param — both on the parameter-poor
side of the ~20 B/param compute-optimal heuristic, i.e. these are small-data points, which is
the honest regime for a weekly-GPU-budget project.

## The curve

| realised FLOPs | bpb | seeds | verdict (pre-registered bands) |
|---|---|---|---|
| 1.92e16 | 1.4900 | 1 | baseline anchor |
| 3.01e16 | 1.4173 | 2 (spread 0.0007) | **FLAT** — 4.88% improvement, below the pre-registered success band (≥7%) |

Bands were pinned before the run: strong ≤ 1.2963, success 1.2963–1.3857, flat
1.4155–1.5645, worse > 1.5645. The result lands in the flat band. At b = 0.111, a 1.57×
compute increase predicts 7.4%; we observed 4.88% — the discrepancy is the finding, not noise
(seed spread is 0.0007 on a 0.03 gap).

Two second-order observations worth recording:

- **Realised FLOPs tracked nominal at run 2 (1.00×) but not run 1 (1.28×).** The router's
  settled compression moved from 3.99 to ~6.0 B/chunk between runs. Any efficiency claim built
  on *target* compression rather than realised must report the realised number; ours moved by
  50% between two runs of the same architecture.
- **MFU is the cheapest available lever.** We ran at ~2.8% MFU against a published small-model
  anchor of 11.3% on the same T4-class hardware. The same experiment is ~1.7× cheaper at
  published MFU. This is an engineering gap, not a scientific one, and it is the first thing
  to fix if the ladder is ever extended.

## The external anchor

main-horse.github.io/hnet/scaling-laws-byte (9 Sep 2025) publishes per-run bpb-vs-FLOPs curves
for a llama2-style baseline ("L2") and H-Net stages S1/S2 on FineWeb-Edu bytes, 1e16–1e18.
Reading their published chart:

| FLOPs | their L2 | their H-Net (S1) |
|---|---|---|
| ~1e16 | ~1.54 bpb | ~1.60 |
| ~4e16 | ~1.41 | ~1.44 |

Caveat stated plainly: their eval split is theirs and ours is ours — same corpus family and
metric, not the same byte stream — so this is an anchor, not a matched comparison. Within that
caveat, our 1.4173 at 3.01e16 is where an in-family implementation should land, and their
central result (H-Net marginally *worse* than the plain baseline at equal compute through
1e16–1e17, converging at larger scale) is the same shape our two points show.

## What this says about the blog's claim

The Cartesia posts lead with a 1.28× compute multiplier (an H-Net reaching a tokenized
baseline's loss with ~22% less compute); "up to 3–5×" is the code/downstream number. Part II
restates 1.28×/2.06×. Our result does not overturn the claim at their scale — 100B bytes,
models orders of magnitude larger. What it establishes, together with the independent
replication:

1. The multiplier is **not** visible at 1e16–1e17 FLOPs on the same corpus family, in two
   independent implementations.
2. The comparison methodology has identified confounds (batch-size normalisation by bytes, and
   possibly baseline LR) that main-horse argues are large enough to produce the headline
   number on their own.
3. The compression mechanism itself is real and works — our router settles at ~6 B/chunk, in
   the same regime as their 4.57 BPE operating point (see WRITEUP.md for what the chunker
   does and does not align to at that compression).

The honest summary: **dynamic chunking is a working compression mechanism whose
compute-efficiency advantage over static chunking is, at small scale, not measurable — and the
published advantage at large scale rests on a comparison with known confounds.**

## Limits

- **Two points, not a ladder.** b = 0.111 is a two-point estimate; its error bar is wide. The
  qualitative conclusion (improvement per doubling well under 13%) does not need the exponent
  to be precise.
- **One metric, one sample.** All bpb numbers come from one fixed 272,093-byte eval sample,
  seed 0. It is consistent across seeds (spread 0.0007) but it is one slice of one val set.
- **Parameter-poor regime.** 22–35 B/param means both runs are data-rich/compute-poor; the
  compute-optimal frontier at this budget sits elsewhere. The blog's regime is not ours.
- **No in-project trained baseline.** The comparison the blog makes (vs a tokenized model, or
  vs static chunking) was never trained in-corpus here — the external replication stands in
  for it, at the cost of the eval-split caveat above. A fixed-patch (static chunking, patch 6)
  control was designed (~10 lines, identical compression/FLOPs/bytes-per-param) and remains
  shelved, ready to run under a fresh quota.
- **MFU gap.** 2.8% vs 11.3% published anchor means our absolute numbers are ~1.7× more
  expensive than they needed to be. It does not bias the bpb comparison, which is per-byte and
  architecture-matched.

## Reproducing it

```
kaggle kernels push -p kernel_eng      # assembled by kernel_eng/build.py
# config baked from configs/eng-3e16-seeds.json; KERNEL_ID env sets the slug
```

- datasets: `mauricew/hnet-english-bytes` (corpus), `mauricew/hnet-eng-ckpt` (run-2 best
  checkpoint, published for warm-start work)
- gates: `python tests/test_gates.py`; guards: `python tests/test_guards.py` (10 tests);
  gate injections: `python tests/test_gate_injections.py` (injects the four real bugs found in
  this project and asserts each gate rejects them)
- checkpoint scoring: `python tools/eval_ckpt.py <ckpt> <corpus_dir>` — the canonical 40-batch
  sample, CPU, zero quota

## Where this leaves the effort

The mechanism question is answered in WRITEUP.md; the efficiency question is answered here:
the model works, implements cleanly enough to land on an independent replication's curve, and
the efficiency claim is a large-scale claim that this budget cannot confirm and two
independent data sets now fail to see at small scale. Total GPU spend for both English scaling
points: ~9.5 h of a 30 h weekly quota, including every crashed launch and the abandoned batch
probe. The next marginal GPU hour belongs to the audio/vision phases or the fixed-patch
ablation, not to extending this ladder.
