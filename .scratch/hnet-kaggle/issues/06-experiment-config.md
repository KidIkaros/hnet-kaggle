# Experiment config

Type: grilling
Status: resolved
Blocked by: 02, 04

## Question

Pin the exact configuration of the Q1 run, so that nobody tunes it mid-flight to rescue a
result.

Decide, with the constraint that ratio dynamics need ≥5000 steps before a run means
anything:

- **Arms.** Q1 needs only `hnet`. Does it also need `byte` and `bpe` as reference points on
  the same data, or does that belong to Q2?
- **Size.** ~8M params was the v1 guess. Fix `d_model`, layer counts for encoder/main/decoder,
  and whether the outer networks are a torch SSM or the `mbdfa` fused scan (the fusion may
  not pay at this size — that is fog, not a decision yet).
- **Compression target N.** Sweep {3, 6, 12} or fix one? The paper matches BPE's 4.57; the
  practitioner guidance is that N is a per-domain hyperparameter.
- **Steps and batch.** Enough for ≥5000 optimizer steps and at least one full pass over
  50 MB/language; state the resulting GPU-hours per run.
- **Seeds.** At least 2, because the published seed spread at comparable scale was ~0.005
  bpb and we will be reading differences of that size.
- **LR.** Per-stage sweep at one width (μP does not transfer at S≥1 — established in
  `docs/ANATOMY.md`), and how many points.
- **Precision.** fp16 autocast + fp32 master weights + GradScaler, gated against an fp32
  reference (no bf16 on sm_75).

Resolve with a config block that can be pasted into a kernel, plus the GPU-hour estimate.

## Answer

**Q1 runs at ~0.84 M params, not 8 M. The 8 M target costs 15.5 h per run and the budget is
9.72 h; the arithmetic is below and the deviation is declared rather than quietly taken.**

**Frozen configuration** (grid written to `configs/q1.json`)

| item | value |
|---|---|
| `d_model` | 128 |
| `n_enc` / `n_main` / `n_dec` | 2 / 2 / 2 |
| `d_ff` (SwiGLU on main blocks) | 512 |
| `d_state` / `d_conv` / `scan_chunk` | 16 / 4 / 64 |
| params | **0.84 M** |
| `target_ratio` N | **6.0** |
| `alpha` | 0.03 |
| steps | **5000, floor, no early stop** |
| batch × seq | 16 × 256 = 4096 bytes/step |
| LR | 1e-3, warmup 200, cosine decay |
| seeds | **2 (0, 1)** |
| arms | **`hnet` only** |
| optimizer | AdamW, wd 0.1, clip 1.0 |

**Sizing arithmetic.** Effective throughput is calibrated from `mbdfa`'s measured 5 kB/s at
1.56 M params on a T4 in fp32 — that is 0.45 MFLOP/byte over 5 kB/s ≈ **2.25 GFLOP/s
effective**, which is the real ceiling because the per-channel scan materialises `(B,L,E,N)`
and the model is memory-bound, not FLOP-bound.

| config | params | MFLOP/byte | s/step | h per 5000-step run |
|---|---|---|---|---|
| 8 M target (d256/e4/m6/d4/ff1024) | 8.31 M | 6.10 | 11.1 | **15.5** |
| d160/e2/m3/d2/ff640 | 1.70 M | 1.32 | 2.4 | 3.3 |
| **chosen** d128/e2/m2/d2/ff512 | **0.84 M** | 0.77 | 1.4 | **2.0** |

2 seeds × 2.0 h = **4.0 h**, inside the 9.72 h remaining this week with room for one retry.
The 8 M arm alone would be 15.5 h and would not finish before the reset, let alone twice.

**The pass-over-corpus condition in this ticket is not reachable and is being superseded.**
One pass over 50 MB × 6 languages is 300 MB; at 0.84 M params that is 28 h. 5000 steps consumes
20.5 MB, i.e. **~7% of one epoch**. The design is therefore **steps-bound, not pass-bound** —
and that is the right trade, because the ratio dynamics need 5000 steps to stabilise while a
partial epoch only costs corpus coverage. Same 4096 B/step also means each language is seen in
proportion to its share of draws, so all six get exposed; no language is starved. Recorded as a
deviation, not as a redesign.

**Arms: `hnet` only, deliberately.** Q1's claim is mechanistic — *does the chunker find
character starts*. A `byte` transformer answers "did hierarchy help", which is Q2's
loss-to-loss question and would double the budget. `hnet-space` (whitespace boundaries) would
isolate learned from hierarchical routing, but at 2.0 h/seed it is the first thing to add after
the primary lands, not before. If the primary shows no lift, `hnet-space` becomes mandatory
before any null is reported — a null without it cannot distinguish "learned routing failed" from
"chunking itself did nothing".

**N fixed at 6, no sweep.** The paper matches BPE's 4.57; practitioners call N per-domain. At
0.84 M params a 3-point sweep is 12 h. N=6 also sits between the dev loop's observed regimes:
`cmn` was fragmenting at ~5 bytes/chunk and `rus`/`jpn` were comfortable at 3-5, so 6 is not
pushed to either extreme. N is swept only after the primary run and only if the quota allows,
and any sweep is reported as a separate table, never merged into the primary.

**Seeds: 2, both reported.** Published seed spread at comparable scale is ~0.005 bpb and Q1
reads differences of that size. Report each seed's curve and the mean; a claim is only made if
both seeds agree in sign.

**Guardrails, set before the push**

- Explicit `max_steps=5000` **and** a 6 h wall-clock guard; the run must never end because the
  session died.
- Log realised FLOPs/byte each eval, not the target. FLOPs/byte moves with realised chunking —
  the same architecture at 3 vs 12 bytes/chunk is a different experiment.
- Ratio loss α stays 0.03 for the whole run; the N-sweep does not retune it.
- No mid-flight change to size, N, LR or batch. If a run dies, restart at the same config and
  the same seed; note the restart count.
- Gate suite (`tests/test_gates.py`) green immediately before each push.
- The pre-registered "chunker engaged" definition from *Gates on Kaggle* is the pass criterion;
  it is not revisited after seeing numbers.

**Consequence to state in the verdict.** At 0.84 M params a null is a statement about 0.84 M,
not about H-Net — the map already says failure is reported as a statement about 8 M, and that
number is now 0.84 M.


## Amendment, applied during the run

Measured throughput on the first T4 attempt was **3.25 s/step**, not the 1.4 s the calibration
implied (the pure-Python scan is launch-overhead bound at this size). Both seeds under one 6 h
guard would have been cut near 3300 steps, under the frozen 5000-step floor. Corrections, applied
at step 75 of the first attempt rather than after the fact:

- **One seed per kernel** instead of both in one: `mauricew/hnet-q1-s0`, then `-s1`. Each gets
  its own 8 h guard, so 5000 steps fit (~4.3 h) with 3.7 h of slack.
- Wall guard raised 6 h -> 8 h total **per kernel**.
- Nothing about size, N, LR, batch or language mix changed. The seeds are still both reported;
  they are simply no longer in the same session.

Same config also means the retry is comparable to the attempt that was killed at step 75.
