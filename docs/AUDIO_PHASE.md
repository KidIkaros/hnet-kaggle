# Audio phase — design (not yet run)

Status: **design only, zero quota spent.** Written 2026-11-01 after the scaling phase closed
(see [RESULTS_ENG.md](RESULTS_ENG.md)). Quota at writing: **21.95 h used, 8.05 h left, reset
2026-11-07** (`tools/facts.sh`, verified — the earlier 10.04 h figure was stale).

## The question

Cartesia's motivation for audio H-Nets is that models should "autonomously abstract raw audio
into phonemes and words." Our alignment study (WRITEUP.md, audio arm) measured the opposite at
small scale: across four encodings of the same 37.8 minutes of speech, the chunker held the
**byte count** of a chunk constant while its **duration** floated by 5.5×; onset alignment was
zero in every arm; and the little-endian / big-endian models placed boundaries at the same
offsets one byte apart (Jaccard 0.036 vs chance 0.114) — a byte-value-class phase, not acoustic
structure.

**So: does acoustic abstraction emerge with compute, or does the byte-phase persist?** This is
the audio version of the mechanism question, asked at ~25× the alignment study's compute, with
the scaling phase's calibrated expectations (b ≈ 0.111 — improvement is slow; do not expect
qualitative emergence to be cheap).

## Phase 0 — data build (free, CPU)

`mauricew/hnet-audio-bytes` is 37.8 minutes — fine for the alignment study, far too small for
an 18.6 M-parameter model (the 3e16 budget alone trains on ~0.6 GB; 37.8 min at 16 kHz PCM16
is ~260 MB, i.e. ~2.3 epochs — thin but the real problem is diversity: 8 clips).

Build `mauricew/hnet-audio2-bytes` from **LibriSpeech train-clean-100** (100 h, ODC-legal
CC-BY-4.0):

- 16 kHz, signed 16-bit little-endian PCM, raw bytes — same serialization family as the
  existing `pcm16le` arm so the diagnostics transfer.
- Same packer discipline as `kernel/audio_pack.py`: document (clip) offsets, fixed val split by
  seed, no overlapping windows, one `val.bin` + offsets file.
- ~11.5 GB packed. Kaggle dataset limit is fine; pack time is CPU-only.
- Keep the original 37.8-min 8 kHz set untouched as the small-scale comparison arm.

Decision to make at build time (cheap to change before any GPU spend): 16 kHz only, or an
additional 8 kHz downsample for direct comparability with the alignment arms. Default:
**16 kHz only** — the onset metric needs the temporal resolution, and the small-scale arm is
already published in the write-up.

## Phase 1 — the ladder (GPU)

Reuse the English scaling harness unchanged (`kernel/eng.py` driver: guards, TrainStep, FLOPs-
derived steps, canonical eval) with an audio corpus swap. Same 18.6 M architecture. Two
budgets, two independent seeds per budget in one session each (the parallel-jobs-per-T4 model;
DataParallel stays dead).

| session | nominal budget | jobs | est. hours |
|---|---|---|---|
| 1 | 1.9e16 | seeds 0+1, one per T4 | ~3.0 |
| 2 | 3.0e16 | seeds 0+1, one per T4 | ~4.65 |
| | | **total** | **~7.65 + launch risk (~0.2/attempt)** |

**Quota math:** 7.65 h against 8.05 h remaining leaves a 0.4 h margin — one failed launch
(0.21 h last time) erases it. **Recommendation: run after the 2026-11-07 reset.** If impatience
wins, the single-budget option is the 3e16 session only (~4.9 h with risk, ~3 h buffer) — it
answers the emergence question; only the exponent is lost.

## Metrics — rule 7, one metric one sample per question

1. **bpb** (bits per byte, audio): driver `evaluate()`, fixed 40-batch audio eval sample, seed
   0, byte count recorded with the number — same protocol as English, numbers comparable across
   the two budgets only.
2. **Onset lift**: the alignment-study metric via `tests/audio_cross.py`'s code path, on a
   held-out LibriSpeech slice with energy-based onset markers. This is THE emergence metric.
3. **Byte-phase diagnostics** (free, from one model): boundary histogram mod 2 (byte-value
   class signature), and the chunk-duration CV test — does chunk *duration* variance shrink
   relative to chunk *byte-length* variance, or does bytes/chunk still lock near the ratio
   target while duration floats?

## Pre-registered predictions and kill criteria

Calibrated by Part 2: the model has no mechanism we know of that makes audio easier than text,
and the measured exponent says improvement is slow.

- **P1 (loss):** bpb improves with budget with exponent in the 0.08–0.15 band (consistent with
  the English measurement). Kill: exponent < 0.05, or `guard_reason()` fires on both seeds at
  either budget (architecture/data problem — stop, do not spend the second session).
- **P2 (emergence):** onset lift rises above the small model's +0.003 but stays below +0.05 —
  byte-phase still dominates at this scale. **Success** (acoustic abstraction emerging): onset
  lift > +0.10 at the 3e16 budget. **Kill / honest negative:** onset lift < +0.01 at 25×
  compute → "the byte-phase persists; no acoustic abstraction in this regime" — consistent
  with the scaling phase's finding and worth writing up as such.
- **P3 (mechanism):** if P2 is flat, the duration test decides *which* byte-phase: duration CV
  still dominated by byte-length → same byte-value-class mechanism as the small model;
  duration CV shrinking while bytes/chunk stays ~6 → the router is starting to track the
  signal even without onset alignment.

Null results are results: P2-kill plus a measured P1 exponent is a complete, publishable
audio-phase answer at this budget.

## What this phase does NOT test

- Phoneme/word segmentation against a reference aligner (the blog's qualitative claim needs
  transcription-aligned eval — possible later, not pre-registered here).
- Multi-stage H-Nets (S>1), the 100B-byte regime, or the 1.28× multiplier on audio.
- The fixed-patch control on audio (shelved with the text control; same ~10-line design).

## Sequence

1. Phase 0 build + CPU gates on the packed bytes (`tests/test_audio_metrics.py` must pass on
   the new corpus format). Free.
2. `tools/facts.sh` re-run immediately before any GPU launch (quota rule).
3. After the 2026-11-07 reset: session 1, poll, analyse; session 2, poll, analyse.
4. Write-up appended to `docs/` in the same style as RESULTS_ENG.md.
