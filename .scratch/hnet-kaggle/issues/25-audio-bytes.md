# Audio: encoding invariance on a waveform

Type: task
Status: open
Blocked by: 23

## Question

Audio is the modality where serialization is most obviously arbitrary: the same sound can be
16-bit PCM, 8-bit mu-law, float32, or bytes of a compressed codec, and each layout puts its
structure in a different place. It is also the modality where "the semantics" are unambiguous --
waveform periodicity, onsets, and phoneme boundaries are measurable ground truth.

So audio tests the same question as vision, on data with a well-defined semantic reference:
does a byte-level chunker align to **periodicity and onsets**, or to the **sample-format phase**?

## Design

- Data: a small speech corpus (a few hours is enough at this scale), the same clips in four
  encodings: 16-bit PCM LE, 8-bit mu-law, float32, and 16-bit PCM BE (a pure byte-order change
  that preserves every sample).
- The byte-order arm is the sharpest single test in the whole roadmap: **PCM LE vs PCM BE are the
  same samples with every byte pair swapped.** If chunking is semantic, boundary *times* should be
  identical; if it is a byte-phase artifact, the boundaries will move with the byte order and
  agree only by chance.
- Metrics: boundary positions in **time** (sample index), agreement between encodings, and
  alignment to onsets/pitch periods.

## Pre-registered prediction

From text and the byte-phase result: the PCM-LE vs PCM-BE pair will show near-chance agreement in
time, and each encoding will show a strong offset-mod-2 concentration of its own. If instead the
two byte orders agree on boundary times, the chunker is following sample content across a
byte-order change, which is the strongest possible evidence for semantic chunking and would
overturn the text reading.

**Kill criterion.** Agreement between LE and BE at or near chance is a positive confirmation of
the byte-phase account, not a null: it is the cleanest single demonstration that the chunker reads
layout, and it is worth reporting on its own.

## Cost

CPU-feasible. The LE/BE pair alone is two 5000-step runs.
