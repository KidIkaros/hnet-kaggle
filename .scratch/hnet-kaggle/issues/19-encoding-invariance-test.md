# Encoding invariance: the second stage of the claim

Type: task
Status: resolved
Blocked by: 02, 11

## Question

The pre-registered claim (ticket 02) has two stages. Stage one -- alignment above chance -- is
measured, replicated, and in the write-up. **Stage two -- that the alignment survives re-encoding
to UTF-16LE -- has never been run.** It is the half of the claim that would distinguish "the
chunker learned characters" from "the chunker learned UTF-8".

The design is fixed in ticket 11: the same text in two encodings, character positions fixed,
byte positions shifted; match on characters; UTF-16LE's lead/trail distinction is offset parity
rather than a DFA; the invariance statistic is **boundary agreement in character space**.

## Method

- The UTF-16LE variant of the corpus is the same documents, re-encoded, with document boundaries
  preserved (`kernel/utf16.py`).
- A model is trained on the UTF-16LE bytes under the same frozen config (0.84 M params, N=6,
  5000 steps).
- Both models are then run on the **same text**: the UTF-8 model on UTF-8 bytes, the UTF-16 model
  on UTF-16LE bytes of the same characters. A UTF-8 boundary at byte `t` maps to the character
  index that byte starts; a UTF-16LE boundary at byte `t'` maps to `t' // 2`.
- Reported per language: `u8_lift` (alignment in UTF-8), `u16_lift` (alignment in UTF-16, with
  the even-offset base rate), `agreement` (Jaccard over boundary character sets) and `char_f1`.

**Passing** = the UTF-16 model also aligns on characters, **and** the two models place boundaries
on the same characters. **Failing** = the UTF-16 model does not align, or aligns somewhere else,
and the chunker has learned the byte phase of one encoding rather than the characters.

The UTF-8 side is well measured already: the ten checkpoints from the replication (seeds 0-9) are
shipped in the dataset, so every UTF-16 seed is compared against ten independent UTF-8 models.

## Answer

**STAGE TWO OF THE PRE-REGISTERED CLAIM FAILS. The chunker does not survive re-encoding: in
UTF-16LE it places boundaries *inside* characters for exactly the scripts where it aligned in
UTF-8. What we measured is a property of UTF-8's byte layout, not of characters.**

Kernel `mauricew/hnet-u16-invariance`, COMPLETE, 2 UTF-16 seeds x 10 UTF-8 checkpoints = 20
paired comparisons per language.

| script | UTF-8 lift | UTF-16 lift | agreement | char F1 | reading |
|---|---|---|---|---|---|
| eng | +0.000 | **-0.354** | 0.272 | 0.413 | UTF-8 was unmeasurable; UTF-16 is clearly anti-aligned |
| rus | **+0.178** | **-0.332** | 0.294 | 0.449 | **sign flips** |
| cmn | -0.153 | **+0.143** | 0.439 | 0.601 | **sign flips** |
| jpn | -0.161 | -0.057 | 0.427 | 0.594 | both negative |
| arb | **+0.254** | **-0.316** | 0.263 | 0.411 | **sign flips** |

Per UTF-16 seed (the UTF-16 side is one model per seed; the UTF-8 side averages ten):

| UTF-16 seed | eng | rus | cmn | jpn | arb |
|---|---|---|---|---|---|
| 0 | -0.373 | -0.265 | +0.217 | -0.043 | -0.223 |
| 1 | -0.334 | -0.399 | +0.069 | -0.072 | -0.408 |

Both seeds agree in sign on every script.

**The metric was audited before the result was believed.** In UTF-16LE every character occupies
two bytes, so character starts are at even byte offsets -- verified on real text (35 characters,
70 bytes, starts at every even offset). A lift of -0.354 therefore means the model's boundaries sit
at *odd* offsets, i.e. **between the two bytes of a character**: an all-odd boundary set would
score -0.5, an all-even set +0.5. The observed value is not a subtle bias; it is a strong,
consistent preference for the wrong side of the character.

**What this does to the study's headline.** The claim in *The claim and its falsifier* had two
stages: alignment above chance **and** survival of re-encoding to UTF-16LE. Stage one holds, in
UTF-8, replicated at n=10. **Stage two fails.** The scripts where the UTF-8 model aligned to
characters (Russian, Arabic) are the scripts where the UTF-16 model most strongly *avoids*
them. Chinese, negative in UTF-8, is positive in UTF-16. Boundary agreement between the two
encodings is 0.26-0.44, so the two models do not even pick the same characters.

**The honest re-reading of everything in this effort.** The measured effect is "the chunker's
boundaries land on UTF-8 character starts in some scripts and not others". The natural
interpretation -- "the chunker learns characters" -- is falsified by the re-encoding. What it
learns is something about **UTF-8's byte layout**, and the cross-script differences (Russian and
Arabic align, Chinese and Japanese do not) are differences in how readily that layout is
recoverable per script. That reframes the five refuted mechanism candidates: width, whitespace,
ASCII density, script type and chunk period were all candidates for *character* alignment, and
the invariance result says the target was never characters to begin with.

**Confounds and their status.** (a) This run trained the UTF-16 model at N=6 while the UTF-8
checkpoints were trained at N=4 -- a real confound, and a matched N=4 run is in flight
(`mauricew/hnet-u16-n4`). (b) A benign alternative for English specifically: in UTF-16LE ASCII is
stored as a byte plus `0x00`, so an odd-offset boundary there falls between a letter and its null
-- the model may be learning the null interleaving. That is still encoding structure rather than
characters, but it is a different mechanism from Russian and Arabic, whose UTF-16 bytes contain
no nulls at all, and it would not explain their equally strong negative lifts.

