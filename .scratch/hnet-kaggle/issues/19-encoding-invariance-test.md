# Encoding invariance: the second stage of the claim

Type: task
Status: open
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

<!-- filled on resolution -->
