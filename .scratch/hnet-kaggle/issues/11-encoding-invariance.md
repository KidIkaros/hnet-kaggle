# Encoding: is learned chunking invariant to the byte representation?

Type: research
Status: resolved
Blocked by: —

## Question

Does the chunker learn *characters*, or does it learn *this encoding*?

Provoked by a comparison table of alternative encodings and by the three 2026 papers below.
`docs/ENCODING.md` holds the full analysis; the short version is that the speed case is
void (UTF-8 validation is 36 GB/s against a model that eats 14 kB/s) but the density case
is real and already published — for tokenizers, never for a learned chunker.

Prior art, all to be read before designing anything:

- **UBE** (arXiv:2610.01984, NeurIPS 2026, 1 Oct 2026) — "encoding floor": 3-byte BMP scripts
  cost 3 symbols; routes them through UTF-16 to reach 2. Token counts fall, cross-lingual
  token-budget disparity narrows, LM quality matched.
- **BBPE16** (arXiv:2602.01717, ICASSP 2026) — UTF-16 BBPE, −10.4% Chinese tokens, −10.3%
  decode iterations.
- **Bit-level BPE** (arXiv:2506.07541) — the same redundancy attacked below the byte boundary.

All three target *tokenizers*. None tests a dynamic chunker. Confirm that by reading them,
and confirm nobody has published an encoding-invariance test of a learned chunker.

This ticket decides:

1. Whether the invariance test (same text in UTF-8 and UTF-16LE, character positions fixed,
   byte positions shifted) is sound as designed, and what a passing/failing pattern looks
   like.
2. The metric change: bits-per-**character** for cross-encoding comparison, bpb retained
   alongside for comparability to the paper. Is that coherent, and what breaks?
3. Whether the DFA alignment tool generalises — UTF-16LE's state machine is trivial (lead vs
   trail unit by offset parity), but confirm the builder can emit it and that the metric in
   *Alignment metric and DFA stream* survives the swap.
4. Whether UTF-16LE alone is enough, or UBE's routing is required for a fair test.

Resolve with: a sound or amended design, and the metric definition.

## Answer

**The design is sound with five amendments. UTF-16LE alone is sufficient; UBE's routing is
not required.**

What the reading established:

1. **UBE does not test a chunker.** It keeps standard BPE and changes only the byte
   representation presented to BPE ("UBE changes only the byte representation presented to
   byte-pair encoding (BPE); the merge rule remains standard"). No learned chunker appears
   anywhere in it. The gap is confirmed, not assumed.
2. **UBE cites H-Net and states the expected direction** (App. H.3): H-Net "learns content-
   and context-dependent byte segmentation jointly with the model, outperforming
   BPE-tokenized Transformers with especially large gains for Chinese and code." Our guess
   about which scripts move first has published support.
3. **UBE kept bpb and added a dispersion statistic** rather than switching to a
   per-character metric: its LM table reports `BPB↓` and **`BPB Gini↓`** over 101 held-out
   mC4 splits. Borrow the Gini; the primary cross-encoding metric still has to be
   per-character, because bpb is encoding-relative.
4. **New adjacent prior art**: FLEXITOKENS (Owodunni et al., 2026) — learnable boundary
   predictors between bytes, to reduce over-fragmentation, applied to ByT5. It is a boundary
   learner like H-Net's router and must be read before any novelty claim.

Amendments:

- **A1 — match on characters, not bytes.** A fixed byte budget gives UTF-16LE half the
  characters of UTF-8 on CJK. Match steps/batches on character count; report byte counts.
- **A2 — dual metric.** Primary: **bits per character**. Secondary: bpb, for comparability
  with the paper and with UBE's tables. Always both; a per-character number without bpb is
  not checkable.
- **A3 — the direct invariance statistic** is boundary agreement at equal compression: of
  the character positions marked under UTF-8, what fraction are also marked under UTF-16LE,
  and vice versa. Alignment-with-characters alone can be an artefact of both encodings
  sharing prefix structure; agreement is not.
- **A4 — the DFA generalises trivially.** Under UTF-16LE lead-vs-trail is offset parity (even
  offset = code unit start); no state machine needed. Add a unit test that emitted offsets
  agree with Python's `str.encode('utf-16-le')` boundaries.
- **A5 — report BPB Gini across scripts** (borrowed from UBE) so dispersion is one number.

Rejected: UBE routing inside the test (adds a second variable on top of encoding; UTF-16LE
isolates the representation question). Rejected: BOCU-1/SCSU (stateful), FSST (not an
encoding), StreamVByte (needs a vocabulary first) — `docs/ENCODING.md` §3.

**Sequencing:** this only earns its runs if the chunker engages at all. It stays behind Q1,
and its result is a mechanism claim, not a compute claim.
