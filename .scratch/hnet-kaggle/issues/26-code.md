# Code: does chunking find syntax, or the byte phase again?

Type: task
Status: open
Blocked by: 23

## Question

Code is the domain where Cartesia reports its strongest multiplier, and it is also the domain with
the most explicit *hard-coded* structure: brackets, indentation, statement boundaries. If a
learned chunker finds syntax anywhere, it should find it here.

The confound that makes text hard to read does not apply: syntax boundaries are not periodic in
bytes, so "aligns to syntax" and "aligns to a byte phase" are separable.

## Design

- Data: permissively licensed code (the map rules out The Stack v2 as gated; use a permissive
  subset or a public Kaggle code corpus). Four encodings as in vision: UTF-8, UTF-16LE, and with
  and without comment/whitespace stripping.
- Metrics: alignment of boundaries to (a) tokenizer-independent syntactic positions -- line starts,
  indentation changes, bracket depth changes -- and (b) the byte-phase statistics from ticket 21.

## Pre-registered prediction

If syntax is learnable at this scale, boundaries should align to line starts and bracket-depth
changes above chance, in *both* encodings (since syntax is encoding-invariant). That would be a
genuine positive result and the first evidence in this effort that a learned chunker tracks
meaning rather than layout.

**Kill criterion.** If alignment to syntactic positions is at chance while the byte-phase
concentration is intact, then even in the domain with the most explicit structure the chunker
reads layout -- which is a strong, general statement.

## Cost

CPU-feasible; data licensing is the main obstacle, not compute.
