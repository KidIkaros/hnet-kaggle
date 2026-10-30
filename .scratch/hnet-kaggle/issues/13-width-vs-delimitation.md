# Separate character width from word-delimitation

Type: task
Status: resolved
Blocked by: 12

## Question

The six Q1 languages confound two explanations for the CJK alignment failure: CJK is both
3 bytes/character *and* unwritten with spaces. Which one breaks the chunker?

Run `hin_Deva` — 2.53 bytes/char (CJK-like width) with a 0.192 space rate (the highest measured,
well above Russian's 0.132) — at N=4 and N=6, seed 0, everything else frozen.

## Answer

**Whitespace delimitation, not character width. Hindi aligns strongly at CJK-like byte width;
the two unwritten scripts do not.**

Kernel `mauricew/hnet-q1-hin`, COMPLETE, seed 0, 7 languages, N=4 and N=6, ~0.7 GPU-h. Lift is
the last-3-eval mean; `@B` is realised bytes/chunk.

| lang | bytes/char | space rate | N=4 lift | N=6 lift |
|---|---|---|---|---|
| rus | 1.80 | 0.132 | **+0.270** | **+0.358** |
| arb | 1.78 | 0.162 | **+0.282** | +0.063 |
| **hin** | **2.53** | **0.192** | **+0.229** | **+0.325** |
| cmn | 2.67 | 0.036 | -0.022 | **-0.209** |
| jpn | 2.68 | 0.020 | -0.023 | **-0.264** |
| deu | 1.02 | 0.133 | -0.010 | -0.029 |
| eng | 1.00 | 0.155 | +0.003 | -0.004 |

Hindi sits 0.14 bytes/char from Chinese and Japanese and is positive at both ratios, while they
are negative at N=6. Its space rate is 5-10x theirs. Width cannot be the discriminator; the
presence of whitespace can.

**The two-factor reading that fits every row:**

1. **Multi-byte characters are necessary** — `eng` (1.00 B/char) and `deu` (1.02) sit at zero, but
   for a structural reason, not a learned one: 99.6% and 98.5% of their bytes are lead bytes, so
   there is almost no lift available. They are unmeasurable on this metric, not negative.
2. **Whitespace-delimited words are necessary** — with multi-byte characters present, the three
   languages that mark word boundaries with spaces (rus, arb, hin) are positive; the two that do
   not (cmn, jpn) are at chance or negative.

The mechanism this suggests: a word boundary is trivially a character start, so a router that
anchors on whitespace scores as character-aligned on any multi-byte script. With no whitespace
there is no anchor, and the router drifts to whatever the ratio term asks for. That is consistent
with ticket 12, where matching chunk length to one character made `cmn` no better.

**Caveat that must travel with this table.** Adding `hin` to the training mix moved `cmn` at N=4
from **-0.206** (six-language sweep) to **-0.022** (this seven-language run). Per-language
alignment is therefore sensitive to what else is in the mixture, and a single-language number is
not a stable property of that language. The `hin` vs `cmn`/`jpn` contrast is safe because it is
within one run at the same N; cross-run comparisons of the same language are not. `arb` at N=6
(+0.063) versus N=4 (+0.282) is a second warning that individual cells carry real variance.

Budget: 0.7 GPU-h.
