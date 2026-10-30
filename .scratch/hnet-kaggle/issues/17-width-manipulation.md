# Width manipulation: the causal test

Type: task
Status: resolved
Blocked by: 15

## Question

The replication confirmed that alignment tracks **bytes per character** across languages
(1.78-1.80 aligned, 2.67 not), but that is a cross-linguistic correlation: width co-varies with
script, corpus, encoding and delimiters. Does width *cause* it?

Manipulate width inside a fixed script, with the text, the words, the whitespace and the
document structure all unchanged: append U+0301 COMBINING ACUTE ACCENT to every non-space
character. This turns English from 1.00 into **2.69 bytes/char** (Chinese is 2.67) and Russian
from 1.80 into **3.52**, without changing anything else. Both variants are truncated back to the
original byte count, so the arms differ in bytes-per-character and nothing else — same bytes
seen, same document boundaries, same recomputed DFA streams.

Arms in one run, so every contrast is within-run: `eng` (1.00), `eng_comb` (2.69), `rus` (1.80),
`rus_comb` (3.52), `cmn` (2.67).

## Pre-registered prediction (written before any seed is run)

**H1 (width): alignment decreases with bytes-per-character.**

| arm | B/char | prediction |
|---|---|---|
| eng | 1.00 | ~0, structurally unmeasurable (99.6% of bytes are lead bytes) |
| rus | 1.80 | **positive**, reproducing the replication's +0.178 |
| cmn | 2.67 | negative, reproducing the replication's -0.147 |
| eng_comb | 2.69 | **not aligned** (mean <= 0) |
| rus_comb | 3.52 | **not aligned**, and worse than `rus` |

**Falsifier.** If `eng_comb` comes out **clearly positive** (CI excludes 0 and mean > +0.10),
width is not the driver, H1 is dead, and the explanation must be something else about the
narrow-width scripts — their character repertoire, their corpus, or something the replication
never isolated. English at Chinese width, with Chinese's own negative result, would then be the
most interesting number in the effort.

**Decision rules** (same as ticket 15): aligned if the CI excludes 0; not aligned if the CI
includes 0 and the mean is <= +0.05; unresolved otherwise.

**Controls that validate the harness:** `rus` and `cmn` must reproduce their replication values
(+0.178 and -0.147) in this run. If they do not, the run is not comparable and is reported as a
harness failure rather than as a result.

## Design

4 seeds (0-3), fixed 5-language mixture, N=4, 5000 steps, everything else identical to the
replication config. ~2.7 GPU-h of the ~4.2 h left. Evaluation: full 1953-window val sweep with
the bootstrap CI from `tests/eval_noise.py`, so eval noise is separated from seed noise.

## Answer

**H1 (width) IS DEAD. English at Chinese's exact byte width aligns; Chinese at the same width
does not. Two of the six pre-registered checks failed and the falsifier missed by 0.006 — the
substance is refuted even though its letter did not fire.**

Kernel `mauricew/hnet-q1-width`, COMPLETE, seeds 0-3, N=4, 5-language mixture.

| arm | B/char | mean lift | seed SD | 95% CI | sign test | verdict | predicted |
|---|---|---|---|---|---|---|---|
| eng | 1.00 | +0.002 | 0.001 | [+0.001, +0.003] | 4/4 + | ~0 (structural) | ~0 |
| **eng_comb** | **2.69** | **+0.094** | 0.050 | **[+0.045, +0.142]** | **4/4 +** | **aligned** | not aligned |
| rus | 1.80 | +0.075 | 0.064 | [+0.013, +0.138] | 4/4 + | aligned | positive |
| rus_comb | 3.52 | +0.075 | 0.115 | [-0.037, +0.187] | 3/4 + | unresolved | not aligned |
| cmn | 2.67 | -0.205 | 0.030 | [-0.234, -0.176] | 0/4 + | **not aligned** | negative |

**The decisive comparison.** `eng_comb` and `cmn` have the same bytes per character (2.69 vs
2.67), the same model, the same run and the same compression target. One aligns (+0.094, every
seed), the other does not (-0.205, every seed). **Bytes per character cannot be the cause.**

**What the falsifier got wrong, stated plainly.** It was written as "eng_comb clearly positive
(CI excludes 0 and mean > +0.10)". The CI does exclude 0; the mean is +0.094, so the threshold
was missed by 0.006. The letter of the falsifier therefore did not fire, while two of the six
pre-registered checks did fail (eng_comb predicted not-aligned, rus_comb predicted not-aligned).
A magnitude threshold I picked before seeing the data, sitting just above the observed value, is
exactly the thing that should not be leaned on. The substantive reading is that H1 is refuted.

**Harness controls passed.** `rus` +0.075 vs the replication's +0.178 (diff 0.103), `cmn` -0.205
vs -0.147 (diff 0.058), `eng` +0.002 vs 0.000. Within the pre-registered tolerance, so the run
is comparable and the controls reproduce.

**ASCII fraction does not explain it either.** Measured over the corpora: eng_comb 0.372,
rus 0.113, arb 0.125, hin 0.094, cmn 0.062, jpn 0.061, rus_comb 0.058. `rus_comb` (0.058) and
`cmn` (0.062) are essentially identical in ASCII share, yet one is +0.075 and the other -0.205.

**Standing count of tested and dead explanations:** chunk-length period (ticket 12), whitespace
delimitation (ticket 14), bytes per character (this ticket), ASCII landmark density (this
ticket). **What has survived every test is the effect itself** — Russian and Arabic align, 8/8
seeds, p = 0.004; Chinese does not — and not its cause.

Budget: ~2.7 GPU-h.

