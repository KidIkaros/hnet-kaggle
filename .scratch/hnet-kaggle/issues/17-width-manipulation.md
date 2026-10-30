# Width manipulation: the causal test

Type: task
Status: open
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

<!-- filled on resolution -->
