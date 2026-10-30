# Where does a byte-level H-Net's chunker actually put boundaries?

**A small-scale, replicated measurement of character alignment in learned chunking, and four
interpretations that the data killed along the way.**

## The claim

At 0.84 M parameters, trained 5000 steps on raw bytes with a 1-stage H-Net, the chunker places
boundaries on UTF-8 **character starts** above chance in Arabic and Russian, and below chance in
Chinese and Japanese. English and German are structurally unmeasurable on this metric.

**And the alignment does not survive re-encoding.** The same text presented as UTF-16LE, with
every character at an even byte offset, moves the model to the *wrong* side of the character in
exactly the scripts where UTF-8 aligned: Russian -0.228 and Arabic -0.326, against UTF-8's
+0.179 and +0.250, with the two encodings agreeing on only 36-46% of boundary characters.
Chinese flips the other way (UTF-8 -0.153, UTF-16 +0.334). **So what this study measures is a
property of UTF-8's byte layout, not of characters** — and the cross-script ordering reverses
between encodings, which means "Russian and Arabic align, Chinese and Japanese do not" is a
property of the script *x encoding* pair, not of the scripts.

This is a statement about a small model on a fraction of an epoch. It is **not** evidence about
the compute multipliers, the large-byte regimes, multi-stage H-Nets, or downstream quality — see
[Limits](#what-this-does-not-show).

The falsifier was fixed before the run: alignment above chance on all three of rus/jpn/arb, or the
`cmn` phase loss being a chunk-length artefact, would have falsified the claim as stated. Neither
happened; see [What we killed](#what-we-killed).

## Setup

| | |
|---|---|
| model | 1-stage byte-level H-Net, `d_model` 128, 2 encoder / 2 main / 2 decoder blocks, `d_ff` 512, `d_state` 16, `d_conv` 4 — **0.84 M params** |
| implementation | pure PyTorch, no `mamba_ssm`, no `flash_attn` — a chunked Hillis-Steele scan with explicit carried state |
| data | `mauricew/hnet-multiscript-bytes`: 50 MB per language from `epfml/FineWeb2-HQ-Plus` (Oct 2026) and `epfml/FineWeb-HQ`, `text` column only, document offsets + 8-state UTF-8 DFA streams |
| scripts | eng, deu, rus, cmn, jpn, arb (+ `hin_Deva` and space-ablated variants in the side experiments) |
| training | 5000 steps, batch 16 x 256 bytes, AdamW, lr 1e-3, `target_ratio` N=6 (replication at N=4), 8 seeds |
| hardware | one Kaggle T4, ~40 min per seed; all evaluation on CPU |
| metric | alignment lift `P(byte is a UTF-8 lead byte | boundary) - P(byte is a lead byte)`, per language, on held-out `val.bin` |

### The metric, and where it cannot see

A boundary at byte `t` is scored against byte `t`'s own UTF-8 state. The metric is a **lift over
the per-script base rate**, not raw precision: raw precision is trivially gamed by a model that
puts boundaries everywhere, and the base rate differs by script (99.6% of English bytes are lead
bytes, 37.5% of Chinese ones).

Two properties matter for reading the results:

- **English and German are unmeasurable, not failing.** 99.6% and 98.5% of their bytes are
  already lead bytes, so there is almost no lift available. Their zeros are structural.
- **The estimator's own noise is small.** On a full 1953-window recomputation of the val stream,
  the 8-window estimator's bootstrap SD is 0.017-0.018 for rus/cmn/jpn/arb. Across-seed SDs are
  0.085-0.169 — **5-10x larger**. The uncertainty in what follows is training stochasticity.

## Results

Ten seeds, fixed 6-language mixture, N=4, every seed 5000 steps. Lift is the mean of the last
three evals per seed; the CI is across seeds.

| script | bytes/char | mean lift | seed SD | 95% CI | seeds negative | verdict |
|---|---|---|---|---|---|---|
| eng | 1.00 | -0.000 | 0.002 | [-0.002, +0.001] | 6/10 | not aligned (structural) |
| deu | 1.02 | -0.001 | 0.013 | [-0.009, +0.007] | 4/10 | not aligned (structural) |
| **rus** | 1.80 | **+0.169** | 0.079 | **[+0.120, +0.218]** | **0/10** | **aligned** |
| cmn | 2.67 | **-0.165** | 0.107 | [-0.232, -0.099] | **9/10** | **not aligned** |
| jpn | 2.68 | **-0.155** | 0.164 | [-0.257, -0.054] | **8/10** | **not aligned** |
| **arb** | 1.78 | **+0.250** | 0.110 | **[+0.182, +0.318]** | **0/10** | **aligned** |

Russian and Arabic are positive in **every seed**; Chinese and Japanese are negative in 9 and 8
of ten. The two groups' CIs do not overlap. Every pre-registered check passes: `arb ~ rus`
(CIs overlap), `cmn ~ jpn` (CIs overlap), and the explicit kill criterion — rus and cmn
overlapping — is not met.

**Pooled over every mixture measured (the honest summary, because the per-mixture numbers
vary):**

| script | n seeds | mean lift | 95% CI | seeds negative | verdict |
|---|---|---|---|---|---|
| eng | 14 | +0.000 | [-0.001, +0.001] | 6/14 | not aligned (structural) |
| eng_comb | 9 | +0.038 | [-0.027, +0.102] | 2/9 | not aligned |
| **rus** | 19 | **+0.146** | **[+0.110, +0.182]** | **0/19** | **aligned** |
| rus_comb | 4 | +0.075 | [-0.037, +0.187] | 1/4 | unresolved |
| **cmn** | 19 | **-0.132** | **[-0.210, -0.053]** | 16/19 | **not aligned** |
| jpn | 10 | -0.155 | [-0.257, -0.054] | 8/10 | not aligned |
| **arb** | 10 | **+0.250** | **[+0.182, +0.318]** | **0/10** | **aligned** |
| tha | 5 | -0.077 | [-0.179, +0.025] | 3/5 | not aligned (unstable) |

Russian and Arabic are positive in **29 of 29 seeds** across every mixture that contained them.
Chinese is negative in 16 of 19 and Japanese in 8 of 10. Everything else is at chance or unstable.

**The split is by script, its cause is not identified, and it does not survive re-encoding.** Bytes per character was the obvious
candidate and it is wrong: an English corpus widened to 2.69 bytes/char — Chinese's exact width,
same script, same words, same spaces — aligns (+0.094, every seed), while Chinese at 2.67 does
not (-0.205, every seed). Width, whitespace delimitation, chunk-length period and ASCII-landmark
density have each been tested and each failed to explain it. The two groups' CIs still separate
cleanly, so the *effect* is solid; only its cause has escaped every hypothesis thrown at it.

## The invariance test: what we measured is UTF-8, not characters

The pre-registered claim had two stages: alignment above chance **and** survival of re-encoding
to UTF-16LE. The second stage had never been run until now, and it fails.

Same corpus, same documents, same text, encoded UTF-16LE (every character at an even byte
offset). A model trained on the UTF-16 bytes is then run on the *same text* as the UTF-8 models,
and boundary positions are compared in **character space**: a UTF-8 boundary at byte `t` maps to
the character that byte starts, a UTF-16LE boundary at byte `t'` maps to `t' // 2`. The UTF-8
side is the ten replication checkpoints.

| script | UTF-8 lift | UTF-16 lift | boundary agreement | char F1 |
|---|---|---|---|---|
| eng | +0.000 | +0.038 | 0.388 | 0.556 |
| rus | **+0.179** | **-0.228** | 0.358 | 0.525 |
| cmn | -0.153 | **+0.334** | 0.592 | 0.736 |
| jpn | -0.268 | +0.014 | 0.528 | 0.689 |
| arb | **+0.250** | **-0.326** | 0.460 | 0.628 |

(The first UTF-16 run trained at N=6 against N=4 UTF-8 checkpoints, a real confound; the table
above is the matched N=4 run, and the N=6 run agrees on every script except English.)

The metric was audited before the result was believed: in UTF-16LE every character occupies two
bytes, so character starts are at even offsets — verified on real text. A lift of -0.228 to
-0.326 therefore means the model places boundaries at **odd** offsets, i.e. *between the two bytes
of a character*. An all-odd boundary set would score -0.5.

**The consequence is larger than the measurement.** In the scripts where the UTF-8 model aligned
to characters, the UTF-16 model most strongly avoids them; Chinese, negative in UTF-8, is the
most positive in UTF-16. The cross-script ordering therefore *reverses* between encodings, which
means the aligned/not-aligned split is a property of the script-and-encoding pair, not of the
script. Five mechanism candidates were refuted earlier in this study (chunk-length period,
whitespace, bytes per character, ASCII-landmark density, script type) — every one of them a
candidate for **character** alignment. The invariance result says the target was never characters.

## What the chunker actually learns: a byte phase, not characters

With "characters" refuted, the obvious next question is what the chunker *does* learn. It is
answerable from the checkpoints with no training: bucket every byte by its **offset within its
character** and compare the boundary distribution to the byte distribution. Concentration 1.0
means no preference; below 1 means boundaries avoid that offset, above 1 means they seek it.

| script | encoding | off0 (character start) | off1 | off2 |
|---|---|---|---|---|
| eng | UTF-8 | 1.000 | 1.032 | 0.695 |
| eng | UTF-16 | **0.290** | **1.710** | — |
| rus | UTF-8 | **1.400** | 0.507 | 0.681 |
| rus | UTF-16 | **0.312** | **1.688** | — |
| cmn | UTF-8 | 0.569 | **2.347** | 0.184 |
| cmn | UTF-16 | **1.306** | 0.694 | — |
| jpn | UTF-8 | 0.528 | **2.191** | 0.403 |
| jpn | UTF-16 | 0.777 | 1.223 | — |
| arb | UTF-8 | **1.474** | 0.408 | 0.754 |
| arb | UTF-16 | **0.454** | **1.546** | — |

**The phase is binary.** Every script-and-encoding pair lands either *on* character starts
(off0 > 1) or *one byte inside* them (off0 < 1, off1 > 1.5). Nothing lands in between. The
same checkpoints reproduce every earlier character lift — rus UTF-8 off0 = 1.400 against its
+0.178 lift, cmn UTF-8 off0 = 0.569 against its -0.153 — so this is a restatement of the same
structure in a unit-free form, not a new measurement.

**The phase flips with the encoding, in both directions.** Russian and Arabic go from "on
character starts" in UTF-8 to "one byte inside" in UTF-16; Chinese goes the other way. The phase
belongs to the script-and-encoding *pair*. That is the mechanism behind the invariance failure:
what looked like character alignment was a byte phase that happened to coincide with the
character grid.

**The router keys on byte-value discontinuities.** The paper's boundary rule compares consecutive
byte embeddings, `p_{t+1} = (1 - cos(q_t, k_{t+1}))/2`. In UTF-8 Chinese the router fires on the
*second* byte of a three-byte character (off1 enriched 2.35x) — where the byte value changes
most — rather than at the character's first byte. That is precisely the failure mode of a
byte-level cosine router on scripts where bytes change more inside a character than across
characters.

**Zero-training transfer.** Feeding the UTF-8 model the UTF-16LE bytes of the same text — an
encoding it has never seen — leaves it producing boundaries at a *consistent* phase (odd offsets,
1.6-1.8x concentration, the same phase the UTF-16-trained models use), while its chunk lengths
collapse from ~3.0 B to 1.2-2.3 B and its boundary rate rises from ~0.30 to ~0.50 per position.
The phase is a property of the byte content; the length statistics are a property of the
training, and they do not transfer.

**The positive claim this study ends with.** Learned chunking in a byte-level H-Net is a
**byte-phase segmenter**. Its apparent alignment to characters is a phase coincidence that
reverses under re-encoding, and the router keys on byte-value discontinuities rather than on
characters. This is falsifiable in a specific way: it predicts off1 enrichment in UTF-16LE for
every script, and stronger off1 preference in scripts whose bytes change most within a
character.

## What we killed

Six readings of this evidence were wrong. All six are recorded because the final claim is only
worth what the number of surviving readings is worth.

1. **An off-by-one in the probe.** The first version compared `bmask[t]` against the lead-ness of
   byte `t+1`, shifting "is this boundary on a character start" onto the next byte. It read
   rus -0.420 and arb -0.446 — boundaries apparently *avoiding* characters — and ~0.000 on
   English, because pure ASCII is shift-invariant. `tests/test_alignment_probe.py` reproduces it
   exactly: an oracle boundary set scores +0.530 correctly and -0.033 under the shift.
2. **The chunk-length / character-width hypothesis.** If Chinese fails because chunks of 4.7 bytes
   are 1.5 characters, then one-character chunks should fix it. They do not: at N=4 `cmn` sits at
   2.97 bytes/chunk — *exactly one character* — and scores its **worst** lift, -0.206. There is no
   N at which `cmn` or `jpn` is positive.
3. **The whitespace hypothesis.** Across seven languages, the aligned ones were the
   whitespace-delimited ones, so whitespace looked like the discriminator. Deleting every `0x20`
   from Hindi and Russian and retraining at the same N left alignment **intact** (+0.081 vs -0.037;
   +0.211 vs +0.162). Whitespace is not it.
4. **A single-seed reading of Hindi.** Hindi scored +0.229 at N=4 in one run and -0.037 in another
   with a different training mixture. Per-language alignment turned out to be sensitive to what
   else was in the mix, and the instability (0.27) exceeded the effects being claimed.
5. **Bytes per character as the cause.** The replication showed alignment tracking width across
   languages, so width was manipulated directly: appending a combining acute to every letter of
   the *same* English text turns it from 1.00 into 2.69 bytes/char — Chinese's exact width —
   with the script, words, spaces, documents and byte count unchanged. English at 2.69 aligns
   (+0.094, every seed); Chinese at 2.67 does not (-0.205, every seed). Same width, opposite
   signs. The pre-registered falsifier's magnitude threshold (+0.10) was missed by 0.006, so its
   letter did not fire, but two of six pre-registered checks failed and the substance is refuted.

6. **"The chunker learns characters."** This was the interpretation the whole study rested on,
   and the invariance test refutes it: in UTF-16LE the model puts boundaries *inside* characters
   for exactly the scripts where UTF-8 aligned. What survives is the measurement — "boundaries
   land on UTF-8 character starts in some scripts and not others" — and not the reading.

What survived every test is the **effect** — Russian and Arabic align, 8/8 seeds, p = 0.004;
Chinese does not — and not its cause, and not its interpretation. Five candidate mechanisms have
now been tested and refuted: chunk-length period, whitespace delimitation, bytes per character,
ASCII-landmark density (`rus_comb` and `cmn` have nearly identical ASCII shares, 0.058 and 0.062,
yet +0.075 vs -0.205), and script type (Thai is alphabetic and fails like Chinese).

**Still open: the mechanism, and it has resisted five tests.** Width was the best predictor
until it was manipulated directly and failed. The next candidate was the script type: Chinese and
Japanese are the only logographic scripts in the set, so Thai — alphabetic, wider than Chinese,
unspaced, and with a *lower* ASCII share than Chinese — decides it. **Thai is not aligned**
(-0.145, 3/3 seeds negative), which kills the logographic explanation: an alphabetic script
fails just as Chinese does.

What survives is narrower and stranger. Thai and Chinese share width, the absence of spaces and a
low ASCII share; they differ in script type and still behave the same. English widened to
Chinese's width, with spaces and a high ASCII share, behaves like Russian. So the effect tracks
something Chinese, Japanese and Thai share and Latin, Cyrillic, Arabic and Devanagari do not —
and it is not width, not whitespace, not ASCII density and not script type.

**The most important limitation, and it is not a mechanism at all:** per-language alignment is
sensitive to *what else is in the training mixture*, by more than the effects being measured.
`cmn` reads -0.205, -0.147, -0.037 and +0.040 across four mixtures; `eng_comb` +0.094, +0.020 and
-0.047 across three; `tha` -0.145 and +0.025 across two. The mixture does not only shift the
mean, it changes the **variance** — `cmn`'s seed SD runs from 0.030 to 0.328 — and it changes
which arms are present at all. The 8-seed replication's clean separation is a **fixed-mixture**
result and is reported as exactly that. Any reader comparing this study to another should treat
the mixture as part of the measurement, and should prefer the pooled sign counts
(25/25 for rus and arb) over any single mixture's magnitude.

This also weakens one earlier reading: `eng_comb` pooled over three mixtures is +0.038 with a CI
spanning zero, so "English at Chinese width aligns" is **not** supported either. What the
manipulation does establish is the asymmetry that matters — English at 2.69 B/char does not show
Chinese's negative result (-0.119 pooled, 14/17 seeds), so width cannot be the cause.

## Limits

- **0.84 M parameters, ~7% of one epoch, one T4.** A full pass over the corpus would take 28 h at
  this size; the design is steps-bound, not pass-bound, because ratio dynamics need 5000 steps to
  stabilise.
- **Per-language numbers are mixture-sensitive, by more than the effect sizes.** Every number in
  the main table comes from one fixed mixture; cross-run comparisons of the same language at
  different mixtures are not valid (`cmn` moves by 0.17 across three mixtures). The write-up
  reports composition with every figure, and the Thai run is reported as a harness failure
  because its `cmn` control drifted outside the pre-registered tolerance.
- **The ratio control never locked.** Realised bytes/chunk ranged 2.4-7.1 against a target of 6, so
  N and realised chunk length are not interchangeable on these scripts. Report realised
  bytes/chunk, never the target.
- **Hindi is unreplicated.** It sits at 2.53 bytes/char, the near-zero crossing, and behaved
  exactly as an unstable near-crossing should — but on the runs that are now known to be
  mixture-sensitive.

## Relation to the published scaling work

The Cartesia H-Net posts make claims about compute efficiency (1.28x / 2.06x) and about
loss-to-loss transfer, which need scaling ladders and matched-bpb comparisons. **None of that is
reachable here** — the crossover is at 100B bytes and our whole weekly GPU budget buys ~1.6 GB —
and this work neither supports nor contradicts those numbers.

What it does speak to is narrower: nobody in that work publishes a per-script alignment
statistic. Their qualitative chunk examples are word-granularity on English, which this metric is
structurally blind to.

Two comparisons are worth stating carefully:

- Our runs settled at 4.4-5.1 realised bytes/chunk, i.e. **at the same compression as their
  4.57 BPE-matched operating point**, and Chinese is not aligned there. Their own pretokenizer
  discussion notes that hardcoded CJK ranges split characters by construction — so at matched
  compression, a hand-engineered prior does what the learned chunker does not, at this scale.
  That is a scale-bounded observation, not a refutation.
- Their Post II argues that pretokenization's role is to align tokens with whitespace-separated
  words and that "learning basic concepts such as words from bytes is a nontrivial problem". Our
  result is weak, small-scale support for the premise and no evidence on the conclusion: the
  chunker finds character structure in some scripts, and we do not show it learning words.

## Reproducing it

```
# one command, on Kaggle, CPU-or-GPU
kaggle kernels push -p kernel_q1     # kernel_q1/main.py is assembled by kernel_q1/build.py
```

- dataset: `mauricew/hnet-multiscript-bytes` (mount at
  `/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/corpus/<lang>/`)
- gates: `python tests/test_gates.py` — 8 gates, exits 0 on CPU with no compiled extensions
- oracle probe: `python tests/test_alignment_probe.py`
- eval noise: `python tests/eval_noise.py <ckpt> <corpus_dir>` — full-val recomputation + bootstrap
- replication report: `python tests/report.py`
- power analysis: `python tests/power_analysis.py`

`tests/test_gates.py` replaces the reference-implementation diff, which is not available here:
`main-horse/hnet-impl` needs a `mamba-ssm` git fork and `goombalab/hnet` needs
`flash_attn==2.8.0.post2`, and neither builds for sm_75. The substitute is an independent second
implementation of the router and the dechunk EMA plus a causal-mask check — a weaker gate, recorded
as such.

## Where this leaves the effort

Fifteen decision tickets, one destination: a defensible answer about what the chunker does, with a
stated falsifier, a replication that met its pre-registered criterion, and the failed readings
kept in the record. The next experiment is the causal width manipulation, and the artifact is the
harness and the corpus rather than the checkpoint — a 0.84 M-parameter model whose ratio control
never locked is a result, not a product.
