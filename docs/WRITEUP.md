# Where does a byte-level H-Net's chunker actually put boundaries?

**A small-scale, replicated measurement of character alignment in learned chunking, and four
interpretations that the data killed along the way.**

## The claim

At 0.84 M parameters, trained 5000 steps on raw bytes with a 1-stage H-Net, the chunker places
boundaries on UTF-8 **character starts** above chance in Arabic and Russian, and below chance in
Chinese. Japanese is unresolved. English and German are structurally unmeasurable on this metric.

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

Eight seeds, fixed 6-language mixture, N=4, every seed 5000 steps. Lift is the mean of the last
three evals per seed; the CI is across seeds.

| script | bytes/char | mean lift | seed SD | 95% CI | sign test | verdict |
|---|---|---|---|---|---|---|
| eng | 1.00 | -0.000 | 0.002 | [-0.002, +0.001] | 4/8 positive | not aligned (structural) |
| deu | 1.02 | +0.002 | 0.014 | [-0.008, +0.011] | 6/8 positive | not aligned (structural) |
| **rus** | 1.80 | **+0.178** | 0.085 | **[+0.119, +0.236]** | **8/8 positive, p=0.004** | **aligned** |
| cmn | 2.67 | **-0.147** | 0.113 | [-0.225, -0.068] | 7/8 negative, p=0.035 | **not aligned** |
| jpn | 2.68 | -0.123 | 0.169 | [-0.240, -0.006] | 6/8 negative, p=0.14 | **unresolved** |
| **arb** | 1.78 | **+0.266** | 0.091 | **[+0.203, +0.330]** | **8/8 positive, p=0.004** | **aligned** |

Arabic and Russian are positive in every seed; Chinese is negative in seven of eight. The two
groups' CIs do not overlap. Japanese is reported as **unresolved**: the pre-registered CI rule
called it "not aligned" by a hair, but the distribution-free sign test gives p=0.14 and two of
eight seeds came back positive (+0.111, +0.173). The conservative reading wins.

**The split is by script, and its cause is not identified.** Bytes per character was the obvious
candidate and it is wrong: an English corpus widened to 2.69 bytes/char — Chinese's exact width,
same script, same words, same spaces — aligns (+0.094, every seed), while Chinese at 2.67 does
not (-0.205, every seed). Width, whitespace delimitation, chunk-length period and ASCII-landmark
density have each been tested and each failed to explain it. The two groups' CIs still separate
cleanly, so the *effect* is solid; only its cause has escaped every hypothesis thrown at it.

## What we killed

Five readings of this evidence were wrong. All five are recorded because the final claim is only
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

What survived every test is the **effect** — Russian and Arabic align, 8/8 seeds, p = 0.004;
Chinese does not — and not its cause. Four candidate mechanisms have now been tested and refuted:
chunk-length period, whitespace delimitation, bytes per character, and ASCII-landmark density
(`rus_comb` and `cmn` have nearly identical ASCII shares, 0.058 and 0.062, yet +0.075 vs -0.205).

**Still open: the mechanism, and it has resisted four tests.** Width was the best predictor
until it was manipulated directly and failed. The next candidate is the script type itself:
Chinese and Japanese are the only logographic scripts in the set. Thai is the cell that decides
it — alphabetic, wider than Chinese, unspaced, and with a lower ASCII share than Chinese — and it
is running now. If Thai aligns, the effect is specific to logographic scripts; if it does not, the
honest answer is a two-factor description with no mechanism, and that the cause is unidentified
at this scale.

## Limits

- **0.84 M parameters, ~7% of one epoch, one T4.** A full pass over the corpus would take 28 h at
  this size; the design is steps-bound, not pass-bound, because ratio dynamics need 5000 steps to
  stabilise.
- **Per-language numbers are mixture-sensitive.** Every number here comes from one fixed mixture;
  cross-run comparisons of the same language at different mixtures are not valid, and the write-up
  should report composition with every figure.
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
