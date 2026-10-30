# Map: H-Net on Kaggle — from here to a verdict and an artifact

Label: `wayfinder:map` · Effort: `hnet-kaggle` · Charted 2026-10-29

## Destination

A Kaggle-only pipeline that answers **one defensible claim** about learned chunking on
multilingual bytes — then hardens the model into something reusable. Verdict first,
artifact second. "Done" = a result table with a stated falsifier and a working checkpoint
that trains, generates and reports chunk alignment on a single T4.

### Notes

- **All execution happens on Kaggle.** No local training runs, no local GPU dependency.
  Kaggle **CPU kernels do not consume GPU quota**, so build + gates + smoke runs go there
  for free; only real training touches the 30 h/week GPU allowance. Push latency (~3 min)
  is the cost, not capacity.
- Budget posture: **no deadline, do it properly.** 9.72 GPU-h left in the week ending
  2026-10-31, 30 h/week after. Quality over speed; do not shrink runs to fit a week.
- **Recency bias on data is wanted.** Prefer 2026 corpora; verify a dataset works before
  committing to it.
- Project root `~/Coding/kaggle/hnet`. Read `docs/RESEARCH.md`, `docs/ANATOMY.md`,
  `docs/PLAN.md` before any session — they hold the claim analysis, construction details,
  and the superseded v1 plan.
- Prior work to reuse, not rebuild: `~/Coding/kaggle/mbdfa` (fused Triton SSM scan, 5.53x,
  bit-exact; UTF-8 DFA state stream per byte; Kaggle CLI notes).
- Skills every session should consult: `kaggle-experiments` (and its
  `references/h-net.md`), `wayfinder`.
- Standing preferences: verdict before artifact; report nulls as nulls; never claim a
  compute multiplier we cannot measure.
- **Input encoding is UTF-8 for the primary** — every published H-Net result uses it and our
  alignment tooling is UTF-8-specific. Speed is *not* a reason to change it (UTF-8
  validation runs at 36 GB/s; the model eats 14 kB/s). Density is. Read `docs/ENCODING.md`.

## Decisions so far

- [Corpus: which datasets, how much, how packed](issues/01-corpus-selection.md) — epfml
  **FineWeb2-HQ-Plus** (Oct 2026) for deu/rus/cmn/jpn/arb + epfml **FineWeb-HQ** for eng;
  stream `text` only, 50 MB/language for Q1, 200 MB for Q2; pack to `.bin` once as a
  Kaggle dataset. Tatoeba, Ultra-FineWeb and fineweb-edu-translated rejected (reasons in
  the ticket).
- [Encoding: is learned chunking invariant to the byte representation](issues/11-encoding-invariance.md)
  — confirmed gap (UBE never tests a chunker); test is sound with 5 amendments — match on
  characters, dual bits-per-character/bpb metric, boundary agreement as the invariance
  statistic, offset-parity instead of a DFA, BPB Gini across scripts. UTF-16LE only.
- [Alignment metric and DFA stream](issues/07-alignment-metric.md) — **ΔF1 against a
  count-matched random set** (raw precision is gamed by boundary density on CJK); curve over
  checkpoints, not an endpoint; reuse `mbdfa`'s `state == 0` as the lead-byte test.
- [Pack the corpus as a Kaggle dataset](issues/05-pack-corpus.md) — `mauricew/hnet-multiscript-bytes`,
  297 MB train + 3 MB val across 6 languages, 54.5k docs; mount-verified (strict UTF-8,
  offsets on DFA state 0). English is a 2024 crawl and boilerplate-dirty.
- [Run Q1](issues/08-run-q1.md) — both seeds 5000 steps on T4, full artifacts. rus +0.27/+0.23
  and arb +0.38/+0.31 clear the bar; cmn -0.16/-0.13 and jpn -0.23/-0.00 don't. Ratio settled at
  ~4.5 bytes/chunk against a target of 6.
- [Verdict](issues/09-verdict.md) — **untested, with a mechanism inside it**: character structure
  is found on whitespace-delimited 2-byte scripts and lost on CJK. Q2 doesn't follow yet; the
  next run must control (N, character width), which was never swept.
- [Thai cell](issues/18-thai-cell.md) — **logographic explanation dead** (tha -0.145, 3/3
  negative). The run also failed its cmn control (-0.205 -> -0.037, sd 0.328), so it is reported
  as a harness failure first. Dominant finding: per-language alignment moves with the training
  mixture by more than the effect size (rus stable, cmn not).
- [Width manipulation](issues/17-width-manipulation.md) — **H1 DEAD.** English at Chinese's exact
  width aligns (+0.094, 4/4) while Chinese at 2.67 does not (-0.205, 4/4). The pre-registered
  falsifier missed by 0.006 on a magnitude threshold, but two of six checks failed. Width,
  whitespace, chunk period and ASCII density are all refuted as causes.
- [Write-up](issues/16-write-up.md) — `docs/WRITEUP.md`: claim, pre-registered falsifier, setup,
  results with CIs and sign tests, the four killed readings, limits, and what the published
  scaling claims can and cannot be told by this.
- [Seed replication](issues/15-seed-replication.md) — **H1 CONFIRMED.** 8 seeds, fixed mixture:
  rus +0.178 and arb +0.266 (8/8 positive, p=0.004), cmn -0.147 (7/8 negative), jpn unresolved
  (p=0.14), eng/deu structural zero. Kill criterion (rus/cmn overlap) not met. Seed noise
  dominates eval noise.
- [Artifact scope](issues/10-artifact-scope.md) — the artifact is the **harness and the corpus**,
  not the checkpoint: packaged repo with 8 green gates, public dataset, one-command reproduction
  kernel. DFA-constrained decoding deferred (it would change the measured boundary distribution).
- [Space ablation](issues/14-space-ablation.md) — **whitespace mechanism refuted.** Deleting every
  space left hin (-0.037 -> +0.081) and rus (+0.162 -> +0.211) at or above their spaced versions.
  Also: hin moved 0.27 between runs, so single-seed per-language numbers are not reproducible.
- [Width vs word-delimitation](issues/13-width-vs-delimitation.md) — **superseded by the space
  ablation.** Reported whitespace, not width.
  Hindi (2.53 B/char, 0.192 space) is positive at N=4 and N=6; cmn/jpn (2.67 B/char, 0.02-0.04
  space) are not. Per-language numbers are mixture-sensitive: adding hin moved cmn N=4 from
  -0.206 to -0.022.
- [Period-matched N sweep](issues/12-period-matched-n.md) — **hypothesis refuted.** CJK is negative
  at every N and worst at exactly one character per chunk; rus/arb positive at every N. The split
  is by script, not by compression target. Width vs word-delimitation remains confounded.
- [Experiment config](issues/06-experiment-config.md) — Q1 at **0.84 M params, N=6, 5000
  steps, 2 seeds, hnet arm only**. 8 M costs 15.5 h/run vs 9.72 h left; a full corpus pass
  costs 28 h, so the design is steps-bound (~7% of an epoch) and that is declared, not hidden.
- [Gates on Kaggle](issues/04-gates-on-kaggle.md) — all five gates CPU-runnable and green in
  kernel `mauricew/hnet-gates`; reference diff replaced by independent router/dechunk
  implementations (weakened, recorded). Pre-registered "chunker engaged" definition: ratio
  rises then bounds, lift > +0.15 on rus/jpn/arb, cmn read as trend, eng by bpb only.
- [Kaggle-only dev loop](issues/03-kaggle-dev-loop.md) — path works end to end on a free CPU
  kernel (range-read parquet -> pack -> validate -> train -> artifacts); corpus validates
  clean except dirty `eng` boilerplate and a 2024-only English dump. Three metric bugs found
  and gated. Corpus is kernel output only, not yet a mountable dataset.
- [The claim and its falsifier](issues/02-the-claim.md) — boundaries align with UTF-8
  character starts above chance **and** survive re-encoding to UTF-16LE. Two-stage
  falsifier; failure is reported as a statement about 8M params, not about H-Net.

## Frontier

- **Thai confirmation seeds (wayfinder:task)** — seeds 3-4 of the identical 4-language mixture:
  tightens Thai (n=5) and re-tests the `cmn` control that drifted outside tolerance.
- **The causal question, now well-posed (wayfinder:task)** — what separates cmn/jpn/tha from
  rus/arb/eng/hin, given that width, whitespace, ASCII density, script type and chunk period have
  all been refuted. Mixture composition is now a first-class variable and must be controlled.

## Not yet specified

- Whether **Q2 (matched-loss cross-script transfer)** is worth running. It does not follow from
  Q1: the chunker is aligned on 2 of 5 measurable scripts, so a transfer comparison would be
  confounded by whether the comparison weighs CJK. It needs its own pre-registered falsifier and
  its own corpus version at 200 MB/language.
- Whether the **NLL-vs-entropy boundary rule** head-to-head earns a slot — Gu's argument that
  BLT/DPT segment on the wrong quantity. A second claim; stays out unless there is room.
- **FLEXITOKENS** (Owodunni et al., 2026) — learnable boundary predictors on ByT5 to reduce
  over-fragmentation. Read before any novelty claim; may already answer part of the router
  question.
- Whether the outer networks need the **fused Triton scan** at 8M params, or a plain torch SSM
  is fast enough on a T4 to skip the port.
- **Code as a domain** (the paper's strongest multiplier claim) — needs The Stack v2, which is
  gated.
- Where the writeup lives and whether it is public. `docs/WRITEUP.md` is written; publishing it
  and the dataset is a decision for the human, not for this map.

## Out of scope

Ruled beyond this destination; they do not graduate.

- Reproducing the **1.28x / 2.06x compute multipliers** — crossover is at 100B bytes for
  1-stage and our whole quota buys ~1.6GB; the per-scale gap is 0.755 vs 0.756 bpb.
- **Downstream benchmark suites** (OLMES, Paloma) — not resolvable at this scale.
- **>10B training bytes**, and any model at ≥100M params.
- **2-stage / recursive H-Nets** — 1-stage only, until 1-stage is understood.
- **Shipping the checkpoint as the artifact** — 0.84 M params, unlocked ratio control,
  mixture-sensitive per-language behaviour. The harness and the corpus are the artifact.

