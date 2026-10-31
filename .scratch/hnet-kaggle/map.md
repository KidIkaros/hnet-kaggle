# Map: H-Net on Kaggle — from here to a verdict and an artifact

Label: `wayfinder:map` · Effort: `hnet-kaggle` · Charted 2026-10-29

## Destination

A Kaggle-only pipeline that tests **Cartesia's efficiency claim** — the compute multiplier of a
byte-level H-Net against a fixed-tokenizer baseline, and whether that advantage grows as the
tokenizer degrades on non-Latin scripts — with the mechanism work already banked serving as the
prior and as a source of falsifiable predictions.

**The mechanism work is not the point; it is the setup.** The user's standing correction (2026-10-31):
building the H-Net was supposed to test the efficiency claim, and the effort spent its first weeks
characterising the chunker instead. What we have from that is real and published (a byte-phase
finding in three modalities), but it cannot substitute for the comparison that was always meant to
happen. The baselines — a BPE Transformer and a byte Transformer — did not exist until now, and
without them no claim about efficiency can be made at all.

**Standing reading of the papers** (from `docs/RESEARCH.md`, 2026-10-29): Cartesia's 1.28x
compute multiplier is measured at 1e19-1e21 FLOPs with a protocol an independent replication
criticised on four counts; that replication found no H-Net compute advantage at 1e16-1e18. The
replication is English-only, and the untested question that a small budget can still address is
whether the H-Net's advantage grows as the fixed tokenizer degrades — largest in Chinese, Arabic
and Russian.

**The claim has moved twice and is now modality-general.** It started as "the chunker learns
characters"; the invariance test killed that; the byte-phase analysis replaced it with "learned
chunking is a byte-phase segmenter, and apparent character alignment is a phase coincidence that
reverses under re-encoding". That question — *does learned structure track the semantics of the
data or the layout of its encoding?* — is not about text, which is what the modality expansion
below exists to test.

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
- [Encoding invariance](issues/19-encoding-invariance-test.md) — **stage two of the pre-registered
  claim FAILS.** In UTF-16LE the model puts boundaries *inside* characters for exactly the scripts
  where UTF-8 aligned (rus -0.228, arb -0.326), and the cross-script ordering reverses. What we
  measured is a property of UTF-8's byte layout, not of characters.
- [Byte-phase analysis](issues/21-byte-phase.md) — **the positive claim.** Boundaries sit at a
  byte phase that is either on character starts or one byte inside them, and which one depends on
  the script AND the encoding (Russian/Arabic on-starts in UTF-8, one-byte-in in UTF-16; Chinese
  the reverse). Zero-training transfer keeps the phase and loses the chunk length. Learned
  chunking is a byte-phase segmenter.
- [Scale ladder](issues/22-scale-ladder.md) — with characters refuted, measure what the
  chunker does learn: offset-phase concentration and zero-training encoding transfer.
- [Scale ladder](issues/22-scale-ladder.md) — is the invariance failure a capacity artifact? 2.6 M
  parameters, UTF-8/UTF-16 twins.
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
- [Seed replication](issues/15-seed-replication.md) — **n=10.** rus +0.169 and arb +0.250 (0/10
  seeds negative each), cmn -0.165 (9/10) and jpn -0.155 (8/10). All pre-registered checks pass;
  the kill criterion (rus/cmn overlap) still not met. Pooled: rus/arb positive in 29/29 seeds. — **H1 CONFIRMED.** 8 seeds, fixed mixture:
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

- **The efficiency comparison (wayfinder:task, GPU+CPU)** — three arms on the same bytes:
  BPE Transformer (the incumbent), byte Transformer (no hierarchy), byte H-Net (ours). Common
  currency is bpb; FLOPs/byte from `src/flops.py` and `src/baseline.py`. Languages eng (control),
  cmn and arb (where a fixed tokenizer is worst). Two rungs so there is a curve, not a point.
  *Prediction from our own mechanism work:* H-Net ties the byte Transformer on Chinese, and any
  H-Net-vs-BPE advantage there comes from BPE's degradation rather than from the chunker learning
  anything. If H-Net beats the byte Transformer on Chinese, the byte-phase finding is wrong.
- **BPE tokenizers (wayfinder:task, CPU)** — in-corpus byte-level BPE per language, 16k vocab;
  `src/bpe.py`.

- **Scale ladder — DONE** ([ticket 22](issues/22-scale-ladder.md)): 2.43 M parameters, UTF-8 and
  UTF-16 arms. **Not one sign flips**; the cross-encoding reversal survives at 2.9x scale, so the
  byte phase is not a small-model artifact. Next rung (8 M) only if a reviewer asks.
- **Vision I — DONE** ([ticket 23](issues/23-vision-bytes.md)): no edge or object alignment
  (+0.017 / +0.025), pixel/row phase enriched, and the phase moves when channels are permuted
  (0.466 -> 0.566); rgb vs bgr agree at chance (0.369 vs 0.364). Remaining: `planar`, `gray`.
- **Audio — DONE** ([ticket 25](issues/25-audio-bytes.md)): chunk duration varies 2.9x at a fixed
  byte budget, onsets unaligned, and LE vs BE agree at 0.036 vs chance 0.114 but 0.217 when
  shifted one byte. Remaining: `f32le` (OOM at step 100).
- **Vision II, 2D H-Net (wayfinder:task, CPU, most engineering)** — patch chunking against known
  object masks; the most directly comparable item to Cartesia's published vision claim.
- **Code (wayfinder:task, CPU, licensing-dependent)** — syntax alignment in the domain where the
  structure is most explicit.
- **Dataset scouting (wayfinder:task, DONE)** — live-checked candidates for all four arms with
  recency, licence, format and a byte-range probe where format mattered; see
  [ticket 27](issues/27-dataset-scouting.md). Audio starts on `espnet/yodas3` (2026-10-07),
  Vision I on `athkot/coco-20-synthetic-50k` (2026-10-30), Vision II on a synthetic generator.
- **BLT-like arm (wayfinder:task, GPU, later)** — the entropy-heuristic boundary rule in the same
  architecture, for a comparison against Cartesia's published claim.

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

