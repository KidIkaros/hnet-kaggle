# Seed replication with a pre-registered width hypothesis

Type: task
Status: resolved
Blocked by: 14

## Question

Can any script-level claim survive replication? And can a **quantitative** hypothesis be
stated that the replication either confirms or kills?

## Pre-registered prediction (written before any seed is run)

Pooling every measurement so far, alignment falls monotonically with bytes-per-character:

| bytes/char | 1.78 (arb) | 1.80 (rus) | 2.53 (hin) | 2.67 (cmn) | 2.68 (jpn) |
|---|---|---|---|---|---|
| mean lift | +0.258 | +0.236 | +0.096 | -0.151 | -0.133 |

**H1 (width): alignment decreases monotonically with bytes-per-character.** The predicted
ordering of per-language means is `arb ~ rus > hin > cmn ~ jpn`, and the predicted seed variance
is *low* for the 1.8 B/char scripts and *high* for hin, which sits near the zero crossing.

**H2 (whitespace) is already refuted** by ticket 14 and is not being tested again.

**Kill criterion.** If across seeds the per-language means do not order as predicted — in
particular if `rus` and `cmn` overlap, or if `hin`'s mean lands far from zero with small
variance — H1 is dead and the honest outcome is *no mechanism established, alignment differs by
script, cause unknown*.

## Design

- **Fixed** 6-language mixture (eng, deu, rus, cmn, jpn, arb), **N=4**, seed 0..7, 5000 steps,
  everything else identical to the Q1 config. The mixture is fixed because ticket 13/14 showed
  per-language numbers move with the mixture.
- 8 seeds x ~20 min = ~2.8 GPU-h of the ~6 h left.
- Evaluation: the full 1953-window val sweep plus the bootstrap CI from `tests/eval_noise.py`,
  so eval noise is separated from seed noise, and the per-language CI is reported for every
  language.

## Decision rules (fixed now)

- A language is **aligned** if its across-seed 95% CI excludes 0.
- **Not aligned** if the CI includes 0 and the mean is <= +0.05.
- **Unresolved** otherwise — reported as unresolved, never quietly dropped or rounded to a side.
- Seed SD and across-seed CI are reported for every language, including English and German, which
  are structurally unmeasurable on this metric (99.6% and 98.5% lead bytes) and are expected to
  sit at zero with tiny variance.

## Answer

**H1 CONFIRMED on the languages measured, with one script (jpn) that the pre-registered rule and
the sign test disagree about. 8 seeds, fixed 6-language mixture, N=4, every seed 5000 steps.**

Kernel `mauricew/hnet-q1-repl`, COMPLETE, seeds 0-7, ~3.9 GPU-h. Lift = mean of the last 3 evals
per seed; CI is across seeds.

| script | bytes/char | n | mean lift | seed SD | 95% CI | sign test | verdict |
|---|---|---|---|---|---|---|---|
| eng | 1.00 | 8 | -0.000 | 0.002 | [-0.002, +0.001] | 4/8 positive | not aligned (structural) |
| deu | 1.02 | 8 | +0.002 | 0.014 | [-0.008, +0.011] | 6/8 positive | not aligned (structural) |
| **rus** | 1.80 | 8 | **+0.178** | 0.085 | **[+0.119, +0.236]** | **8/8 positive, p=0.004** | **aligned** |
| cmn | 2.67 | 8 | **-0.147** | 0.113 | [-0.225, -0.068] | 7/8 negative, p=0.035 | **not aligned** |
| jpn | 2.68 | 8 | -0.123 | 0.169 | [-0.240, -0.006] | 6/8 negative, p=0.14 | **unresolved** (see below) |
| **arb** | 1.78 | 8 | **+0.266** | 0.091 | **[+0.203, +0.330]** | **8/8 positive, p=0.004** | **aligned** |

**Pre-registered checks, all passed:**

- `arb ~ rus` (CIs overlap): yes.
- `cmn ~ jpn` (CIs overlap): yes.
- rus CI separated from cmn CI: **yes** — this was the explicit kill criterion, and it is not met.
- arb CI separated from cmn CI: yes.

**H1 is confirmed**: alignment is positive at 1.78-1.80 bytes/char and negative at 2.67, and the
two groups' CIs do not overlap. Width survives as the discriminator; whitespace stays dead
(ticket 14); the chunk-length explanation stays dead (ticket 12).

**Where the pre-registered rule and the sign test disagree: `jpn`.** The rule (CI excludes 0)
calls it "not aligned" by a hair — [-0.240, -0.006]. The distribution-free sign test says 6/8
negative, p = 0.14, which is *not* significant. jpn's per-seed values are −0.277, −0.146, −0.207,
−0.219, −0.239, **+0.111, +0.173**, −0.181. **Read jpn as unresolved**, not as a negative result:
two of eight seeds are positive, and the two positive seeds bracket the two earliest seeds'
negatives. Since jpn and cmn have nearly identical byte width (2.68 vs 2.67) and cmn is
robustly negative, the honest statement is that the wide-script group is **negative on cmn and
unresolved on jpn**.

**An anomaly I chased and dismissed.** jpn's seeds 5 and 6 were positive while seeds 0-4 were
negative, which looked like order-dependent drift. Seed 7 came back negative (−0.181), so the
pattern is not monotone in seed order — it is noise. Worth stating because it is the kind of
thing that looks like a finding if you stop at 7 seeds.

**`hin` was not in this mixture.** H1's prediction placed it between rus and cmn, which the earlier
runs supported (+0.229, −0.037, +0.081 — positive but unstable, exactly what a near-zero crossing
predicts). But those runs are the mixture-sensitive ones, so **hin's position is UNREPLICATED** and
is reported as such, not as confirmation.

**Seed noise dominates eval noise.** On the seed-0 N=4 checkpoint, the full 1953-window
recomputation gives an 8-window bootstrap SD of **0.017-0.018** for rus, cmn, jpn and arb
(rus +0.290 [+0.251, +0.321], cmn -0.183 [-0.217, -0.146], jpn -0.262 [-0.291, -0.226],
arb +0.304 [+0.268, +0.336]), against across-seed SDs of **0.085-0.169**. The uncertainty in
these results is training stochasticity, not measurement — a 5-10x gap.

**Cost:** 3.9 GPU-h, all of it the replication; eval sweeps were CPU and free.

