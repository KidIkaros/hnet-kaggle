# Space ablation: is the whitespace prior causal?

Type: task
Status: resolved
Blocked by: 13

## Question

Ticket 13 concluded that whitespace delimitation, not character width, decides whether the
chunker aligns to character starts. That was a cross-linguistic correlation with n=7 and every
other property varying too. Make it causal: delete the `0x20` bytes from Hindi and Russian, retrain
at the same N in the same run, and see whether alignment collapses.

## Answer

**It does not collapse. The whitespace mechanism is refuted, and the run also shows that
single-seed per-language numbers move further between runs than the effects being claimed.**

Kernel `mauricew/hnet-q1-ablate`, COMPLETE, seed 0, N=4, six languages in one run so every
contrast is within-run. Lift = last-3-eval mean.

| language | space rate | lift | bytes/chunk |
|---|---|---|---|
| hin | 0.076 | -0.037 | 3.52 |
| **hin_nospace** | **0.000** | **+0.081** | 3.60 |
| rus | 0.073 | +0.162 | 3.47 |
| **rus_nospace** | **0.000** | **+0.211** | 3.70 |
| cmn | 0.018 | -0.225 | 3.62 |
| jpn | 0.012 | -0.096 | 3.26 |

Deleting every space leaves Hindi and Russian at or slightly **above** their spaced versions. If
the router were anchoring on whitespace, both should have fallen toward the CJK numbers. Neither
did.

**The result that matters more than the ablation.** `hin` scored **+0.229** at N=4 in the
seven-language run (ticket 13) and **-0.037** here, with the only differences being the training
mixture and the presence of a near-duplicate language. `cmn` moved from -0.022 to -0.225 across
the same two runs. Single-seed per-language alignment is therefore **not reproducible across
mixtures**, and the magnitude of that instability (0.27 for hin) exceeds every effect this effort
has claimed.

**Caveat on this run specifically.** The mix contains two near-identical copies of Hindi and two of
Russian (with and without spaces), so 2/3 of every batch is duplicate content in a slightly
different form. That is a strange training distribution and is a plausible reason `hin` dropped.
The paired within-run contrasts (hin vs hin_nospace, rus vs rus_nospace) remain valid — they are
the same run — but the absolute levels here should not be compared to the other runs.

**What survives, stated as weakly as the evidence allows:**

- `rus` is positive at N=4 in all three runs that measured it (+0.301, +0.270, +0.162, +0.211).
- `cmn` is at or below zero in all three (-0.206, -0.022, -0.225).
- `jpn` is at or below zero in all three (-0.280, -0.023, -0.096) but with large spread.
- `hin` and `arb` are positive in some runs and near zero in others — **not established**.
- No mechanism is established. Width is out (ticket 13), whitespace is out (this ticket).

**Consequence: no script-level claim can be made from single-seed runs.** The next step is not
another mechanism hypothesis, it is a variance estimate: several seeds of one fixed config, so each
language gets a CI, and only languages whose CI excludes zero get reported.
