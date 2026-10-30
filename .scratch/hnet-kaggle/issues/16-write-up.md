# Write-up

Type: task
Status: resolved
Blocked by: 15, 10

## Question

Write the result up so that a reader can check it, and so that the failed readings are part of
the record rather than omitted.

## Answer

`docs/WRITEUP.md` (10 KB), covering: the claim and its pre-registered falsifier; the setup
(0.84 M params, 50 MB/language, 5000 steps, 8 seeds, one T4); the metric and its two structural
properties (English/German unmeasurable; eval noise 5-10x smaller than seed noise); the results
table with per-seed SD, CI and sign test; the four killed readings (probe off-by-one, chunk-period
hypothesis, whitespace hypothesis, single-seed Hindi); the limits (mixture sensitivity, unlocked
ratio control, unreplicated Hindi); the relation to the published scaling claims, stated as what
this does and does not bear on; and reproduction commands for the dataset, the 8 gates, the oracle
probe, the eval-noise harness and the replication report.

The write-up keeps the mechanism open rather than asserting one: bytes-per-character is the best
predictor, but the only causal test we ran (whitespace) came out null, and Hindi is unreplicated.
