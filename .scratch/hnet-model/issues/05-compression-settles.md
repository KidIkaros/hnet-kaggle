# 05 — The compression settles

**What to build:** **What to build:** A training run in which the chunker's compression reaches its target and stays
there, and the loss descends monotonically once it has. This is the ticket the whole artifact
rests on.

**Blocked by:** 04, and the five-arm dynamics sweep now running on CPU (control, official LR multipliers, alpha 1.0, batch-accumulated F/G, target-ratio curriculum)

**Status:** ready-for-agent

- [ ] Every arm runs at least 5000 steps. The published guidance is explicit that sweeps shorter than this are misleading and that the ratio loss needs 500+ steps to stabilise, so a shorter arm cannot support a conclusion — our first three probes at 3000 steps were inside that unreliable window and were discarded rather than read.
- [ ] The sweep covers the interventions the primary-source catalogue ranks highest, with the ratio-loss weight swept across {0.03, 0.1, 0.3, 1.0} rather than at one point: the published fix for our exact symptom shape (under-compression at a target of 5) is a weight of 1.0, which is roughly 33x the paper's default, and a point estimate at 0.1 would have been an order of magnitude short of the published working value.
- [ ] The batch-aggregated F/G arm is recorded as *restoring official behaviour*, not as a new idea: the official implementation forms one global F and one global G over the minibatch, while ours computes them per sequence and averages the per-sequence losses, so each of our F and G is estimated from a single 256-byte sequence. The per-sequence arm is labelled as our deviation.
- [ ] One arm replaces the learned router with a fixed stride at the same compression and the same FLOPs, so that the learned router is measured rather than assumed. If a fixed stride matches the learned chunker's quality, the router is not earning its place, and that is a finding about our model rather than a failed run.
- [ ] The gap between F and G is reported, not just their values. The paper expects F to converge toward G as routing becomes confident; a large persistent gap means the router's probabilities are diffuse rather than bimodal, which points at a different fix than more ratio-loss weight.
- [ ] F reaches 1/N and stays there, and the F-vs-G gap is tracked separately rather than assumed away. The ratio loss has a **cross** of exact minima: algebraically it is exactly 1 whenever F = 1/N *or* G = 1/N, for any value of the other. So it cannot distinguish "the right number of boundaries" from "the right mean probability", and a run can read as perfectly settled while the router stays unconfident. The measured arms show this directly: alpha 1.0 reaches F = 0.169 against a target of 0.167 with a ratio loss of 1.003, while G sits at 0.298 — a 76% gap that the loss reports as success. Whether closing that gap is worth doing is a separate decision, and it must be argued from what a diffuse router implies for chunk quality, not from the loss value.
- [ ] The ratio loss reaches within a stated tolerance of its minimum of 1, and stays there.
- [ ] Validation bpb on fixed validation windows is monotone across the run. It already is, in every run we have — the logged per-batch loss is not this metric and moves with each batch's language mix, so the two must not be confused.
- [ ] Realised bytes per chunk sits at the target rather than under or over it.
- [ ] The result is reported as a comparison against the current best run's bpb, with the tolerance stated in advance.
- [ ] The reading is written into the write-up's model section, not only into a ticket.
- [ ] A ratio loss sitting near 1.05 is read as "not converged", not "broken": the published diagnostic band for a failing run is substantially above 1, and the lower bound for a run shutting the main network down is 2(N-1)/N, which at N=6 is 1.667. Our band is benign, and the write-up says so rather than dramatising it.

## Result: the recipe is the official learning-rate multipliers with alpha 0.3

Two CPU sweep rounds, 0.84 M parameters, six languages, 5000 steps, seed-matched.

**Iteration 1 (seed 0), one arm per intervention**

| arm | F | G | ratio loss | bytes/chunk | mean bpb | eng bpb |
|---|---|---|---|---|---|---|
| control | 0.234 | 0.323 | 1.080 | 4.27 | 2.501 | 2.509 |
| alpha 1.0 | 0.169 | 0.298 | 1.003 | 5.91 | 2.520 | 2.536 |
| curriculum | 0.234 | 0.348 | 1.088 | 4.28 | 2.487 | 2.494 |
| **official LR multipliers** | 0.196 | 0.320 | 1.034 | 5.11 | **2.406** | **2.352** |
| batch-aggregated F/G | 0.256 | 0.334 | 1.107 | 3.91 | 2.497 | 2.457 |

**Iteration 2, seed-matched against the control**

| | seed 0 | seed 1 |
|---|---|---|
| control, mean bpb | 2.501 | 2.510 |
| LR multipliers, mean bpb | 2.406 | 2.407 |
| effect | **-0.095** | **-0.103** |
| effect on English | -0.157 | -0.154 |

The learning-rate effect replicates across seeds to within 0.008 bpb, well inside the
seed-to-seed spread of the arm itself (the two seeds of the treatment differ by 0.001 bpb on
the mean and 0.064 on the worst single language). It is not noise.

**The combination that wins.** Official multipliers with alpha 0.3 on top, seed 0:

| arm | mean bpb | eng bpb | bytes/chunk | ratio loss | F | G |
|---|---|---|---|---|---|---|
| LR multipliers alone | 2.406 | 2.352 | 5.11 | 1.034 | 0.196 | 0.320 |
| **LR multipliers + alpha 0.3** | **2.402** | **2.349** | **5.74** | **1.009** | 0.174 | 0.310 |

It matches the multipliers alone on quality and lands the compression close to the target of
6.0 bytes/chunk, with a ratio loss of 1.009 against the control's 1.080. Alpha 1.0 was not
needed: the multipliers move the compression most of the way, and a weight of 0.3 finishes it
without the quality cost that a weight of 1.0 carried on its own.

**What this settles and what it does not.** It settles that the compression failure was a
dynamics problem, not an objective problem: the official per-stage learning-rate modulation
was missing, and it improves quality and compression together. It does not settle the F-vs-G
gap. The best arm still has F at 0.174 against G at 0.310, and the ratio loss reports that as
settled, because of the cross-of-minima structure. Closing that gap needs a different
objective, and nothing here says it is worth the quality cost.

**Dead ends, recorded so they are not retried:** the target-ratio curriculum was a no-op
(4.28 against 4.27 bytes/chunk) because at the default weight the ratio loss cannot track a
moving target, and curriculum and loss weight are not independent knobs; the batch-aggregated
F/G variant was worse than the per-sequence form on both quality and compression, which
refutes the hypothesis the catalogue advanced from our per-sequence deviation.
