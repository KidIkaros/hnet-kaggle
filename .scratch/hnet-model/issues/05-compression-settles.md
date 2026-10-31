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
- [ ] F and G both reach within a stated tolerance of 1/N and remain there across the final quarter of the run.
- [ ] The ratio loss reaches within a stated tolerance of its minimum of 1, and stays there.
- [ ] Validation bpb on fixed validation windows is monotone across the run. It already is, in every run we have — the logged per-batch loss is not this metric and moves with each batch's language mix, so the two must not be confused.
- [ ] Realised bytes per chunk sits at the target rather than under or over it.
- [ ] The result is reported as a comparison against the current best run's bpb, with the tolerance stated in advance.
- [ ] The reading is written into the write-up's model section, not only into a ticket.
- [ ] A ratio loss sitting near 1.05 is read as "not converged", not "broken": the published diagnostic band for a failing run is substantially above 1, and the lower bound for a run shutting the main network down is 2(N-1)/N, which at N=6 is 1.667. Our band is benign, and the write-up says so rather than dramatising it.
