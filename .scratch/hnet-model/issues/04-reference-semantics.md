# 04 — Port the reference implementation's semantics

**What to build:** **What to build:** Our model behaves like the reference implementation wherever we were wrong —
starting with the per-stage learning-rate multipliers, which our implementation never had, plus
whatever the research catalogue of published interventions turns up.

**Blocked by:** None — can start immediately (the catalogue of published interventions is in flight; see the research note)

**Status:** ready-for-agent

- [ ] Per-stage learning-rate multipliers are applied as the official code applies them, not as a paraphrase: the outer stage's multiplier (2.0) goes on **every** parameter of that stage — including the main network, which is a submodule of it — and the main network's own multiplier (1.5) is then applied **on top**, so the main network's effective multiplier is 3.0. The official code's own comment says this; our first implementation used 1.5 for the main network and was wrong by a factor of two.
- [ ] Every semantic difference found between our model and the reference is recorded, and the reference's behaviour is taken as correct.
- [ ] A gate compares our forward pass against the reference implementation's on the same input, within the tolerance the existing reference-diff pattern uses.
- [ ] Each intervention the catalogue recommends is either implemented or explicitly deferred, with a reason.
- [ ] Two setup questions the sources imply are answered explicitly: whether our smoothing is chunk-level or byte-level (the paper's ablation says removing the smoothing makes compression fluctuate severely), and what our step-0 compression ratio was (a bad initial ratio is reported to rarely recover).
- [ ] The catalogue's dead ends are recorded where the work could be tempted to repeat them: no hardcoded or kth-value-forced compression ratio, no adaptive alpha, no boundary noise or random-flip sampling, and a compression spike is not by itself a broken run.
- [ ] The catalogue file is the reference for this ticket's decisions, and each choice made here cites the entry it came from.
- [ ] The three small-scale probes already running are read for what they show about the multipliers and the ratio-loss weight, and their results are recorded.
