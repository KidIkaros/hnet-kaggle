# 04 — Port the reference implementation's semantics

**What to build:** **What to build:** Our model behaves like the reference implementation wherever we were wrong —
starting with the per-stage learning-rate multipliers, which our implementation never had, plus
whatever the research catalogue of published interventions turns up.

**Blocked by:** None — can start immediately (the catalogue of published interventions is in flight; see the research note)

**Status:** ready-for-agent

- [ ] Per-stage learning-rate multipliers are applied as the reference specifies: 2.0 for the outer networks, 1.5 for the main network, with the main network's multiplier applied again on top.
- [ ] Every semantic difference found between our model and the reference is recorded, and the reference's behaviour is taken as correct.
- [ ] A gate compares our forward pass against the reference implementation's on the same input, within the tolerance the existing reference-diff pattern uses.
- [ ] Each intervention the catalogue recommends is either implemented or explicitly deferred, with a reason.
- [ ] The three small-scale probes already running are read for what they show about the multipliers and the ratio-loss weight, and their results are recorded.
