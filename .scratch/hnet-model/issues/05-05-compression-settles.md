# 05 — The compression settles

**What to build:** **What to build:** A training run in which the chunker's compression reaches its target and stays
there, and the loss descends monotonically once it has. This is the ticket the whole artifact
rests on.

**Blocked by:** 04, and the results of the three small-scale probes already running

**Status:** ready-for-agent

- [ ] F and G both reach within a stated tolerance of 1/N and remain there across the final quarter of the run.
- [ ] The ratio loss reaches within a stated tolerance of its minimum of 1, and stays there.
- [ ] Validation bpb shows no rise across the final quarter of the run.
- [ ] Realised bytes per chunk sits at the target rather than under or over it.
- [ ] The result is reported as a comparison against the current best run's bpb, with the tolerance stated in advance.
- [ ] The reading is written into the write-up's model section, not only into a ticket.
