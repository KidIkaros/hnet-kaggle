# 09 — Byte-scored multiple-choice evaluation

**What to build:** **What to build:** An evaluation that answers "is this model usable" beyond bpb — multiple-choice
tasks scored by byte likelihood, which a tokenizer-free model can do natively.

**Blocked by:** 08

**Status:** ready-for-agent

- [ ] The harness scores a multiple-choice task by summing byte log-likelihoods over each candidate and taking the argmax.
- [ ] The metric is audited against a synthetic oracle before any result is reported, in the pattern the other metrics here used.
- [ ] Results are reported for the shipped checkpoint alongside the published bpb.
- [ ] The harness runs on CPU against a stored checkpoint.
- [ ] The task set is versioned, so a later checkpoint is scored against the same tasks.
