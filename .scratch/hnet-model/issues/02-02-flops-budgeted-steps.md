# 02 — FLOPs-budgeted step counts

**What to build:** **What to build:** A run can be specified by an allowance of compute rather than by a step count.
The accounting converts that allowance into a step count, and the accounting is itself checked
against measured wall-clock.

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] A config expresses a FLOPs allowance and the training loop derives its step count from it.
- [ ] The analytic FLOPs-per-byte figure agrees with measured wall-clock within the existing parity tolerance, across at least three model shapes.
- [ ] Byte-level, chunk-level and attention costs are each charged at the resolution they actually run at.
- [ ] The derived step count is recorded in the run's own output, next to the metrics.
- [ ] A run specified by a budget and the same run specified by its step count produce the same schedule.
