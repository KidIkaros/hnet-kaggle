# 06 — Escalation: the reference parameterisation work

**What to build:** **What to build:** If three iterations of our own dynamics fixes fail, the alternative route — the
reference implementation's parameterisation work — is ready to execute rather than improvised under
pressure. This ticket stays open until the settling ticket succeeds.

**Blocked by:** 05, and only executed if 05 has failed three times

**Status:** ready-for-agent

- [ ] The trigger is explicit: three iterations of fix, train, evaluate that have not beaten our current bpb.
- [ ] The reference's parameterisation findings are read, and the applicable ones are listed with their reported effects.
- [ ] The escalation run is specified before it is launched, with its own pass criterion.
- [ ] The escalation is reported either way — including if it also fails.
