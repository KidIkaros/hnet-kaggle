---
status: accepted
---

# Fix the compression dynamics before spending scale

The next block of compute goes to making the compression settle at small scale, and only then
to scaling the winner. Scaling first was rejected because the replication's own account says
chunking behaviour is not scale-consistent, and because our 2.4 M-parameter runs sit around
1e14 FLOPs — two orders of magnitude below the smallest rung in any published comparison — so a
scale run would mostly measure the wrong thing at great cost.

Consequence: the three small-scale probes run first and their outcome decides whether scale
happens at all, rather than scale being the default.
