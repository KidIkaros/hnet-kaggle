---
status: accepted
---

# We optimise around the byte phase rather than trying to fix it

Our router settles on a byte phase rather than on content. For now we accept that and optimise
the model around it, because the published comparison is against a learned chunker rather than
against a content-aligned one — the phase is not automatically disqualifying, and the
intervention needed to change it would put the FLOPs-matched comparison at risk.

Consequence: the model work does not depend on making chunking semantic, and a content-alignment
intervention stays available later without blocking the artifact. A future reader looking at a
model whose boundaries follow byte layout should know that was a choice, not an oversight.
