---
status: accepted
---

# We optimise the implementation before we spend the quota

Our pure-PyTorch H-Net runs at roughly 1e11 FLOPS on a T4, against the ~1e13 a tensor-core
implementation reaches — which is why a 1e16 FLOPs run costs a week instead of a day. Before the
comparable rung, the Hillis-Steele scan is replaced with a compiled chunked associative scan and
the main network uses SDPA.

The gate is **bit-exactness**: the optimised path must reproduce the current implementation's
outputs exactly, not merely within a tolerance. A looser gate would buy more speed, but the
current implementation is the reference for the architecture's semantics, and a tolerance-shaped
gate would let a subtle divergence hide behind it. The byte-exact gate also means the optimised
path can be trusted as the base for everything downstream.

Consequence: the speed work happens before, not during, the expensive run, and the existing
implementation stays in the tree as the reference the gate checks against.
