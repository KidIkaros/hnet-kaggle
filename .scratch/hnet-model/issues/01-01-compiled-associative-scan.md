# 01 — Compiled chunked associative scan

**What to build:** **What to build:** The same model, but the scan that carries state through the encoder and decoder
runs as a compiled chunked associative scan instead of a Python Hillis-Steele loop, so a training
step costs materially less wall-clock. Behaviour is unchanged: the outputs are identical, not
merely close.

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] The optimised path reproduces the current implementation's outputs bit-exactly on a fixed batch of inputs — logits, boundary mask, chunk sequence and loss alike.
- [ ] The existing gate suite still passes, unchanged.
- [ ] Measured step time on a T4 improves by at least 3x against the current implementation at the same batch and sequence length.
- [ ] The current implementation remains in the tree and reachable, as the reference the gate checks against.
- [ ] The profile that motivated the change is recorded, so the gain is attributable rather than incidental.
