# 10 — transformers-compatible class

**What to build:** **What to build:** The model loads the way any other model does — from_pretrained, with its own
configuration — and generates through the standard interface.

**Blocked by:** 08

**Status:** ready-for-agent

- [ ] A custom PreTrainedModel subclass loads from a published checkpoint directory.
- [ ] The configuration and the generation configuration are present and complete.
- [ ] The class is registered so the standard auto-class entry points find it.
- [ ] generate() produces the same bytes as the native sampling path for the same seed and parameters.
- [ ] The class works with the byte-level API: bytes in, bytes out, no vocabulary anywhere.
