# 07 — The comparable rung

**What to build:** **What to build:** The shippable rung — a model of roughly 30M parameters trained on English for
about 3e16 FLOPs — producing the checkpoint and the bpb that gets set beside published numbers.

**Blocked by:** 01, 02, 03, 05

**Status:** ready-for-agent

- [ ] The run is specified by a FLOPs allowance, and its geometry is in the published family at roughly 30M parameters.
- [ ] Held-out bpb is computed on the metric the published figures use.
- [ ] The compression metrics that held in the settling ticket still hold at this scale, not just at small scale.
- [ ] The result is set beside the published BPE and byte-H-Net figures, with the corpus difference stated inside the comparison.
- [ ] The checkpoint is stored with the metrics and the config that produced it.
- [ ] The actual rung is recorded, whether it landed at the intended FLOPs or below them.
