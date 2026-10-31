---
status: accepted
---

# The shippable comparison runs on the published setup, not ours

To make "verifiable against published numbers" real, the shippable rung is trained in English,
on documents sorted by byte length, with the step count set by a FLOPs budget rather than by a
step count. The published BPE and byte-H-Net numbers are then the external bar, instead of our
own earlier runs on a different corpus.

Amended 2026-10-31: the corpus is **2-3 GB of English (a fineweb-edu mirror or fineweb-2
English)**, not the 10 GB FineWeb-10BT sample. The pragmatic call: one download rather than ten
gigabytes, and enough tokens that a 30M-parameter model is not data-starved. The cost is that
our bpb is not *exactly* beside the published figures — same metric, same measurement discipline,
different corpus — so the write-up states that plainly rather than implying a strict comparison.

Consequence: the six-language corpus stops being the substrate for the shippable artifact — it
stays as the science track — and the rung aims at 3e16 FLOPs and roughly 30M parameters, fixed
upward or downward once the speed work reveals what the budget actually affords.
