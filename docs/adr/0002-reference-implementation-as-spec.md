---
status: accepted
---

# The reference implementation is the spec; our model stays ours

Our H-Net is a from-scratch pure-PyTorch reimplementation, and it is missing published details
(no per-stage learning-rate multipliers, d_state 16 against the paper's 128, no expand, no sliding
window attention). Rather than patching from the paper alone, or vendoring the reference repo
(which hard-requires flash-attn 2.8.1 and therefore cannot run on a T4 at all), we treat the
reference implementation's source as the specification and port its semantics into our own
pure-PyTorch model.

Consequence: our model remains ours and remains runnable on Kaggle's T4, while matching the
reference's behaviour on the details we were getting wrong. Where the two disagree, the
reference wins and the difference gets recorded.
