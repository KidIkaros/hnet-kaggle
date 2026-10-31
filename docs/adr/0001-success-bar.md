---
status: accepted
---

# The success bar for this effort is a usable model, not a number

The bar is three things together, not one: the training dynamics must settle (F and G reach
1/N and stay, the loss stops destabilising), the model must reach quality verifiable against
published numbers, and it must generate usable text. The end state is a real, tangible model
that could stand in for a tokenizer in someone else's pipeline — not a concept and not a
demonstration.

This was decided after the effort spent five weeks measuring a mechanism it had not been asked
to measure, because "workable" and "better than what others use" were never pinned to numbers.
Consequence: every run now has an explicit pass/fail against these three, and the spec is
written to them rather than to a research question.
