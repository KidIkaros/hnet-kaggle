---
status: accepted
---

# The artifact is a transformers-compatible model first, a package second

The end state is a Hugging Face `transformers`-compatible class — a custom `PreTrainedModel`
with `config.json` and `generation_config.json`, registered so `from_pretrained` works —
plus a public checkpoint and a tokenizer-free byte API. A standalone installable package follows
as a thin wrapper over the same code.

Explicit non-goals for now: no tokenizer, no fixed vocabulary, no PyPI-first packaging. The
value of the artifact is that someone can point an existing pipeline at raw bytes and get a
model that behaves like the tokenized one they had.

Amended 2026-10-31: weights and code are published on **GitHub first**; the Hugging Face Hub
repository is created when the checkpoint is judged worthy of it, not on the day training ends.
The first release of the harness covers sampling with temperature and top-k, prompt
continuation, and integration with `generate()`.
