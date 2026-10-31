# Map: the shippable tokenizer-free H-Net

Label: `wayfinder:map` · Effort: hnet-model · Charted 2026-10-31

## Destination

A byte-level hierarchical model that trains stably, trains fast enough to reach a rung where
published numbers exist, evaluates against those numbers, and ships as something a person can
load and use. The spec is `SPEC.md`; the tickets are in `issues/`, numbered in dependency order.

## Why this is a separate workstream

The science workstream (`.scratch/hnet-kaggle/`) answered a different question — what the chunker
learns — and its answer is now a section of the write-up rather than the deliverable. Its map and
ticket history stay as the record. This workstream exists because the goal is the model.

## Frontier

Everything with no blockers can start now, and all of it is CPU work:

- 01 compiled chunked associative scan (bit-exact gate)
- 02 FLOPs-budgeted step counts, gated against wall-clock
- 03 English corpus at 2-3 GB, byte-length sorted
- 04 port the reference implementation's semantics
- 08 sampling and prompt continuation

Gated behind them: 05 the compression settles, then 06 (contingency) and 07 the comparable rung.
Behind 07 and 09: 10 the transformers-compatible class, 11 the release, 12 the write-up.

## Standing constraints

- The GPU is reserved for the shippable rung. The multilingual, image and audio arms continue on
  CPU only.
- Three iterations of fix-train-evaluate on the dynamics, then escalate to the reference's
  parameterisation work rather than tuning further.
- Every metric is audited against a synthetic oracle before it is allowed to support a claim.
- The optimised implementation must reproduce the current one bit-exactly.
