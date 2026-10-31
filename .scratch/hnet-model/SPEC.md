# Spec: a shippable tokenizer-free H-Net

Status: ready-for-agent
Workstream: hnet-model
Written: 2026-10-31
Decisions: see `docs/adr/` (0001-0009) and `CONTEXT.md` for the vocabulary this spec uses.

## Problem Statement

We have a from-scratch, pure-PyTorch reimplementation of a hierarchical byte-level language
model. It runs, it trains, and it produces interesting measurements — but it does not produce a
model. Three things stand between it and a usable artifact:

**Its compression does not reach its target.** The chunker's boundary statistics settle well away
from where they should be: in our longest run the mean boundary probability sat at 0.305 against
a target of 0.167, the ratio loss never dropped below about 1.03, and the realised compression
was 4.56 bytes per chunk against a target of 6 — a 24% miss in the wrong direction.

*Corrected 2026-10-31:* an earlier draft of this spec also claimed the loss destabilised late in
training, on the strength of the logged per-batch loss. That was a misreading: the logged loss is
one batch's cross-entropy, and it moves with that batch's language mix. Measured on fixed
validation windows, bpb descends monotonically in every run we have — the 2.43 M run goes
2.882 -> 2.292 with no late rise, and the two arms of the current sweep go 3.092 -> 2.476 and
3.070 -> 2.446. **The model converges.** The gap is the compression target, not stability, and
the programme is narrower than this spec first claimed.

**It is too slow to reach a comparable rung.** At roughly 1e11 FLOPS on a T4, against the ~1e13 a
tensor-core implementation reaches, a 1e16-FLOP run costs a week of quota. The published
comparisons start at 1e16.

**There is no artifact.** No sampling code, no evaluation harness, no checkpoint anyone could
load. We have boundary dumps and bpb on our own corpus.

Underneath all three sits a course correction: this effort spent its first weeks measuring a
mechanism it was never asked to measure — whether the chunker aligns with characters — instead of
building the model the papers are about. That measurement is real, it replicates across text,
images and audio, and it becomes a section of the write-up. It is not the deliverable.

## Solution

A byte-level hierarchical model that trains stably, trains fast enough to reach a rung where
published numbers exist, evaluates against those numbers, and ships as something a person can
load and use.

Concretely: the compression settles (F and G reach 1/N and stay, bpb descends monotonically), the
implementation runs at a fraction of tensor-core peak so the comparable rung fits in the budget,
the model is trained on English with FLOPs-budgeted step counts at a rung around 3e16 FLOPs and
roughly 30M parameters, and the result is evaluated in bpb and in byte-scored multiple choice
against the published BPE and byte-H-Net figures — then released as a `transformers`-compatible
class with a public checkpoint and a byte-level API.

The 3B-at-10x-less-compute claim stays as the north star. We do what our budget can measure and
state plainly what we cannot.

## User Stories

1. As a researcher, I want the compression dynamics to settle, so that I can tell a working
   training run from a broken one without reading every log.
2. As a researcher, I want the router's boundary probability to converge to the compression
   target, so that the model holds its intended compression instead of drifting.
3. As a researcher, I want the logged loss and the validation loss kept distinct, so that a
   single batch's language mix is never mistaken for instability — the two move differently, and
   only the second says anything about the model.
4. As a researcher, I want the ratio loss to reach its theoretical minimum of 1, so that the
   compression target is being met rather than fought.
5. As a researcher, I want per-stage learning-rate multipliers implemented as the reference
   implementation specifies them, so that our model is not silently missing a published
   stability trick.
6. As a researcher, I want every semantic detail of the reference implementation ported into our
   pure-PyTorch model, so that our behaviour differences from it are known rather than guessed.
7. As a researcher, I want the scan replaced with a compiled chunked associative scan, so that
   the same FLOPs budget costs hours instead of a week.
8. As a researcher, I want the main network to use scaled-dot-product attention, so that
   attention is not the bottleneck.
9. As a researcher, I want the optimised implementation to reproduce the current one exactly, so
   that speed does not come at the cost of correctness.
10. As a researcher, I want FLOPs-per-byte accounting to agree with measured wall-clock, so that
    a step count derived from a compute budget is trustworthy.
11. As a researcher, I want step counts determined by a FLOPs budget rather than a fixed number,
    so that runs of different sizes and shapes are comparable.
12. As a researcher, I want the training corpus to be English, documents sorted by byte length,
    so that the measurement protocol matches the published comparisons.
13. As a researcher, I want the corpus large enough that the model is not data-starved, so that
    the rung is limited by compute rather than by tokens.
14. As a researcher, I want the shipped rung to be the largest one that fits the remaining quota
    without gambling, so that the effort finishes with a result rather than a plan.
15. As a researcher, I want a stopping rule on the dynamics work, so that the effort changes
    approach instead of iterating on a fix that is not working.
16. As a researcher, I want the escalation path to be the reference implementation's
    parameterisation work rather than more of our own tuning, so that a third failed iteration
    does not become a fourth.
17. As a researcher, I want the byte-phase finding preserved as a section of the write-up, so
    that the explanation for our model's chunking behaviour is not lost.
18. As a reader of the write-up, I want one document with the model as the headline and the
    mechanism as a section, so that the mechanism is read as the explanation rather than as the
    result.
19. As a reader of the write-up, I want the corpus difference from the published setup stated
    plainly, so that our bpb is not mistaken for a strictly comparable number.
20. As a reader of the write-up, I want to know what we could not reach, so that the ambition is
    visible without the claim being overstated.
21. As a model user, I want to load the model with `from_pretrained`, so that using it needs no
    custom plumbing.
22. As a model user, I want to feed it bytes and get bytes back, so that I never have to think
    about a vocabulary.
23. As a model user, I want to continue a prompt and get a plausible continuation, so that I can
    judge the model by reading it.
24. As a model user, I want temperature and top-k sampling, so that I can control how repetitive
    the output is.
25. As a model user, I want to use the standard generation interface, so that the model behaves
    like any other one I have loaded.
26. As a model user, I want the model card to state the licence, the training data and the
    measured quality, so that I can judge whether to depend on it.
27. As a model user, I want the weights and code published on GitHub, so that I can inspect and
    fork them.
28. As a future contributor, I want the training pipeline to be reproducible from a config, so
    that I can run the same experiment without reverse-engineering the scripts.
29. As a future contributor, I want the existing test gates to keep passing, so that the
    optimisation work cannot silently change the model's semantics.
30. As a future contributor, I want the science track to keep running alongside, so that the
    byte-phase work in other modalities is not abandoned when the artifact ships.
31. As the effort owner, I want the GPU reserved for the shippable rung, so that quota goes to
    the model rather than to parallel science.
32. As the effort owner, I want the science track to run on CPU, so that it continues at no GPU
    cost.

## Implementation Decisions

**The model is ours; the reference implementation is the spec.** The reference implementation
cannot run on our hardware, so its semantics — including the per-stage learning-rate multipliers
the paper specifies — are ported into our own pure-PyTorch model. Where the two disagree, the
reference wins and the difference is recorded.

**Optimisation comes before scale.** The scan becomes a compiled chunked associative scan and the
main network uses scaled-dot-product attention, gated by bit-exact agreement with the current
implementation. A looser numerical tolerance would buy more speed and a weaker guarantee; the
existing implementation stays in the tree as the reference the gate checks against.

**Steps are set by a FLOPs budget.** Runs are specified by an allowance of compute, converted to a
step count through the analytic per-byte accounting, with the accounting itself gated against
measured wall-clock.

**The rung is English, byte-length sorted, and about 3e16 FLOPs at roughly 30M parameters.** The
corpus is two to three gigabytes of English from a fineweb-edu mirror or fineweb-2 English. The
actual rung is fixed after the speed work reveals what the quota affords, in either direction,
rather than gambled on.

**The compression target is fixed, not tuned away.** If the dynamics will not settle at the
target ratio, the fix is in the optimisation, not in moving the target — though a curriculum that
*ramp* to the target is an allowed mechanism, because it ends at the same target.

**The interventions come from the published record, not from our own invention.** A cited
catalogue of every published intervention for the same failure — with the ones reported as not
helping, and the numbers behind each — is the reference for the dynamics work. Where two sources
disagree, the disagreement is recorded rather than resolved by preference. The catalogue also
corrects a mis-attribution in the existing research notes, which had attributed a low-resource
chunking paper to an unrelated title.

**The learned router is measured, not assumed.** One arm replaces it with a fixed stride at the
same compression and the same FLOPs. If the fixed stride matches the learned chunker, the router
is not earning its place — which is a statement about our model, and one we are willing to
publish either way.

**The byte phase is accepted.** Our router settles on a byte phase rather than on content. The
published comparison is against a learned chunker rather than a content-aligned one, so the phase
is not disqualifying; the model work does not depend on making chunking semantic.

**The stopping rule is three iterations.** Three cycles of fix-dynamics, train, evaluate. If the
model still has not beaten our own current numbers, we escalate to the reference implementation's
parameterisation work instead of tuning further.

**The artifact is a `transformers`-compatible model.** A custom `PreTrainedModel` with its own
configuration and generation configuration, registered so `from_pretrained` works, plus a public
checkpoint and a byte-level API. A standalone package follows as a thin wrapper. Weights and code
go to GitHub first; the Hugging Face Hub repository comes later, when the checkpoint is judged
worthy of it.

**The first harness release covers sampling, prompt continuation and the standard generation
interface.** Streaming, long-context handling, batched generation, and varlen document packing are
not in the first release.

**Evaluation is bpb plus byte-scored multiple choice.** Multiple-choice tasks are scored by byte
likelihood, which a tokenizer-free model can do natively. The suite grows from there.

**The science track continues on CPU.** The multilingual, image and audio arms keep running at no
GPU cost, and their findings are folded into the single write-up.

## Testing Decisions

A good test here checks external behaviour — outputs, metrics, numbers a user would see — and not
the shape of the internals. Four seams, each a distinct contract, with the existing seams reused
where they already exist.

**Seam 1, end-to-end: a training run.** A config in, and a checkpoint plus bpb, F, G, ratio loss,
realised bytes per chunk and a generated sample out. This is the highest seam and the one that
matters most; most verification happens through it. The existing corpus and kernel build
machinery is reused rather than replaced.

**Seam 2, the model: the forward pass.** Carries the bit-exactness gate between the current and
optimised implementations, and the reference-diff gate between our semantics and the reference
implementation's. Prior art in the repository: the gate suite and the alignment-probe test, both
of which caught real bugs by checking an externally visible property against a synthetic oracle.

**Seam 3, accounting: FLOPs per byte.** Gated against measured wall-clock, following the existing
FLOPs-parity pattern, so that a step count derived from a compute budget is trustworthy.

**Seam 4, the artifact: load and generate.** A checkpoint loads through the public interface and
produces bytes; the byte-scored evaluation harness runs against it. This is the contract a user
depends on, so it is tested as a user would use it.

Metric audits come before metric use. Every new metric is pointed at a synthetic oracle — a
known-good chunker must score high, a random one must score near zero — before it is allowed to
support a claim. Two of the three units bugs in this project were caught exactly this way, and
one of them had already produced a plausible number.

## Out of Scope

- Training a BPE Transformer or a byte Transformer as a baseline. External published numbers are
  the comparison; no other model is built.
- The 3B-at-10x-less-compute claim as a tested result. It is the north star, not a deliverable.
- Publishing to the Hugging Face Hub in this effort.
- A PyPI package.
- Streaming inference, long-context handling, batched generation, varlen document packing.
- Interventions to make chunking content-aligned.
- Non-Latin scripts in the shippable model.
- Vision and audio models, beyond the existing science-track arms.

## Further Notes

**Dependencies in flight.** Three small-scale probes are running to establish whether the missing
learning-rate multipliers or a stronger ratio-loss weight is what stops the compression settling;
they are the first experiment of the dynamics phase. A background research pass is producing a
cited catalogue of every published intervention for the same failure, including the ones already
reported as not helping, so that the dynamics phase does not rediscover them.

**What the write-up becomes.** One document. The byte-phase finding becomes a section that
explains why our model's chunking looks arbitrary, and the model-quality result is the headline.
The existing write-up is restructured rather than replaced.

**Where the effort's artefacts live.** The spec and its tickets are a new tracker root, separate
from the science track's existing map and ticket history, which stay as the record of what was
learned about the chunker.
