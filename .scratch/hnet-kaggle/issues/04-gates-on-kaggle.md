# Gates on Kaggle

Type: grilling
Status: resolved
Blocked by: 03

## Question

Which of the five correctness gates survive a Kaggle-only, no-compiled-extension
environment, and what replaces the ones that do not?

The five from `docs/PLAN.md`: EMA/scan exactness, causality, compression control (the
hockey stick), FLOPs parity, reference diff. The problem is the fifth: the reference
implementation `main-horse/hnet-impl` needs a `mamba-ssm` git fork, and the official
`goombalab/hnet` needs `flash_attn==2.8.0.post2` which will not build for sm_75. Neither is
a drop-in on a T4.

This ticket decides:

- Whether to attempt the `hnet-impl` build on a Kaggle CPU/GPU kernel at all (cost: one
  session, unknown payoff), or substitute a second independent implementation of the router
  and dechunk (the two components most likely to be silently wrong).
- Which gates are CPU-runnable (exactness, causality, and probably FLOPs parity) and which
  need a GPU.
- What counts as "the chunker engaged" in a numeric, pre-registered way — the hockey-stick
  signature is qualitative in the literature ("collapse, spike above init, settle toward
  1/N within ~500 steps"), and a gate needs a threshold.
- What to do when a gate cannot be run at all: record the weakening explicitly in the
  result, rather than quietly dropping it.

## Answer

**All five gates are CPU-runnable and all five now run on Kaggle — including a real
replacement for the reference diff. Suite: `tests/test_gates.py`, executed by kernel
`mauricew/hnet-gates` v2 (CPU, no extensions, COMPLETE, exit 0, 8/8 PASS, ~12 s).**

Verified on Kaggle, not just locally: the same file passes in both places, and the CPU
image (3.13, torch 2.11) runs it without any compiled dependency.

**Gate-by-gate disposition**

| gate | survives? | how |
|---|---|---|
| scan exactness | yes, CPU | `chunked_scan` vs `sequential_scan`, fp32 max-abs < 1e-5, including a non-multiple-of-chunk length so the carry path is exercised |
| dechunk EMA exactness | yes, CPU | naive per-chunk recursion vs `DeChunkLayer`, < 1e-5 |
| causality | yes, CPU | bump byte t at three positions; positions < t must not move (encoder, router, main, decoder all in one pass) |
| compression control | yes, CPU | 30 steps: ratio loss must fall **and** bytes/chunk must rise |
| FLOPs parity | yes, CPU | analytic counter, checked for direction, not just magnitude |
| **reference diff** | **no** | replaced — see below |

**The reference diff is gone; here is what replaces it.** `main-horse/hnet-impl` needs the
`mamba-ssm` git fork, `goombalab/hnet` needs `flash_attn==2.8.0.post2` (sm_80+), and neither
is a drop-in on a T4 or a CPU session. The two components most likely to be silently wrong are
the ones the reference would have caught, so they get **independent second implementations**:

- **router** — re-derived from the paper's own formula with `F.cosine_similarity`, compared to
  `Router.forward` on a 1e-5 bound, plus the invariant that `p_0 == 1` so every sequence opens
  a chunk (the multi-turn bug in the source notes).
- **dechunk** — a Python recursion over chunk starts, independent of the fused scan.
- **chunk layer** — every slot must hold the byte at its boundary index, in order, and padding
  slots must be zero.
- **causal mask** — a separate mask check inside the gate above, since the main network's mask
  is built by hand.

**This is a weakened gate and it is recorded as one.** An independent implementation catches
transcription errors in the component, not a misreading of the paper — both implementations can
be wrong the same way. What it cannot catch: a wrong module order, a wrong EMA decay direction,
a wrong residual placement. Those are held by the anatomy notes and by the exactness gate, not
by this one. If a GPU with sm_80+ is ever available, one reference-diff run should be taken and
this ticket amended rather than treated as settled.

**Already paid for itself:** writing the dechunk reference established that `STE(c) = c + sg(1-c)`
evaluates to **exactly 1**, so the confidence gate is a gradient device and never a multiplier in
forward. Anyone "fixing" it into a real multiplier changes dechunk output scale. A dedicated gate
pins value-is-identity and gradient-through-p.

**Pre-registered definition of "the chunker engaged"** (numeric, fixed before any Q1 run)

- **Alive**: realised bytes/chunk rises monotonically for 500 consecutive steps and exceeds
  `1.5x` its step-0 value.
- **Bounded**: realised bytes/chunk settles inside `[0.5N, 2N]` for the fixed target `N`.
- **Character-aware**: on `rus`/`jpn`/`arb` the alignment lift is `> +0.15` averaged over the
  last 3 evals, and `align_on_lead_byte > 0.85`.
- **Not the density artefact**: raw precision alone is not evidence; the lift over the
  per-script base rate is.
- **Threshold fallback**: if bytes/chunk leaves `[0.5N, 2N]`, engage the cosine-decayed
  kth-value threshold with fake `p>=0.5` for the ratio loss/STE and real `p>=thresh` for
  chunking, and report both.
- **`cmn` is read as a trend, not a pass/fail**, since the dev loop showed it degrading as
  chunks shortened (lift +0.173 -> -0.078). **`eng` is scored by bpb only** — nearly every
  ASCII byte is a lead byte, so its alignment lift is structurally ~0.

**Failure of these gates is a statement about the run, not about H-Net** — 0.45-8 M params on
one T4. The suite must be green before any GPU kernel is pushed; a red gate burns quota silently
because the loss still goes down.

