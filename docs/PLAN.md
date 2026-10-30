# H-Net on Kaggle: experiment plan (v2, revised after the anatomy pass)

Supersedes v1. What changed, and why:

- v1 proposed testing the paper's 1.28x compute multiplier. **Dead.** H-Net's own curves
  cross BPE at 100B bytes (1-stage) / 30B (2-stage); ablations were run at 36B bytes; our
  whole remaining weekly quota buys ~1.6GB at 1e17 FLOPs. The per-scale gap is 0.755 vs
  0.756 bpb at 760M — the multiplier is a fitted-law extrapolation, not a visible gap — and
  the one public test at our scale (1e16-1e18) found no win on English.
- v1's ladder design was also invalid on its own terms: ratio dynamics need 500+ steps to
  stabilise and the practitioner guidance is never to sweep under ~5000 steps. A "cheap
  rung" is not cheap if it measures initialisation instead of architecture.
- The anatomy pass (`ANATOMY.md`) turned up two questions that *are* reachable, are
  unmeasured by anyone, and cost little.

Budget unchanged: 9.72 GPU-h left this week (reset 2026-10-31), 30 h/week after. One T4.

---

## What we test

**Q1 — Mechanism (cheap, novel).** Does a learned chunker discover *characters* before
words on non-Latin scripts? Concretely: what fraction of chunk boundaries fall on UTF-8
lead bytes, per script, as training progresses — and does the boundary rule (NLL vs
entropy) change that?

Nothing published answers this. The 2026 post shows multi-byte-aware chunks in a figure but
publishes no alignment statistic, and states the original LH-Net "doesn't segment characters
correctly, even after both stages of chunking". H-Net++ reports 73.8% F1 on gold
*morphological* boundaries for Persian — a related but different measurement, on a
morphologically rich Latin-script language.

**Q2 — Transfer (medium, the paper's most striking claim).** At *equal validation bpb*,
does the byte-level H-Net transfer to held-out scripts better than a BPE Transformer at the
same validation bpb? That is Post II's loss-to-loss claim, and unlike a compute multiplier it
is a **ratio at matched loss**, which is robust to absolute scale. Post II claims
byte ≈ H-Net > BPE, with a 1.5x gap from the byte ceiling to BPE.

**Q3 — Directional (optional).** At one fixed FLOPs budget in the undertrained regime, is the
H-Net-vs-BPE gap smaller on Chinese than on English? Directional evidence only — one budget
cannot establish a slope, and it should be reported as such.

Explicitly **not** tested: the 1.28x/2.06x multipliers, downstream suites (no OLMES/Paloma
at this scale), and anything needing >10B training bytes.

---

## Design

**Arms** (all read identical bytes, same document boundaries):
- `bpe` — Transformer + in-corpus BPE (GPT-2-style pretokenization), vocab sized to match
  realised bytes/unit.
- `byte` — same Transformer, 256-way vocab, no chunking. Required control: without it,
  "hierarchy helped" and "bytes helped" are not separable.
- `hnet` — 1-stage, outer nets = our Triton-scan SSM, N swept.
- `hnet-space` — same architecture, whitespace boundaries. Isolates "learned" from
  "hierarchical". Only if budget allows.

**Outer networks must be SSMs.** The E/D ablation is unambiguous: pure-Transformer outer
nets (T2-T2) are "by far the worst... despite using more FLOPs", and the effect persists on
BPE-token inputs, so it is about compression into fixed state, not byte handling. This makes
`mbdfa`'s fused Triton scan a prerequisite rather than an optimisation — v1 had this
backwards.

**Non-negotiable construction details** (each is a documented run-killer, from `ANATOMY.md`
§7):
- `W_q`, `W_k` identity-initialised, excluded from reinit.
- `W_res` zero-initialised, fp32. Kaiming init removes the ratio-loss hockey stick and the
  run stalls.
- Norm scales trainable at S≥1 with η ∝ 1; keep `base_std` small — high init σ collapses
  compression at step 0 and "dooms all high σ runs".
- Dechunk EMA: `A = −1`, `dt = log(1/(1−p))`, `b = p`, `c = 1`; gate against a naive loop.
- Ratio loss α = 0.03; report **realised** compression ratio, never the target.
- kth-value threshold pre-registered as the intervention of choice if the ratio wanders
  (cosine-decayed, with "fake" `p ≥ 0.5` for the ratio loss/STE and "real" `p ≥ thresh` for
  chunking). ATDC exists because this is a known weak point.

**Minimum run length: 5000 steps.** Below that the ratio dynamics dominate and the
comparison is meaningless. This constrains the design far more than FLOPs do.

**LR**: swept per arm, and per stage for the H-Net, at one width. μP transfer does not hold
at S≥1 — structurally, since routing alignment is capacity-dependent — so no extrapolation.

**Precision**: fp16 autocast + fp32 master weights + GradScaler (no bf16 on sm_75), gated
against an fp32 reference.

---

## Phase 0 — build and gate (local RTX 2070, zero quota)

    src/model.py    H-Net S=1: encoder, router, chunk, main, dechunk, decoder
    src/data.py     byte packing, per-language splits, in-corpus BPE trainer
    src/flops.py    analytic FLOPs/byte for all arms
    src/gates.py    the five gates, runnable locally
    src/train.py    shared loop (WSD, autocast, GradScaler, eval bpb)
    kernel/main.py  self-contained Kaggle script kernel (inlines src/*)

| # | gate | criterion |
|---|---|---|
| G1 | EMA/scan exactness | fused/borrowed scan vs naive loop: max abs diff < 1e-5 fp32 |
| G2 | causality | perturbing byte t+1 changes nothing at positions ≤ t |
| G3 | compression control | within 500 steps: ratio loss collapses → spikes above init → settles toward 1/N (the hockey stick) |
| G4 | FLOPs parity | analytic FLOPs/byte vs wall clock within 15% across arms |
| G5 | reference diff | same seed/batch vs vendored `hnet-impl` or a second router implementation: loss within 2% after 100 steps, boundary masks agree ≥90% |

G5 catches a silently wrong router. If `hnet-impl` will not build locally (it needs a
mamba-ssm fork), substitute a second independent router implementation and record that the
gate was weakened.

Exit: all gates green, plus a 5-minute smoke train producing readable chunk boundaries on
each of eng/deu/rus/jpn/cmn.

---

## Phase 1 — Q1 mechanism (Kaggle, ~2-3 GPU-h)

One H-Net, ~8M params, mixed multilingual bytes, N ∈ {3, 6, 12}, ≥5000 steps each (3 runs).

Measurements, per script, over training:
1. realised compression ratio vs target;
2. **boundary-to-character alignment** — fraction of boundaries on UTF-8 lead bytes, using
   `mbdfa`'s DFA state stream (exact, free);
3. boundary-to-word alignment on English (±1 byte of a space);
4. 20 visualised examples per language;
5. **NLL-vs-entropy boundary rule** — same architecture, boundaries from a proxy model's NLL
   threshold instead of the learned router. Gu argues NLL is correct and entropy (BLT/DPT) is
   not, and nobody has published the head-to-head. Cheap, novel either way.

Kill criterion: if the compression ratio sits at 0.4-0.5 with no hockey stick by 30% of the
run, the chunker is dead. Apply the kth-value threshold; if that fails, stop and report the
failure rather than scaling up.

---

## Phase 2 — Q2 transfer (Kaggle, ~8-12 GPU-h)

Train each arm to a **matched validation bpb** on in-domain data (eng+deu+rus), then measure
held-out-script bpb (jpn, cmn, ara) at that matched loss. 3 arms × 2 matched-loss targets ×
2 seeds = 12 runs. Match by stopping at a target val bpb rather than by FLOPs, so the result
is a ratio at matched loss and does not depend on our FLOPs estimate being right.

Deliverable: (val bpb, held-out bpb) points and a fitted slope per arm — the loss-to-loss
law. Prediction from Post II: `byte` best, `hnet` ≈ `byte`, `bpe` worst. If `hnet` lands on
the `bpe` curve, Post II's mechanism does not reproduce at this scale, and that is the
finding.

---

## Phase 3 — the model we keep (~2 GPU-h)

Only if Phase 1-2 produce signal. What makes it ours:

1. **Runs where the reference cannot.** No `flash_attn`, no `mamba_ssm`; sm_75-safe. The
   official repo cannot run on a T4 at all.
2. **Fused Triton scan in the outer networks** — already built and gated at 5.53x in
   `mbdfa`, and now load-bearing rather than optional.
3. **DFA-constrained decoding** — free at decode time, makes UTF-8 validity structural.
4. **Multilingual-first**: the only H-Net variant with published CJK boundary-alignment
   numbers.

---

## Schedule and budget

| when | what | GPU-h |
|---|---|---|
| now → Oct 30 | Phase 0 on the 2070, gates, smoke train | 0 |
| Oct 31 → Nov 6 | Phase 1 + first Phase 2 arms | ~6-8 |
| Nov 7 → Nov 13 | Phase 2 remaining arms/seeds | ~6-8 |
| after | Phase 3, writeup | ~2 |

Phase 0 is local, so a slip costs time rather than quota. Kaggle rules carried over from
`mbdfa/NOTES.md`: dataset mounts at `/kaggle/input/datasets/<owner>/<slug>/` (resolve with
glob); `kernels logs` is empty while RUNNING (use `-f`); pushing over a running kernel does
not cancel it (cost 11.2 h once) and deleting does; always set `max_steps` **and** a
wall-clock guard.

---

## What would make us wrong

1. **Chunker never engages** at 8M params on our data. Q1 still yields a result (alignment of
   a partially-learned chunker is measurable, and reproducing the collapse is informative),
   but Q2 becomes untestable — say so; do not report Q2 as a null about H-Nets.
2. **BPE baseline under-trained.** Sweep LR per arm; report the curve, not the best point.
3. **Matched-loss matching hides a compute difference.** Two arms can hit the same val bpb at
   very different FLOPs; report realised FLOPs/byte and do not convert the loss-to-loss
   result into a compute multiplier.
4. **Data too small.** 48MB/language means multiple epochs; cap epochs and watch for the
   overfit knee.
5. **H-Net++ and ATDC already exist.** We are not first to "H-Net on a hard language" — we
   are first to a per-script boundary-alignment measurement and to the NLL-vs-entropy
   head-to-head. Claim only that.

---

## Open decisions

1. **Corpus**: Tatoeba 6-language 48MB (ready, free, multi-epoch) vs ~1-2GB from public
   Kaggle mirrors of fineweb-edu + fineweb-2 zh/ja/ar (better scale, +1-2h prep).
2. **Scope**: Q1 only (mechanism, ~3 GPU-h, safe) vs Q1+Q2 (adds the transfer claim,
   ~12-15 GPU-h, needs two quota weeks).
3. **Q3**: include the fixed-FLOPs cross-language gap as a directional add-on, or leave it.
