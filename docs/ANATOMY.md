# Anatomy of H-Net: how it was actually built

Synthesis of arXiv:2507.07955v2 (incl. appendices), the `goombalab/hnet` source, the
authors' own retrospective (`goombalab.github.io/blog/2025/hnet-past`), the independent
`main-horse/hnet-impl` project and its engineering/parameterization write-ups, the 2026
Cartesia posts, H-Net++ (2508.05628), ATDC (2605.30080) and *When Tokenizers Fail*
(2608.27658).

`RESEARCH.md` states the claims and the regime. This file is about construction: what
they tried, what they kept, why each piece is there, and what breaks.

---

## 1. Origin: a negative result, then a decade-old idea

Gu's own account starts with a failure. S4 (2021) contained an unreported autoregressive
U-Net with two downsampling stages for CIFAR-10 pixels. He ran the same backbone on
WikiText-103 and:

> "no matter what I did, the hierarchical version was always noticeably worse than the
> isotropic backbone for language modeling."

Diagnosis: "fixed-width pooling just didn't make sense for language" — it is "essentially
the same as convolutions, or at least has the same inductive bias". The same backbone
*worked* for audio (SaShiMi), because audio has "an underlying notion of uniform sampling
rate". SpaceByte and BLT later reproduced the ablation; H-Net's own `H-Net (pool)` rung is
the same result again.

So the founding question was not "can we beat BPE" but: **can chunking be a learned
primitive rather than a fixed one?** Gu's framing: attention is a primitive, recurrence is
a primitive, chunking should be too. And the blocker was stated plainly — "grouping
discrete units of data together is, of course, a discrete selection problem, which is
generally very hard for differentiable optimization. Unfortunately, I had absolutely no
idea how to do this."

**First idea (2024): constant-information chunking.** Train an AR model, read conditional
distributions, group units until the accumulated NLL of the chunk crosses a threshold.
Non-differentiable, requires a proxy model — which is, structurally, BLT. It was explored
by a student "but didn't get too far".

**The pivot:** Sukjun Hwang proposed routing as differentiable selection, "loosely
inspired by mixture-of-experts … MoE is essentially a discrete selection problem like the
one we want to solve." Gu: "at the time, I actually basically didn't know anything at all
about how MoE worked."

That lineage matters for reading the design: the router is an MoE gate, the ratio loss is
an MoE load-balancing loss, and the straight-through estimator is the MoE/discrete-latent
trick. H-Net is MoE-shaped compression, not tokenizer-shaped compression.

---

## 2. What was tried and abandoned

From the posted late-2024 whiteboard notes and the retrospective:

| idea | outcome | why |
|---|---|---|
| Fixed-width pooling / U-Net downsampling | **abandoned** | always worse than isotropic for LM (Gu's own 2021 result, reproduced by SpaceByte, BLT, H-Net-pool) |
| NLL-threshold chunking (constant information) | **abandoned** | not differentiable; needs a separately trained proxy model |
| **Gumbel noise** for the discrete boundary | **abandoned** | "we had tried many variations of Gumbel noise already… But it never seemed to help empirically." DPT also used Gumbel-softmax and "had stability issues" |
| Fresh input embeddings injected after the main network | **abandoned** | "more stable and helped us get early versions working, but didn't allow multi-stage recursive chunking… so it was abandoned relatively quickly" |
| Generic sparsity loss instead of a targeted ratio | **abandoned** | "sort of worked but it didn't seem as consistent" |
| Removing post-network norms + residual projections | **tried hard, failed** | "We spent a long time trying to remove these… But in the end, they seem very useful—maybe essential" |
| Entropy-based boundaries (BLT's rule) | **rejected on principle** | see §3 |
| Speculative decoding on top | **no-op** | measured, didn't help |

Two things Gu admits are under-reported: the per-network norms and the init rescaling that
balances them — "we might have forgotten to report this in the paper." And the scale of the
search: "it felt like there were **12 different axes** that were all critical and that all
had to interact together in exactly the right way."

---

## 3. Position against the prior work, precisely

| work | what it does | H-Net's delta |
|---|---|---|
| **Hourglass Transformer** (Nawrot 2022) | autoregressive U-Net, *fixed* downsampling | dynamic, learned boundaries |
| **MegaByte** (2023) | multiscale, fixed-size patches | same |
| **Dynamic Pooling Transformer** (Nawrot 2023) | U-Net + binary boundary predictor, **Gumbel-softmax** for the discrete choice | Gu: "the first, and I think essentially only, prior attempt at an end-to-end hierarchical autoregressive model with dynamic chunking" — and it "had stability issues and underperformed their alternative variants that relied on more heuristics and supervision". Diagnosis: premature, because Mamba didn't exist yet and "neither was Mamba, which we found to be quite important" |
| **SpaceByte** (2024) | chunk on whitespace/"space-like" bytes | boundaries learned, not delimited. Strong baseline — `H-Net (space)` ≈ BPE Transformer at Large |
| **BLT** (2024) | entropy-heuristic boundaries from a separately trained proxy | Gu's tl;dr: "**BLT is not actually end-to-end**". Three deltas: boundaries are heuristic not jointly learned; needs an auxiliary conditional-entropy model; **cannot be nested** ("in all our experiments, we found H-Net (2-stage) > H-Net (1-stage)"). Also judged BLT's chunk-aware cross-attention and hash n-gram embedding tables unnecessary |

**The sharpest technical claim in the whole retrospective** is about entropy vs
information content. BLT and DPT segment on conditional *entropy* `H(x_t | x_<t)`. Gu's
original idea used *negative log-likelihood*. Counterexample, verbatim:

> a distribution with most mass on word w "would have very low conditional entropy and so
> any entropy-based heuristic would never create a new chunk. But in the rare event that
> the actual next word is a different word v ≠ w, the model should be very surprised and
> draw a boundary."

This is testable and nobody has published the test. It is also an argument that the
BLT-like baseline in Post II is handicapped by its own boundary rule, not just by not
being end-to-end.

---

## 4. The architecture, in construction order

    x (bytes) -> encoder E -> routing -> chunk -> main M -> dechunk -> + residual -> decoder D -> logits

**Routing.** `p_t = (1 − cos(W_q x_t, W_k x_{t−1}))/2`, `b_t = 1{p_t ≥ 0.5}`, `p_1 = 1`.
`W_q, W_k` are **identity-initialised** and excluded from reinit, so at step 0 the router
computes raw cosine similarity between adjacent encoder states — no learning required to
be meaningful. The inductive bias is explicit: a semantic boundary is a direction change.

**Downsample.** `x[b]` — masked select. Beats mean/max pooling and cross-attention
(App. E.1), because the encoder learns to pack the chunk's context into the retained
vector. That is why the encoder must be an SSM (§6).

**Dechunk.** EMA over the chunk sequence, `z̄_t = P_t z_t + (1−P_t) z̄_{t−1}`, with `P` the
selected boundary probability. Implemented by borrowing Mamba-2's chunked scan with
`A = −1`, `dt = log(1/(1−p))`, `b = p`, `c = 1`. Its purpose is gradient flow: it turns a
discrete select into a continuous interpolation, and it self-corrects low-confidence
boundaries. Originally called the "EMA detokenizer"; the paper's ablations call it the most
important single component — without it, "compression ratios fluctuate severely throughout
training, preventing the model from learning consistent chunking boundaries".

**Upsample + residual.** `c_t = p_t` at boundaries else `1 − p_t`; `STE(c) = c + sg(1−c)`;
repeat the last chunk vector forward; multiply by `STE(c_t)`; add `W_res · x_enc` where
**`W_res` is zero-initialised in fp32**. Zero init is load-bearing: it forces the main
network to earn its contribution rather than free-riding on the encoder, and it is what
produces the characteristic ratio-loss hockey stick.

**Ratio loss.** `L = L_AR + α Σ_s L_ratio`, `α = 0.03`, with
`L_ratio = N/(N−1)·[(N−1)FG + (1−F)(1−G)]`. Gu on it: "a little artificial", introducing
"an annoying new hyperparameter (thankfully, the only one!)". It is the MoE load-balancing
loss applied to a compression target.

**Stability layer.** RMSNorm closing every sub-network; projection on the residual path
only (both-path and main-path projections were tried and were worse — they interfere with
gradient flow); per-stage LR multipliers `λ = 2.0, 1.5, 1.0`.

---

## 5. What the ablations actually establish

| ablation | result | reading |
|---|---|---|
| H-Net (pool) vs all | worst of the family | fixed-width chunking is not effective — the founding negative result, confirmed |
| H-Net (space) | ≈ BPE Transformer at Large | a good delimiter heuristic already gets you parity |
| 1-stage vs space | 1-stage wins | learned boundaries beat a strong heuristic |
| 2-stage vs 1-stage | 2-stage wins at both scales; overtakes BPE after 30B vs 100B bytes | hierarchy depth is where the real gain is |
| w/o smoothing | compression ratio unstable, large BPB loss | the EMA is the enabling trick |
| w/o cosine routing | worse, less interpretable chunks | similarity is a better inductive bias than per-position prediction |
| w/o STE | worse but not broken | stabiliser, not mechanism |
| **E/D = T2-T2** (pure Transformer) | **by far the worst, despite more FLOPs** | SSMs are load-bearing in the outer nets |
| same, on BPE-token input | Mamba still wins | it's compression into fixed state, not byte handling |
| H-Net vs FLOPs-matched byte MoE | H-Net much better | not generic sparsity |
| hybrid main net (3:1 Mamba:Transformer) | better than pure Transformer | standard hybrid results carry over |
| DNA: H-Net vs isotropic | 3.6× data efficiency at 29-66M params, 1-8B bp | the largest small-scale effect in the paper — in a domain with *no* tokenizer heuristic at all |

That last row is the most useful data point for anyone with a small budget, and the paper
barely headlines it.

---

## 6. Why the outer networks must be SSMs

The mechanism is a compression bottleneck: the main network only ever sees one vector per
chunk, so the encoder must decide what to pack into that vector. An SSM's whole structure
is "compress the prefix into a fixed-size state", which is the same operation. A causal
Transformer instead keeps an explicit KV cache of every position — it has no pressure to
summarise, and empirically it fails to ("struggles to compress byte sequences effectively,
resulting in both computational waste and degraded modeling performance").

This is confirmed on BPE tokens, so it is not about bytes. It is about the layer being
able to compress at all.

**Consequence for anyone building this:** the encoder/decoder cannot be `conv + SDPA`.
That is the T2-T2 configuration. The `mbdfa` Triton scan is not an optimisation here; it is
a prerequisite.

---

## 7. Parameterization: what actually makes it train

From `main-horse/hnet-impl`'s systematic study — the most detailed public account, and
largely absent from the paper.

**The S=0 baseline** uses spectral init plus a dualised Muon: `σ_k ∝ sqrt(min(1, d_k/d_{k−1})/d_{k−1})`
for hidden matrices, `σ = 1` for embeddings, and the **dualising** Muon variant (keep
singular vectors, discard singular-value magnitudes) so that `‖ΔW‖_* = Θ(sqrt(n_ℓ/n_{ℓ−1}))`.
RMSNorm γ is **not trained** at S=0.

**At S≥1 the rules change**, because norms stop being absorbable:

- `W_res` **zeros**, `W_q/W_k` **identity**, `pad_dimension` zeros.
- Norm scales become **trainable with η ∝ 1** — specifically for the isotropics that
  contribute to the residual. Reason: if `RMS(W_res)` grows, the main network cannot grow
  its own RMS to match, so it "will start painfully producing gigantic outlier coordinates
  to pass information out, which will destroy models in the long run."
- The dechunk EMA shrinks variance only slightly, so the residual branch and the main-net
  branch are near-equal in magnitude — which is exactly why `RMS(W_res)` growth is fatal.

**The headline negative result:** μP does not transfer end-to-end at S≥1, and the failure
is structural, not a bug. Routing gradient analysis: `p_t = Θ(1/√d)` and `Δp_t = Θ(1)` only
**iff** `q_{t−1}, k_t` stay "mostly unaligned across widths". But a real dynamic tokenizer
*should* shift its alignment as capacity grows — so "**by design**, any μP scheme that
assumes consistent alignment between sequence positions across widths is not going to work
for a H-Net."

Practical answer: sweep LRs per stage at one width, accept that they won't transfer, or
freeze a small pretrained encoder/decoder and reuse it as a "pretrained tokenizer".

**Initialisation traps that kill runs:**

- **High init σ collapses compression at step 0.** "↑σ ⇒ ↓L1:L0 at init": a large init plus
  a prefix scan/conv1d creates correlated adjacent states, the realised compression starts
  near zero, "which generally never goes back up. **This dooms all high σ runs.**"
- **Short runs mislead.** Ratio loss takes 500+ steps to stabilise; "traditional H-Net
  hparam sweeping is misleading if done with steps < 5000."
- **Batch size cannot be fixed per step** — routing changes how many units reach the inner
  network, so a "step" is not a constant amount of work. This is why FLOPs-matching a
  H-Net is circular (§3 of RESEARCH.md).

**The one intervention that worked** for a wandering compression ratio: a cosine-decayed
**kth-value threshold**. Keep a "fake" `b_t = (p ≥ 0.5)` for the ratio loss and the STE, but
chunk on a "real" `b_t = (p ≥ thresh)` where `thresh` starts at
`(1 + Φ⁻¹(1 − 1/N)/√d)/2` and is annealed to 0.5 by cosine over a cutoff. This flattens the
hockey stick and removes step-0 VRAM spikes. It is the closest thing to a fix for the
failure mode that *When Tokenizers Fail* reports.

---

## 8. Failure signatures to instrument

| symptom | meaning |
|---|---|
| ratio loss ≫ 1 for a long stretch | network wants to send everything to the main net — dying |
| ratio loss far below the floor | network is shutting the main net down entirely |
| no hockey stick in the first ~500 steps | run will be bad; do not scale it up |
| realised compression stuck at 0.4-0.5 | chunker never learned; documented for low-resource settings (boundary rate 0.43-0.49 against a 0.056 target, chunks of 200-1200 bytes) |
| loss rises with width | the threshold hack is in play; LR schedule broke |
| boundaries land at word *starts* | expected behaviour — the decoder must predict each word's first byte unaided |
| `stage1_L` chunks almost every letter on `AAAA…` | known quality artefact in the released checkpoints |

---

## 9. Engineering reality

- The dependency that matters is **`mamba_split_conv1d_scan_combined`** — described as "the
  first major roadblock against efficient H-Net training". `causal_conv1d` additionally
  asserts `d_conv ≤ 4`.
- `flash_attn` is genuinely optional in the independent implementation; rope and RMSNorm
  kernels were reimplemented in plain torch. The only flash-attn-shaped dependency left is
  the varlen attention call, replaceable by cu_seqlens-padded SDPA.
- Nested jagged tensors were **removed** in favour of hand-rolled `cu_seqlens`
  (`cu_outer` for documents, `cu_inner = pad(b.cumsum(0))[cu_outer]` for chunks): "pure
  eager mode execution of NJTs will easily double CPU overhead".
- Cost structure at S=2 small: **~60% non-matmul time per Mamba block**, **~2/3 of total
  time in mamba + dechunk**, **~40% non-matmul time overall**, "disappointingly low" MFU
  relative to a Transformer. A redundant fused-tensor copy pathology alone cost ~20% of
  small Mamba layers.
- Decode is where the architecture gets interesting: when the router says "no boundary",
  the inner network **is not stepped at all**. H-Nets "implicitly implement spec decode /
  adaptive compute" — the per-token compute is genuinely data-dependent at inference.
- Gu's own caveat on efficiency: "we considered FLOPs, not wall-clock time", and "our
  current implementation is still a bit slower than isotropic models during training".

---

## 10. What is still unknown

1. **No formal scaling laws.** Gu: "what we did aren't formal scaling laws"; the paper says
   it did not pursue them "due to resource constraints".
2. **The entropy-vs-NLL boundary rule has never been tested head to head.** It is a
   one-experiment question and nobody has run it.
3. **The crossover point as a function of tokenizer quality is unmeasured.** H-Net++
   (Persian, 1.4B tokens, −0.159 BPB vs GPT-2-fa, 73.8% F1 on gold morphological
   boundaries) is the only evidence in that direction, and it cost ~2,700 A100-hours.
4. **N and α are per-domain.** ATDC exists precisely because "the compression ratio remains
   predominantly manually-tuned"; it proposes a curriculum from low to high N plus the
   analytic N ↔ BPIC relation for tracking chunk size.
5. **CJK boundary behaviour is undocumented.** The 2026 post shows stage-0 chunks that
   respect multi-byte characters in a figure, but publishes no alignment statistic, and
   notes the original LH-Net "doesn't segment characters correctly, even after both stages
   of chunking".

---

## 11. Implications for a small-scale build

1. **Outer networks must be SSMs** — otherwise the mechanism never engages and a null
   result is uninformative. Reuse the `mbdfa` fused scan.
2. **Do not sweep at fewer than ~5k steps.** The ratio dynamics need 500+ steps just to
   stabilise; a 1k-step sweep measures initialisation, not architecture.
3. **Instrument the hockey stick from step 0**, and treat "no spike" as a failed run rather
   than a data point.
4. **Expect high init σ to doom the run**; keep `base_std` small and the router at identity.
5. **Keep `W_res` at zero** and the norm scales trainable; these two are what make the
   residual branch and the main branch commensurate.
6. **Use the kth-value threshold** if the ratio wanders, and report the realised compression
   ratio, not the target.
7. **μP will not transfer** — sweep per arm, at one width, and say so.
