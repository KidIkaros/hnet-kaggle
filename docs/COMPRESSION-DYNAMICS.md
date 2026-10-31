# Compression dynamics: a primary-source catalogue of interventions

Research date: 2026-10-31. Question: **what concrete, published interventions exist to make an
H-Net's compression dynamics *settle*** — i.e. to make the chunker's boundary ratio converge to
the target ratio instead of oscillating — and what did each measurably do?

Our symptom (from our own checkpoints, `kernel_q1/`): mean boundary probability `G = 0.305`
against a target of `1/6 = 0.167`; selected fraction `F = 0.219`; ratio loss never below
~1.03; validation bpb 2.27 at step 3200 rising back to 2.46 by step 4800. We are missing the
paper's per-stage LR multipliers.

Every entry below is a *published* intervention with a source. Nothing here is our own
proposal; the "Synthesis" section at the end is explicitly ours and separately labelled.

---

## 0. Source inventory (all read locally unless marked "fetched")

| Source | Handle | What it is |
|---|---|---|
| H-Net paper | arXiv:2507.07955v2 (ICLR 2026) | Hwang, Wang, Gu. The originating paper. §2.2, §2.3, App. C, §3.3 |
| Official code | `github.com/goombalab/hnet` (MIT) | `hnet/utils/train.py`, `hnet/models/hnet.py`, `hnet/modules/dc.py` |
| Replication code | `main-horse/hnet-impl` (MIT) | local clone at `~/.hermes/cache/scratch/hnet-impl` |
| Replication blog | `main-horse.github.io/hnet/*` | Parts 1 (param-s0), 3 (intuitions), 4 (param-s1), 6 (scaling-laws byte) |
| Authors' blog | `goombalab.github.io` | "H-Nets: the Past" / "the Future" |
| Cartesia posts | `cartesia.ai/blog/from-tokens-to-bytes-part-1/-2` | 2026-10-28, authors' own scaling posts |
| SOMBRERO | arXiv:2601.22805v1 | Aleph Alpha. Boundary-placement steering. **Fetched from web** |
| ATDC | arXiv:2605.30080v1 | Adaptive Targeted Dynamic Chunking. **Fetched from web** |
| When Tokenizers Fail | arXiv:2608.27658v1 | H-Net adapted for low-resource Indic. **Fetched from web** |
| H-Net++ | arXiv:2508.05628 | Persian morphologically-rich. **Fetched from web** |
| Dynamic LCMs | arXiv:2512.24617 | Hierarchical concept model, compression-ratio ablations. **Fetched from web** |
| DC-DiT | arXiv:2603.06351 | Diffusion analogue of H-Net's ratio loss. **Fetched from web** |
| ConceptMoE | arXiv:2601.21420 | λ sweep for a MoE-style H-Net ratio loss. **Fetched from web** |
| H-Net-Mamba-ASR | `github.com/anshulk-cmu/H-Net-Mamba-ASR` | Speech port; measured realised compression drift. **Fetched from web** |

Note on naming: our repo's `docs/RESEARCH.md` §7 calls arXiv:2608.27658 "All Roads Lead to
Likelihood". That is a mis-attribution in the existing notes — 2608.27658 is *"When Tokenizers
Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages"*. "All Roads Lead
to Likelihood" is arXiv:2503.01067 (Swamy et al., RL fine-tuning) and is **unrelated** to
compression dynamics. The 2608.27658 paper *is* the low-resource boundary-ratio-failure paper
that RESEARCH.md §4.1 quotes correctly on substance, so the fix is to the title, not the finding.

---

## 1. Interventions with quantitative compression-dynamics effects

### 1.1 Increase the ratio-loss weight α (SOMBRERO) — **helps; the single clearest published fix for a wrong settled ratio**

- **What changes:** only the weight on H-Net's existing ratio loss. The paper's α = 0.03
  becomes ω = 1.0. Nothing else about the loss, the router, or the architecture changes.
- **Source:** arXiv:2601.22805v1, §4.1 Setting, and Table 1 (design ladder).
  URL: `https://arxiv.org/html/2601.22805v1`. Text at §4.1: *"H-Net uses a loss weight of
  ω = 0.03 for the ratio loss. We found that this weight results in a substantial mismatch of
  empirical and target compression rate (C_emp = 3.95 versus C_tar = 5.0), as also observed by
  Hwang et al. (2025). Since such a mismatch ruins the FLOP-matching that was done for the
  target compression rate, we increase the ratio loss weight to ω = 1.0, which results in
  C_emp = 4.97."*
- **Quantitative effect (1B scale, FineWeb-Edu-class mix, 50k steps, 839B bytes, FLOP-matched
  to 1.95 ± 0.01 GFLOPs/byte):**
  - α = 0.03 → realised compression `C_emp = 3.95` vs target `C_tar = 5.0` (21% short).
  - α = 1.0 → `C_emp = 4.97` (0.6% short). **Essentially exact.**
  - Direction: raising α moves the *settled* ratio toward the target, it does not fix an
    oscillation per se — the paper reports the settled values, not the trajectory shape.
  - The Sombrero design ladder (Table 1) reports the H-Net baseline at `C_emp = 4.975` against
    `C_tar = 5.0`, BPB 0.6701, boundary enrichment `B` 1.19; `+Byte-Level Smooth` gives BPB
    0.6571 / `C_emp` 4.986 / `B` 2.612; `-Confidence-Weight` BPB 0.6571 / `C_emp` 4.968 /
    `B` 2.81; `+Sigmoid Chunking` BPB 0.6578 / `C_emp` 4.986 / `B` 2.606; `+CAB Loss (0.01)`
    BPB 0.6568 / `C_emp` 4.999 / `B` 3.035; `H-Net+CAB Loss` BPB 0.664 / `C_emp` 4.977 /
    `B` 2.774. At 5.85 GFLOPs/byte and 2.19B params with `C_tar = 4.6`: `H-Net Scale Up`
    `C_emp = 4.588`, `Sombrero Scale Up` `C_emp = 4.604`.
  - Varying `C_tar ∈ {4, 5, 6, 7}` (FLOP-matched, 20k steps): *"in all cases, the empirical
    compression rate closely matches the target compression rate."*
- **Two caveats we could not resolve from the source, stated rather than glossed:**
  1. **Which ω the Table-1 "H-Net" row used is not stated.** The §4.1 text reports ω = 0.03 →
     `C_emp = 3.95` and ω = 1.0 → `C_emp = 4.97`, but the ladder's H-Net row shows
     `C_emp = 4.975` — i.e. consistent with the ω = 1.0 setting, not the ω = 0.03 one. The
     ladder's H-Net baseline therefore appears to *already include* the weight increase, so
     the ladder measures the *other* Sombrero components on top of ω = 1.0, not on top of the
     paper's α = 0.03. **We could not find a statement resolving this.**
  2. **The paper's "BPB by 0.037" claim does not match its own Table 1.** The text says
     *"Adding CAB to H-Net directly improves BPB by 0.037"*, but Table 1 gives H-Net 0.6701 →
     H-Net+CAB Loss 0.664, a difference of **0.0061**; and H-Net → full Sombrero is 0.0133.
     We quote the number as the paper states it and give the table values alongside; we did
     not find the 0.037 figure reproduced anywhere else in the paper.
- **Author's verdict:** helping. This is the only source we found that reports a *target-rate
  miss at α = 0.03* and a *fix by raising α*, with numbers on both sides.
- **Caution for us:** SOMBRERO measures the mismatch on a 1B model with 839B bytes of training.
  Our model is ~0.8M params on ~1-2GB of data. Their effect is a *settled-state* correction,
  and they do not report a trajectory or step-count-to-settle.

### 1.2 Move the confidence-weighted smoothing from chunk level to byte level (SOMBRERO) — **helps; explicitly framed as a stability fix**

- **What changes:** H-Net's smoothing module (Eq. 5) is an EMA over the *chunk* sequence,
  `ȳ_i = c_{T(i)}·y^B_i + (1−c_{T(i)})·ȳ_{i−1}`, where `c` is the confidence of the chunk's
  *boundary* only. SOMBRERO instead expands the backbone sequence to byte resolution **first**
  (by repetition), then applies the EMA per byte: `x^b_i = c_i·x^k_i + (1−c_i)·x^b_{i−1}`.
  Two consequences: (a) every per-byte confidence `c_i` participates, not just boundary ones,
  so non-boundary positions get gradient; (b) the smoothing operates on equidistant entries.
- **Source:** arXiv:2601.22805v1, §3.3 "Byte-Level Smoothing"; Figure 2 (synthetic
  change-point task); Table 1 row `+Byte-Level Smooth`.
- **Quantitative effect:**
  - Synthetic (Appendix A): with chunk-level smoothing, the sigmoid chunker *"initially
    overcompresses (compression is capped at 8.0) and exhibits high MSE, recovering only
    slowly and unstably"*; with byte-level smoothing *"MSE increases only briefly, compression
    does not overshoot, and the sigmoid quickly recovers and surpasses the cosine
    parameterization."*
  - 1B real: BPB 0.6701 → 0.6571; boundary enrichment `B` 1.19 → 2.612; `C_emp` 4.975 → 4.986.
  - SOMBRERO's stated mechanism for the instability it fixes: chunk-level smoothing *"ignores
    all confidences c_j for non-boundary positions"* and *"does not account for the varying
    expansion ranges"*, so it *"can bias the initial learning signal"*, driving confidences and
    boundary scores down, *"which can result in fewer chunk boundaries and an 'overcompression
    bias' in early training."*
  - They also report that with byte-level smoothing the per-byte confidence re-weighting at
    fusion becomes unnecessary (row `-Confidence-Weight`: BPB 0.6571 unchanged, `C_emp` 4.968).
- **Author's verdict:** helping, and it is presented as the *stability* intervention of the
  paper ("stabilizes boundary learning by applying confidence-weighted smoothing at the input
  level rather than on realized chunks", abstract).
- **Relevance to our symptom:** our `G = 0.305` vs target `0.167` with `F = 0.219` means the
  router is placing boundaries *more often than target* while its probabilities are *also*
  higher than target — i.e. the inverse of SOMBRERO's overcompression bias, but it is the same
  mechanism class: confidence/score geometry that does not match the realised mask.

### 1.3 Curriculum on the *target* compression ratio N (ATDC) — **helps; explicitly aimed at stabilisation**

- **What changes:** the target `N` in the ratio loss is not fixed. `N(t) = N_init` for a warmup
  `t < T_w`, then linearly interpolated to `N_fnl`; after that a performance-driven trigger
  multiplies N by γ = 1.05 when the windowed LM loss falls below a proficiency threshold τ.
  `N_init = 5.0`, `N_fnl = 6.5`, `T_w = 60%·T`, τ = 0.05, γ = 1.05, window W = 100 steps.
  The loss form is H-Net's, re-expressed as a "localized balancing loss"
  `L_bal` with the `N/(N−1)` normalisation kept *"to ensure that the balancing loss remains
  numerically stable and comparable in magnitude across different phases of the N schedule,
  preventing the regularization from dominating the global loss as N grows"*.
- **Source:** arXiv:2605.30080v1, §IV-C, §V-C1–§V-C4, Table I, Table III.
- **Quantitative effect (1-stage byte H-Net, 100B FineWeb-Edu subset, 8×H200, batch 256):**
  - 680M: base BPB 0.783 / BPIC 5.58 → ATDC BPB 0.778 / BPIC 4.52. Avg downstream 50.4 → 50.7.
  - 1.3B: base BPB 0.766 / BPIC 4.35 → ATDC BPB 0.760 / BPIC 4.37. Avg downstream 51.1 → 51.7.
  - 350M ablation, adapting N from 8.0 to 9.5: BPIC 5.51 vs fixed-N 5.11 / 5.42 / 5.68; avg acc
    47.3 vs 46.8 / 46.2 / 47.0. Validation BPB essentially unchanged (~0.828) across all.
  - Robustness (HellaSwag perturbations, 1.3B): base avg 30.8 → ATDC 31.6.
- **Author's verdict:** helping — *"the proposed method exhibits more stable training dynamics
  and superior final performance across diverse downstream tasks compared to models using fixed
  compression ratios"*.
- **Important caveat:** ATDC's BPIC numbers show the adaptive run's realised compression
  (4.52) is *further* from N = 6 than the base run's (5.58) at 680M, so the "stabilisation"
  claim is about training dynamics and downstream accuracy, not about hitting the nominal N.

### 1.4 Compute the ratio loss over *accumulated* micro-batches instead of per-sequence (Dynamic LCMs) — **helps; the closest published analogue to our F/G mismatch**

- **What changes:** the boundary statistics `F` and `G` used in the ratio loss are accumulated
  over `K` micro-batches before the loss is formed, rather than computed per-sequence and
  averaged.
- **Source:** arXiv:2512.24617, §8.2 "Global Regularization via Gradient Accumulation", Table 4.
- **Quantitative effect (2.3B params, 1T tokens, target R = 4):**
  - Realised compression ratio: **3.92 (global) vs 3.15 (per-sequence)**, target 4.0.
  - Downstream avg: global 0.3038/0.6296/0.2457/0.3507/0.3220/0.6806 (ARC-C, ARC-E, CSQA,
    HellaSwag, OBQA, PIQA) vs per-sequence 0.2858/0.6242/0.2228/0.3499/0.3280/0.6785 —
    **+2.1% average improvement**, better on 5 of 6 tasks.
- **Author's verdict:** helping. The paper's framing: *"enforcing a fixed compression ratio per
  sequence is overly restrictive. Real-world data exhibits varying information density. By
  relaxing the constraint to operate at the batch level, the global regularization allows the
  model to learn adaptive behavior."*
- **Relevance to us:** our ratio loss is computed per minibatch. If our F/G are estimated on a
  small batch (our `batch: 16`, `seq_len: 256`), the statistic is noisy and the loss can
  plateau on sampling noise rather than on the true rate.

### 1.5 Replace the cosine router with a linear sigmoid router (SOMBRERO) — **inconclusive / context-dependent**

- **What changes:** `p_i = ½(1 − cos(q_i, k_{i−1}))` → `p_i = sigmoid(W·x^E_i)`. Boundaries still
  `b_i = 1[p_i > 0.5]`, confidence `c_i = max(p_i, 1−p_i)`.
- **Source:** arXiv:2601.22805v1, §3.2 "Chunker"; Figure 2; Table 1 row `+Sigmoid Chunking`.
- **Mechanism claimed:** *"for high-dimensional, randomly initialized representations, cosine
  similarities concentrate around 0, so p_i concentrates near 0.5. Driving scores toward 0 or 1
  then requires precise query/key alignment and thus becomes feasible mainly after the encoder
  learns meaningful features. A simpler linear sigmoid predictor lacks this 'delayed learning'
  effect; we hypothesize that the delay can be helpful when early training dynamics otherwise
  drift toward over- or under-compression."*
- **Quantitative effect:** in the synthetic task the sigmoid variant *"initially overcompresses
  (compression is capped at 8.0) and exhibits high MSE"* under chunk-level smoothing — but
  *"quickly recovers and surpasses the cosine parameterization"* under byte-level smoothing.
  At 1B: `+Sigmoid Chunking` BPB 0.6578 (vs 0.6571 with cosine under byte-level smooth),
  `C_emp` 4.986, `B` 2.606.
- **Author's verdict:** **inconclusive** — the sign of the effect flips depending on which
  smoothing variant is used. It is only safe *given* §1.2.
- **Note:** ConceptMoE (arXiv:2601.21420, §4.5.3) tested the same contrast in a token-level
  MoE setting and found the **linear router better on training loss** (0.003 lower) but **much
  worse downstream** (34.4 vs 36.4 average), attributing it to overfitting. Two sources, two
  settings, opposite sign — treat router parameterisation as unsettled.

### 1.6 Confidence-alignment boundary (CAB) loss (SOMBRERO) — **helps placement, and incidentally tightens the ratio**

- **What changes:** add `L_CAB = mean_t (1 − sg[P_{t+1}] − p_t)²`, where `P_{t+1}` is the model's
  own next-byte probability, both clamped to `[1e−6, 1−1e−6]`. Weight 0.01.
- **Source:** arXiv:2601.22805v1, §3.4; Table 1 row `+CAB Loss (0.01)` and `H-Net+CAB Loss`.
- **Quantitative effect (1B):** H-Net + CAB alone → BPB 0.664 (vs 0.6701), boundary enrichment
  `B` 2.774 (vs 1.19), `C_emp` 4.977. Full Sombrero → BPB 0.6568, `B` 3.035, `C_emp` 4.999.
  Text: *"Adding CAB to H-Net directly improves BPB by 0.037 and substantially increases boundary
  enrichment."*
- **Author's verdict:** helping for boundary *quality*; it is not primarily a ratio-fixing
  intervention. Note it supervises *where* boundaries go, not *how many*.

### 1.7 Init from a pretrained subword backbone rather than randomly (When Tokenizers Fail) — **helps dramatically, in a frozen-backbone setting**

- **What changes:** byte embedding `E[b] = (1/|T_b|)·Σ_{v∈T_b} W_v`, i.e. the mean of the
  frozen backbone's subword embeddings containing each byte value. Plus a chunk-alignment loss
  `L_align = (1/C)Σ_i (1 − cos(z_i, t̄_i))` and interleaved POS supervision (every k = 10
  steps) to give the router structural signal.
- **Source:** arXiv:2608.27658v1, §3.2, §6 ("Analysis"), Figure 3, Tables 3–4.
- **Quantitative effect (Qwen3-1.7B backbone, 20k steps, 500K Hindi sentences):**
  - **Random init:** boundary ratio *"fluctuates between 0.43 and 0.49 throughout training"* vs
    target ρ* = 0.056; chunk length stays 200–1200 bytes vs target 18 bytes/word. BPB starts
    ~9.5, drops to 3.6, then **rises** to plateau at 4.2 with no further improvement over 20k
    steps.
  - **Backbone init:** boundary ratio drops from 0.50 at init to **≈0.05** (target 0.056).
    BPB starts 4.17 and converges smoothly to 3.13 within 7.5k steps (25% relative reduction).
  - POS supervision: chunk-level POS accuracy 43.1% → **59.5%** (+16.4 pts), byte-level
    essentially unchanged (32.5 → 32.8).
  - Alignment loss: without it, chunks-per-word collapses to 0.05–0.13 across six languages
    (each chunk spans 8–17 words) and 1:1 alignment drops to ~0%; with it, 1:1 is 82.5–93.2%.
- **Author's verdict:** helping, decisively — but *in a setting we are not in* (frozen subword
  backbone, low-resource, 500K sentences). The paper's own framing of the failure mode is
  exactly ours: *"the boundary ratio fluctuates between 0.43 and 0.49 throughout training"*.
- **The paper's explicit judgement on the default ratio loss:** *"The standard boundary ratio
  loss (Eq. 3) often fails to learn linguistic boundaries in low-resource settings due to
  insufficient supervision signals."*

### 1.8 Curriculum on sequence length (H-Net++) — **helps; reported for training stability**

- **What changes:** three-stage curriculum — (1) 0–50k steps: fixed 256-byte sequences; (2)
  50k–200k: interleaved {256, 512, 1024, 2048}-byte sequences with probabilities
  {0.4, 0.3, 0.2, 0.1}; (3) 200k+: uniform up to 4096 bytes.
- **Source:** arXiv:2508.05628, "Curriculum Learning" §, Figure 2.
- **Quantitative effect:** H-Net++ reaches BPB 1.183 vs GPT-2-fa 1.342 (0.159 lower, "12% better
  compression"), 5.4pp on ParsGLUE, 69.4% vs 45.3% under ZWNJ corruption, 73.8% F1 on gold
  morphological boundaries, 8–10× compression at the final router level. *"Morphological
  segmentation quality improves across curriculum transitions"* (Figure 4).
- **Author's verdict:** helping. Rationale given: *"Byte-level models struggle with long
  sequences early in training."*
- **Caveat:** H-Net++ is a substantially different architecture (GRU-based router, hyper-prior,
  context-mixer). The *curriculum* is the transferable part.

### 1.9 Cross-entropy vs ratio-loss gradient balance (Dynamic LCMs) — **diagnostic, not an intervention**

- **What it says:** the compression-ratio regulariser is structurally *weaker* than the LM
  objective. `∇L_total = ∇L_CE (anti-compression) + λ∇L_aux (pro-compression)`, and because
  `‖∇L_CE‖ ≫ λ‖∇L_aux‖`, *"the CE loss eventually dominates, forcing the predictor to reduce
  segmentation despite the regularization term."*
- **Source:** arXiv:2512.24617, §8.1, Eq. (24).
- **Observed effect in their ablation:** a *learned* boundary predictor with compression-rate
  regularisation showed *"severe instability: after initial compression to ~2000 tokens, the
  compressed length steadily increases, eventually stabilizing at ~4300 tokens (1.9×
  compression)"* — a slow "creep-up" toward less compression. A *rule-based* cosine predictor
  with a fixed threshold τ stayed at ~2000 tokens (4×) throughout.
- **Why it matters to us:** this is the published statement of the mechanism that most closely
  matches our symptom — a boundary ratio that drifts away from target under CE pressure. Note
  the direction: theirs creeps toward *less* compression, ours sits at *more* compression than
  target (`F = 0.219` vs target `0.167`).

### 1.10 λ sweep for the auxiliary ratio loss (ConceptMoE) — **helps the ratio, hurts the loss**

- **What was swept:** the auxiliary-loss weight λ on a MoE-style, H-Net-derived load-balancing
  loss (two "experts" = boundary vs non-boundary selections), on `ConceptMoE-A0.5B-12B`, 243B
  tokens, target compression R = 2.
- **Source:** arXiv:2601.21420, §4.5.1 "Auxiliary loss weight", Figure 7.
- **Quantitative effect:** λ ∈ {0.03, 0.1, 0.5, 1.0}. *"As λ increases, training loss degrades
  while all configurations achieve compression ratios close to 2."* They therefore set λ = 0.03
  for all other experiments.
- **Author's verdict:** the ratio is *already* satisfied at λ = 0.03 in their setting; larger λ
  only costs loss. **This is the opposite conclusion to SOMBRERO's §1.1**, and the difference
  is the setting: ConceptMoE's target R = 2 was already met at λ = 0.03.
- **Directly relevant to our probe:** our α = 0.1 probe is a point on exactly this axis, and
  this source says λ > 0.03 costs training loss. See the synthesis for how we read that against
  SOMBRERO.

### 1.11 Ratio-loss weight λ = 0.03 by grid search (DC-DiT) — **helps; the ratio is only approximately enforced**

- **What was done:** *"For DC-DiT, the ratio loss weight is set to λ = 0.03 after performing a
  grid search."* The loss is H-Net's Eq. (10), applied to a diffusion backbone.
- **Source:** arXiv:2603.06351, §"Dynamic Chunking Diffusion Transformer" and Table 1.
- **Quantitative effect:** *"the achieved compression ratios are close to but not exactly equal
  to the target N, consistent with the soft nature of the ratio loss"*, at N = 4 and N = 16.
- **Author's verdict:** helping; the residual target miss is reported as intrinsic to the
  soft loss form.

### 1.12 The paper's own four stability tricks (H-Net §2.3) — **reported as helping; one is a boundary-ratio ablation**

- **The four tricks, concretely:**
  1. **Norm balance** — RMSNorm at the *end* of every network component (ℰ^s, 𝒟^s, ℳ), so the
     main network's output does not dominate the encoder's residual. Without it *"outputs from
     deeper components (especially the many-layered main network) would dominate the residual
     signals from earlier encoder networks through imbalanced feature norms."*
  2. **Separation of two streams** — project the *residual* path only (option (i)); the paper
     reports option (ii) (main-network input only) *"disrupts the main network's gradient flow
     and had worse training dynamics"* and option (iii) (both) *"underperforms despite
     additional parameters"*.
  3. **Zero-init the residual projection** — *"this residual connection is initialized close to
     0; earlier versions of H-Net found this to be an important detail, but it may be less
     important when combined with additional techniques such as LR modulation."*
  4. **Per-stage LR multipliers** — λ⁰ = 2.0, λ¹ = 1.5, λ² = 1.0 as a workable default;
     the general rule is
     `λ^s = sqrt(N^GPT · (Π_{i=s}^S N^i)/(Π_{i=0}^S N^i) · D^S/D^s)`, `N^S = 1.0`,
     with `N^GPT = 4.6` (GPT-2 bytes/token on FineWeb-Edu).
  - **Plus the smoothing module and the STE**, which are §2.2 not §2.3.
- **Source:** arXiv:2507.07955v2 §2.3 (Norm Balance / Separation of Two Streams / Learning
  Rate Modulation) and App. C (Learning Rate Modulation). The LR-multiplier default values
  (λ⁰ = 2.0, λ¹ = 1.5, λ² = 1.0) are in App. C; the general rule is Eq. (11) there.
- **Quantitative effect:** the paper gives **no** per-trick compression-ratio numbers. The
  ablation evidence that *is* quantified is on compression stability:
  - §3.3 / Fig. 7 (Large scale, 36B tokens, FineWeb-Edu): removing the **smoothing module**
    → *"compression ratios fluctuate severely throughout training, preventing the model from
    learning consistent chunking boundaries. This instability directly manifests as
    substantial performance degradation."* Removing **cosine routing** or the **STE** is
    *"less critical"* but *"both … exhibit importance in training stability and final
    performance. These components help maintain consistent compression ratios."*
  - §3.1 model comparison: `H-Net (space)` — i.e. H-Net's architecture with a *fixed*
    space-like chunker — improves over `SpaceByte++` because of *"our training improvements to
    the network (Section 2.3), in particular, adding post-network norms, residual projections,
    and learning rate multipliers on the outer networks."*
  - LR modulation's stated effect on chunking specifically: *"since outer stages directly
    influence the chunk boundaries that inner stages depend on, the higher learning rates in the
    outer stages seem to accelerate learning the chunking mechanism."*
- **Author's verdict:** helping (LR modulation: *"the model achieves more stable training
  dynamics and improved convergence behavior across the entire hierarchy"*).
- **Exact multiplier semantics in the official code** (verified locally):
  - `goombalab/hnet`, `hnet/models/hnet.py:150–159` — `_apply_lr_multiplier` applies
    `lr_multiplier[self.stage_idx]` to **every parameter** of that stage, then recurses into
    the main network, so a nested main network gets its own stage's multiplier applied *on
    top of* the outer one. The comment at line 154 says so explicitly:
    *"we apply lr_multiplier to all parameters, and then for the main stage (which may have
    another hierarchy), we just apply it again there."*
  - `goombalab/hnet`, `hnet/models/mixer_seq.py:64–76` — `apply_lr_multiplier` applies
    `lr_multiplier[0]` to the embeddings and to the head, and `self.backbone._apply_lr_multiplier(lr_multiplier)`
    for the rest.
  - `goombalab/hnet`, `hnet/utils/train.py:43–89` — `group_params` groups parameters by their
    `_optim` dict, so `lr` and `lr_multiplier` are per-group; `lr_multiplier` is documented as
    *"A list of learning rate multipliers, one for each stage of the hierarchy, with the outer
    stages first (e.g. [3.0, 1.7, 0.9])."*
  - `main-horse/hnet-impl`, `src/hnet_impl/config_hnet.py:52–58` — the same rule in closed
    form: `(n_gpt * n_prod / self.N_compress[-1] * self.d_model[-1] / d) ** 0.5`, with
    `n_gpt = 4.6` default. `src/hnet_impl/modeling_hnet.py:317–329`
    (`split_params_by_hierachy`) groups params by how many times `.main_network` appears in
    the name — i.e. by depth — which is what an implementation must do to apply per-stage
    multipliers.
  - Note the replication's reading of the rule: main-horse interprets
    `λ^s = sqrt(N^GPT · ΠN^i/ΠN^i · D^S/D^s)` as *"just the application of 2 heuristics"*
    (η ∝ 1/√D and η ∝ √batch), and **removes the `D^S/D^s` factor** on the grounds that their
    μP setup already accounts for it
    (`main-horse.github.io/hnet/param-s1/`, cached at
    `~/.hermes/cache/web/main-horse.github.io-ef49a51748c8c34b.cache.md`, "LR Modulation"
    section).

### 1.13 Sharpen or perturb the boundary probability before thresholding (ConceptMoE) — **helps the ratio, at a training-loss cost**

- **What changes:** instead of a hard `b_n = 1[p_n > 0.5]`, the probability is sharpened
  piecewise and then *sampled*:
  `p^sharp_n = p_n^(1/τ)` if `p_n ≥ 0.5` else `1 − (1−p_n)^(1/τ)`; then
  `b^train_n ~ Bernoulli(p^sharp_n)`. The τ-sharpen makes low-confidence boundaries (near 0.5)
  far more likely to flip, which is the point: it simulates the train/eval distribution shift
  that otherwise flips marginal boundaries.
- **Source:** arXiv:2601.21420, §3.3 "Random flip boundary", Eq. (5)–(6); §4.6 "Boundary Noise
  for Robustness".
- **Quantitative effect:** with `τ = 6` (~4% of tokens flipped) the training/eval compression
  gap closes — *"Evaluation compression ratios remain stable at R = 1.5"* (arXiv:2601.21420
  §4.3) versus R = 1.81 without the correction. §4.6: *"noise regularization normalizes the
  probability mean closer to 1/R (e.g. 0.667), stabilizing evaluation compression ratios."*
  Bernoulli noise *"yield[s] higher training loss at convergence, with smaller τ producing
  larger degradation"*; Gaussian noise (σ = 0.1) *"maintains comparable training loss."*
- **Author's verdict:** helping for the ratio (train→eval stability), at a measurable
  training-loss cost for Bernoulli noise.
- **Relevance to us:** the reported mechanism — that the probability mean sits *below* 1/R
  because *"many boundaries hover near the 0.5 threshold"* — is the mirror image of our
  G = 0.305 sitting *above* target 0.167. In both cases the score distribution is not
  concentrated at the extremes. This is a token-level MoE variant of a router, so the
  mechanism may not transfer, but it is the only source we found that intervenes on the
  *shape* of the score distribution rather than on the *pressure* on it.

### 1.14 The smoothing module and the STE (paper §2.2, ablation §3.3) — **the most load-bearing single component for ratio stability**

- **What it is:** the EMA smoothing module (Eq. 5) and the straight-through estimator
  (Eq. 7), both in the dechunking layer. Removing smoothing replaces `z̃_t = z̄_{Σb_k}` with
  `z̃_t = ẑ_t` (direct upsampling).
- **Source:** arXiv:2507.07955v2 §2.2.2 and §3.3 (Fig. 7).
- **Quantitative effect:** §3.3 gives the qualitative-but-clear statement *"Without this
  module, compression ratios fluctuate severely throughout training"* — this is the paper's own
  statement that smoothing is what makes the ratio settle. No numeric ratio trajectories are
  printed in the HTML; they are only in Fig. 7's middle/right panels.
- **Author's verdict:** *"The smoothing module proves essential for stable training
  dynamics."*
- **Local reference implementation:** `main-horse/hnet-impl`,
  `src/hnet_impl/modeling_hnet.py:91–148` (`DeChunkLayer`) implements the EMA through Mamba-2's
  chunked scan with `A = −1`, `dt = −log(1−p)`, `b = p`, `c = 1`; `modeling_hnet.py:25–36`
  defines the STE; `modeling_hnet.py:210–217` is the ratio loss
  (`keep_expert = n·f·g`, `drop_experts = n(1−f)(1−g)/(n−1)`), matching Eq. (10) with
  `n = N_compress[s+1]/N_compress[s]` (`modeling_hnet.py:161–164`).

---

## 2. Interventions tried and reported as NOT helping

These matter as much as the positives: several are tempting and are already known to fail.

| # | Intervention | Source | Reported outcome |
|---|---|---|---|
| N1 | **Hardcoded / kth-value fixed compression ratio** (replace `p ≥ 0.5` with a global kth-value threshold, i.e. force the ratio directly) | `main-horse.github.io/hnet/param-s1/` "Forced compression" → "Obvious bad approach", cached `main-horse.github.io-ef49a51748c8c34b.cache.md:265–286` | **Helps the ratio, breaks the model.** *"obviously, this breaks causality, and cannot ever be used in a real H-Net run."* It does permit μP transfer in the narrow sense (*"optimal LR doesn't drift away"*), but *"the results are fucked – loss increases with width, instead of the other way around."* |
| N2 | **DeepSeek-style aux-free threshold** (replace the ratio loss with an EMA-of-error threshold update, `u=1e−3, β=0.95`, clamp `[0.1, 0.9]`) | same file, lines 290–316 | **No.** In a single test run it *"made compression factor more stable"* and gave *"no difference in loss"*; in general *"No, because I clamp updates within [.1,.9] for numerical stability, and it is easy for bad runs to increase/decrease p_t far beyond that."* Also *"actual bf16 will quickly saturate and cause routing to become increasingly volatile."* |
| N3 | **Balanced (adaptive) α** — lerp the ratio-loss weight by sign of `(F − N·numel)`, EMA 0.95, step 0.01, clamp `[0.01, 100]` | same file, lines 318–334 | **No.** *"But it does not improve results (1e17 also crashed a bit)."* The author attributes this to the same conclusion as N1: *"forcing a model to do more compression is bad."* |
| N4 | **Cranking the ratio-loss weight to kill compression spikes** | `main-horse.github.io/hnet/intuitions/` "Stability in the long run", cached `main-horse.github.io-5263ce99d2a79784.cache.md:275–281` | **No, and it was the reverse of expected.** After cloning spiking runs with *"ratio loss weight cranked up"*, *"it was always true that the runs with spiking compression would outperform the ones without, in compute-equal cross-entropy."* |
| N5 | **Treating compression-ratio spikes as a failure signal** | same file, lines 265–273 | **Explicitly retracted.** The author's first hypothesis *"Spikes in compression ratio, even in the long run, are always bad"* was revised to: *"Spikes in compression ratio are not proof of a failed run. Sometimes this is due to instability, but sometimes the model is doing what's optimal."* |
| N6 | **Increasing the target compression ratio with model scale** (as a fix for compression drift) | `main-horse.github.io/hnet/scaling-laws-byte/`, cached `main-horse.github.io-2e8af310f855453a.cache.md:114–126` | **No.** *"it does grow, but accounting for that does not help the issue."* |
| N7 | **Architectural interventions aimed at controlling compression growth** | same file, line 126 | **No, abandoned.** *"the analysis above convinced me my H-Nets were failing due to uncontrolled growth in compression ratio, leading to various poor attempts at architectural interventions to control it. Eventually, I concluded my assumptions were wrong."* |
| N8 | **Pretrained encoder / frozen routing** (load a pretrained H-Net's enc+dec, freeze, sweep at D=[1024,1024] with cosine-decayed threshold tracking, then widen) | `main-horse.github.io/hnet/param-s1/` "Pretrained encoder", cached `main-horse.github.io-ef49a51748c8c34b.cache.md:404–418` | **Failed, then partially fixed.** *"Well, no, because I used a parametric output norm, which predictably causes leftwards η_adam drift … But this should be fixable if you replace the norm scale with a down projection instead."* The author's own summary: *"End-to-end H-Nets are a poor fit for μP. Although ugly, separate 'pretrained tokenizers' will work."* He calls the recommendation *"repulsive"* because it abandons end-to-end training. |
| N9 | **Adaptive LR modulation based on the *active* number of tokens seen** (instead of the static N-based rule) | same file, lines 395–402 | **Never attempted.** *"I don't bother attempting this, as it's highly unlikely this adjustment will fix anything, given the variation in batch size (with the thresh update trick) is on the order of ±10%."* |
| N10 | **A threshold-tracking scheme that DID work but only for init** — cosine-decayed kth-value threshold, `ratio=(N−1)/N`, `kth = p_flat.kthvalue(int(ratio*p_flat.numel()))`, `w = cos(π·step/cutoff/2)`, `t = kth.lerp(0.5, 1−w)`, all-reduced; with separate "fake" `b_t` (`p ≥ 0.5`, used for the ratio loss and STE) and "real" `b_t` (`p ≥ thresh`, used for chunking) | same file, lines 336–384 | **Helped only the init, not the long run.** *"compression at init will become much more stable"*, the *"hockey-stick to flatten out"*, and it fixes step-0 VRAM spikes. But *"problematically, there is still a wide variation of compression in the long run"*, and *"lower muon LR → slower ascent of compression to target 0.2. This causes all of the runs below a certain muon LR to strongly underperform."* A step-0 init formula is given: `thresh = (1 + Φ⁻¹(1 − 1/N)/√d)/2`, which *"will either under/overshoot the target, depending on your encoder's init std."* |
| N11 | **Changing the router's LR group** (segregated routing LR for `W_q, W_k, W_res, γ_m`) | same file, lines 387–394 | **Never attempted** — proposed as a way to stabilise the LR search space, *"Of course, this doesn't help if optimal hparams aren't width invariant to begin with."* |
| N12 | **H-Net (pool)** — fixed-width pooling instead of learned chunking | arXiv:2507.07955v2 §3.1, Table 1/2 | **Worse than learned chunking** on bpb and downstream, but this is a chunking-*strategy* ablation, not a dynamics fix. |
| N13 | **Non-Mamba outer networks** (T2-T2 encoder/decoder) | arXiv:2507.07955v2 §3.3, Fig. 8 | **Worse**, and it *"down-compresses sequences poorly compared to other configurations, thus using more compute in the main network"* — i.e. a *compression-efficiency* regression, not a stability one. |

---

## 3. Diagnostics published for the *shape* of the dynamics (no intervention)

Useful because they define what "settled" and "wrong" look like.

- **The hockey stick.** A correctly implemented H-Net *"will (1) rapidly shrink towards zero
  (2) begin a quick spike upwards, often beyond initialization levels (3) gradually taper off
  towards the target compression factor. And all of this happens within the first ~500 steps."*
  — `main-horse.github.io/hnet/intuitions/`, cached
  `main-horse.github.io-5263ce99d2a79784.cache.md:207–215`. The three phases are (i) the
  encoder/decoder's byte-level n-gram gains dominate, so selection probs shrink toward 0;
  (ii) the main network becomes useful and compression *"inverts & overcompensates"*; (iii) once
  the easy gains are used up, driving ratio loss toward 0 dominates and compression declines to
  target (lines 218–226).
- **The expected trajectory of the ratio.** *"starts by randomly selecting L₀/L₁ ≈ 0.4 of all
  byte vectors"* → *"quickly learns to pick almost exactly 1/N bytes, but with ~random
  assignment"* → *"eventually learns reasonable chunking results, that look like words."*
  (same file, lines 200–203).
- **Reading the ratio loss.** *"if ratio loss remains substantially above 1, that indicates the
  network wants to send an excessive number of bytes to the main network … your network is
  dying. If ratio loss is far below 1 (lower bound 2(N−1)/N), then the network is trying very
  hard to shut down the main network entirely."* — `main-horse.github.io/hnet/intuitions/`,
  cached `main-horse.github.io-3fab8748cc681977.cache.md:101–102`.
  **Note the lower bound**: with N = 6, `2(N−1)/N = 1.667`. Our stuck value of ~1.03–1.05 sits
  *between* 1 and that bound, i.e. it is neither "dying" nor "shutting down" by this metric — it
  is simply not falling toward 1.
- **Ratio-loss steps needed.** *"In an average small-scale H-Net run, it takes 500+ steps for
  ratio loss to stabilize close to 1 … traditional H-Net hparam sweeping is misleading if done
  with steps < 5000."* — `main-horse.github.io/hnet/param-s1/`, cached
  `main-horse.github.io-ef49a51748c8c34b.cache.md:205–214`.
- **Init-time compression collapse.** *"high σ + prefix-scan/conv1d causes more correlation
  out-of-the-box, bringing the starting compression ratio to a negligible value, which
  generally never goes back up. This dooms all high σ runs."* (same file, line 196). Directly
  relevant to our symptom: if our initial ratio was already far off target, it may never
  recover.
- **The zero-init residual projection is load-bearing for the hockey stick.** *"With kaiming
  uniform the network never encounters the initial 'hockey-stick' ratio loss spike in favor of
  the main network (higher loss → more tokens in main net)"* — `main-horse.github.io/hnet/param-s1/`,
  cached `main-horse.github.io-ef49a51748c8c34b.cache.md:110` and
  `main-horse.github.io-7224514f832ad472.cache.md:153`. The paper itself says the residual
  near-zero init *"may be less important when combined with additional techniques such as LR
  modulation"* (arXiv:2507.07955v2 §2.3).
- **Realised-vs-nominal compression in a port.** H-Net-Mamba-ASR (speech, N = 2, ratio-loss
  weight β = 0.03) measures a realised keep-fraction of **0.4224 against a 0.5 target — 15.5%
  over-compression**, and states it *"is not corrected by tuning β, since that would break grid
  uniformity."* Source:
  `github.com/anshulk-cmu/H-Net-Mamba-ASR/blob/main/docs/experimental_plan.md`, lines 147 and
  533.
- **"Compression is a contest."** *"What ratio loss wants & what the main network wants are
  different. The latter wins over the former after many steps & compute."* —
  `main-horse.github.io/hnet/intuitions/`, cached
  `main-horse.github.io-5263ce99d2a79784.cache.md:299`.

---

## 4. Synthesis — OURS, not a source claim

Everything in §1–§3 is attributed. This section is our reading. We flag inference explicitly.

### 4.1 What our numbers actually mean, given the published formulae

Our `F = 0.219`, `G = 0.305`, target `1/6 = 0.1667`, N = 6. Using the paper's Eq. (10):
`L_ratio = N/(N−1) · [(N−1)·F·G + (1−F)(1−G)]` = `1.2 · [5·0.219·0.305 + 0.781·0.695]` =
`1.2 · [0.334 + 0.543]` = **1.052**. That matches our observed "ratio loss never below ~1.03"
*exactly*, which is a good check that our implementation of Eq. (10) is faithful.

Now the structural observation: with F = 0.219 and G = 0.305 we have **G > F**, i.e. the router's
*mean* boundary probability is far above its *realised* selection rate, and both are above
target. Because the loss is a product of F and G, and both are large, the loss is nearly flat in
G near this point — the gradient w.r.t. G is `N/(N−1)·[(N−1)F − (1−F)]` = `1.2·[1.095 − 0.781]`
= `0.377`, which is *not* small, but it is multiplied by α = 0.03 (or 0.1) and then has to
compete with the CE gradient. This is exactly the regime Dynamic LCMs describe as
*"anti-compression"* CE gradients dominating a weak regulariser (arXiv:2512.24617 §8.1) —
except in the *opposite* direction from their creep-up.

**The direction of our miss is the same as SOMBRERO's, and this matters for which fix applies.**
Our realised compression is `1/F = 4.57` bytes/chunk against a target `N = 6`, i.e. we
**under-compress** (too many boundaries), and we sit at the ratio loss's *upper* side of the
target. SOMBRERO measured the same shape: `C_emp = 3.95` against `C_tar = 5.0`, again
under-compressed, and fixed it by raising the ratio-loss weight from 0.03 to 1.0 (§1.1).
ConceptMoE's "λ > 0.03 is unnecessary" (§1.10) does **not** transfer to us, because their
target was already met at λ = 0.03. **Our reading, flagged as inference:** we are in
SOMBRERO's regime, not ConceptMoE's, so the direction of the α intervention is the same as
theirs — but their working value (1.0) is ~33× the paper's 0.03, and our α = 0.1 probe is
only 3.3×.

**Our implementation computes the ratio loss per-sequence, and this is a candidate
intervention in its own right.** `kernel_q1/main.py:371–372` computes
`Fs, G = bmask.float().mean(dim=1), p.mean(dim=1)` — i.e. per-sequence statistics — and then
averages the per-sequence losses. That is exactly the variant Dynamic LCMs call *"Normal"*, and
it is the one they measured at realised ratio **3.15 against a target of 4.0**; their
*"Global Parser"*, which accumulates F and G over K micro-batches before forming the loss,
reached **3.92** (§1.4). Our batch is 16 sequences of 256 bytes — so each F and G is estimated
from 256 samples, which is small, and the loss is a mean of 16 noisy per-sequence terms rather
than one estimate over 4096 bytes. This is a cheap, no-architecture-change intervention that we
have not tried.

**A second structural diagnostic, which the paper explicitly predicts and we violate.** The
paper says: *"as our architectural design encourages the routing module to make confident
decisions (i.e. boundary probabilities approaching 0 or 1), F naturally converges toward G, and
the loss effectively guides the model toward the desired compression ratio."* (arXiv:2507.07955v2
§2.2.3). In a well-behaved H-Net, `F ≈ G`. **Ours are not: F = 0.219 vs G = 0.305, a 39% gap.**
That gap is the signature of a router that has *not* learned to make confident decisions — its
`p_t` distribution is still diffuse rather than bimodal at 0/1. It also explains why the loss
is stuck: with `F` and `G` both wrong and *not* equal, Eq. (10) cannot reach its minimum of 1
at `F = G = 1/N`, and there is no single scalar gradient that fixes a mismatch between a
discrete count and a mean. **Inference, flagged as ours:** the missing ingredient may be
"confidence sharpening" rather than "more ratio-loss weight" — every published intervention
that adds *pressure* (higher α, curriculum on N, global statistics) leaves the F-vs-G gap
unaddressed, whereas the two interventions that attack the *shape* of the score distribution
are SOMBRERO's byte-level smoothing (§1.2, which explicitly says it *"reduces confidences (and
increases boundary scores) of non-boundaries"*) and ConceptMoE's boundary-noise/sharpening
(arXiv:2601.21420 §3.3, which sharpens `p` before sampling and reports that noise
*"normalizes the probability mean closer to 1/R"*). **This is a hypothesis to test, not a
finding.**

**Inference, flagged as ours:** the flatness of Eq. (10) in G when F is already too large is
why a modestly larger α may not be enough to move a *settled* F, and why a **global**
(statistics accumulated over micro-batches) ratio loss — §1.4 — is a materially different
intervention from a per-batch one, not just a noisier version of it.

### 4.2 Ranked shortlist of interventions most likely to fix *our* symptom

Ranked by (evidence strength) × (fit to the specific symptom), all from §1.

1. **Per-stage LR multipliers (λ⁰ = 2.0, λ¹ = 1.5), with the official code's semantics.**
   Already in our probe list. Strongest support: the paper lists LR modulation as one of four
   load-bearing stability tricks and says specifically that higher outer-stage LRs *"seem to
   accelerate learning the chunking mechanism"* (§1.12); main-horse independently shows the
   outer/main LR split is what determines how fast compression ascends to target
   (N10: *"lower muon LR → slower ascent of compression to target"*, and *"all of the runs
   below a certain muon LR to strongly underperform"*). **Implement it exactly as
   `goombalab/hnet` does**: one multiplier per stage applied to *all* parameters of that stage,
   with the main network's multiplier applied *on top of* the outer's for nested stages
   (`hnet/models/hnet.py:150–159`). Our `kernel_q1/main.py:229–239` already has an
   outer/main split — check it matches that composition rule rather than replacing.
   *Caution:* main-horse's own sweeps found a byte H-Net wants a **lower** base LR than a
   BPE transformer of the same size, and their scaling-law result is that LR modulation
   *biases in H-Net's favour*. So λ⁰ = 2.0 is a starting point, not a tuned value.
2. **Raise the ratio-loss weight α — and the evidence says go much further than α = 0.1.**
   - *For us:* SOMBRERO measured a real target miss at α = 0.03 (3.95 vs 5.0) and a fix at
     α = 1.0 (4.97) (§1.1). **Our miss has the same shape** — we under-compress
     (1/F = 4.57 vs target 6) exactly as they do — so their fix should transfer. Their
     working value is 1.0, i.e. **~33× the paper's 0.03**; our α = 0.1 probe is only 3.3×.
   - *Against us:* ConceptMoE found λ ∈ {0.03, 0.1, 0.5, 1.0} all reach the target ratio and
     that larger λ *degrades training loss* (§1.10); main-horse's N4 found cranking α up
     made spiking runs *worse* in compute-equal CE; N3 (adaptive α) did not help and crashed a
     1e17 run.
   - **Our reading (inference):** the disagreement is explainable — ConceptMoE's target was
     *already met* at 0.03, so raising α could only cost loss; ours is *not* met, which is
     SOMBRERO's situation. So α up is plausible for the ratio — but N4 and ConceptMoE both
     warn that α trades against CE, and our bpb is already destabilising.
     **Recommendation: keep α as its own arm, but make it a wide sweep, not a point.**
     α ∈ {0.03, 0.1, 0.3, 1.0} with the bpb curve monitored, not just the ratio. If α = 1.0
     moves the ratio to target but costs bpb, that is a *different* trade from what we have
     now (neither a correct ratio nor a stable bpb).
3. **Compute the ratio loss over accumulated micro-batches (global F/G), not per-sequence.**
   §1.4. This is a **new arm**, cheap, no architecture change, and it targets something we
   have not considered: our `kernel_q1/main.py:371–372` estimates F and G *per sequence*
   (256 samples each) and averages 16 noisy per-sequence losses. Dynamic LCMs measured that
   exact variant ("Normal") at realised ratio 3.15 vs target 4.0, and their "Global Parser",
   which accumulates F and G over K micro-batches before forming the loss, reached **3.92**
   (§1.4). Their framing — *"enforcing a fixed compression ratio per sequence is overly
   restrictive"* — is a plausible explanation for why our per-sequence F sits at 0.219 and
   refuses to move: each sequence's loss is individually satisfiable at the wrong global rate.
   **Do this before, or simultaneously with, changing α** — otherwise an α sweep confounds
   two things.
4. **Curriculum on N (ATDC-style), i.e. ramp the target rather than the weight.**
   §1.3 is the only source that combines a *stable trajectory* claim with a *stable settled
   value* and a *reported final BPB*, and it directly addresses a ratio that must travel a long
   way from where it starts. Concrete recipe adapted to us (the mechanism is ATDC's; the
   numbers are ours): we currently sit at realised `N_eff = 1/F = 4.57` against a target of 6,
   so run `N_init ≈ 4.5` and ramp to `N_fnl = 6.0`, with ATDC's trigger (raise N only when the
   windowed LM loss plateaus; ATDC used τ = 0.05, γ = 1.05, W = 100 steps).
   **This is a new arm, not in our current three.**
5. **Byte-level confidence smoothing (SOMBRERO §3.3).** Strong and specifically about
   stabilisation, but it is an architecture change in the dechunk path, and our `G > F`
   asymmetry points at a *different* problem from theirs (their overcompression bias is
   early-training; ours looks like a persistent ratio-vs-score mismatch). Lower priority
   than 1–4, but it is the highest-confidence *stability* intervention if we want one.
6. **Everything else** — CAB loss (helps placement, not the count), sigmoid router (inconclusive,
   two sources disagree), curriculum on sequence length (§1.8, plausible but we have no
   published compression-ratio number for it at our scale), backbone-init byte embeddings
   (§1.7, needs a pretrained subword backbone we do not have).

### 4.3 Already-tried-and-known-not-to-help — do not repeat

- **Do not hardcode or kth-value-force the compression ratio.** It breaks causality and makes
  loss increase with width (N1). It would also make our F/G comparison meaningless.
- **Do not use a DeepSeek-style aux-free threshold.** Clamped threshold + bf16 → volatile
  routing (N2). We are on fp16 on T4, which is strictly worse for this than the bf16 the
  author was already worried about.
- **Do not use an adaptive/balanced α.** It did not improve results and crashed a 1e17 run
  (N3).
- **Do not treat a compression-ratio spike as proof the run is broken.** Explicitly retracted
  (N5); spikes can be the optimal behaviour (N4).
- **Do not spend compute trying to make the ratio *increase* with scale** (N6) or on
  architectural compression-control interventions of the kind main-horse abandoned (N7).
- **Do not assume that raising α is free.** ConceptMoE measured a training-loss cost (§1.10)
  and main-horse measured a compute-equal CE cost (§N4).
- **Do not trust any conclusion drawn from runs shorter than ~5000 steps.** Ratio loss needs
  500+ steps to stabilise and sweeps under 5000 steps *"are misleading"* (§3). Our three probes
  are 3000–4800 steps — i.e. **inside the unreliable window**. This is a design flaw in the
  current probe plan that we should fix before drawing conclusions from it.
- **Do not add boundary noise / random-flip sampling to chase the ratio.** ConceptMoE's
  sharpening-and-sampling (§1.13) closes a *train-vs-eval* compression gap, not a
  train-time ratio error, and it costs training loss at convergence. It also breaks the
  deterministic b_t that our ratio loss and STE both consume.
- **Do not read a ratio-loss number of ~1.05 as "the network is dying".** main-horse's
  diagnostic band is *"substantially above 1"* (dying) vs *"far below 1"* with lower bound
  `2(N−1)/N` (shutting down the main net) (§3). At N = 6 the lower bound is 1.667, so our
  1.03–1.05 sits in the benign band — it means "not converged", not "broken".

### 4.4 Two checks on our own setup that the sources imply, before spending more compute

- **Are we actually running the paper's smoothing module?** The paper's own ablation says
  removing it makes *"compression ratios fluctuate severely throughout training"* (§1.14). Our
  `kernel_q1/main.py:271–291` implements an EMA over main-network outputs with a STE gate,
  which looks like the paper's Eq. 5 — but the *chunk-level vs byte-level* distinction that
  SOMBRERO makes (§1.2) is worth checking explicitly, because the paper's version ignores
  non-boundary confidences.
- **Is our residual projection zero-init?** `kernel_q1/main.py:256–258` does
  `nn.init.zeros_(self.residual_proj.weight)`, matching the official code
  (`goombalab/hnet` `hnet/models/hnet.py:106`). Good. But main-horse notes the paper's
  near-zero init *"may be less important when combined with additional techniques such as LR
  modulation"* — so once we add LR modulation this is no longer a suspect.
- **Did our initial ratio land near target?** main-horse: high init σ can put the starting
  compression ratio at a negligible value that *"generally never goes back up"* (§3). Our
  `G` is 0.305 against 0.167 — we should check the step-0 value.

---

## 5. What we still do not know

1. **No source reports a *trajectory* for the ratio loss after the intervention.** Every
   quantitative result in §1 is a *settled* value (C_emp, BPIC, BPB) measured at end of
   training. Nobody publishes "ratio loss fell below 1.05 by step X". So we cannot predict
   *how long* any of these interventions takes to settle, only where it lands.
2. **No source reports an intervention that fixes an *oscillation* specifically.** Every
   "stabilisation" claim (SOMBRERO §3.3, ATDC) is about a slow drift or a bias, not a
   limit-cycle. main-horse's N4 and N5 establish that spikes are often benign; nobody
   publishes a recipe for distinguishing a benign spike from a fatal one.
3. **The α question is genuinely unresolved in the literature.** SOMBRERO (§1.1) says raise α
   to 1.0 to hit the target; ConceptMoE (§1.10) says λ > 0.03 costs loss and 0.03 suffices;
   main-horse (N3, N4) says adaptive/cranked α did not help and hurt at compute-equal CE. We
   found no source that reconciles these. Our own reading in §4.2 (the disagreement is
   explained by whether the target was already met at α = 0.03) is inference, not a finding.
4. **Nobody reports the ratio dynamics at our scale.** The smallest published H-Net-scale
   compression-dynamics experiments are SOMBRERO at 1B/839B bytes and ATDC at 350M–1.3B on
   100B FineWeb-Edu. Our model is ~0.8M params on ~1–2GB. main-horse's smallest H-Net runs
   are also in this regime but report only *qualitative* shape.
5. **The paper's Fig. 7 compression-ratio panels are the closest thing to published
   trajectories for the component ablations**, and we only have them as images — the HTML
   carries the prose ("fluctuate severely") but no numbers.
6. **Whether the ratio loss should be computed per-sequence or globally is unsettled in the
   *original* H-Net lineage.** The official code comments on it explicitly:
   *"NOTE: This is the loss we used for all experiments. It computes the loss on each
   device/minibatch, and then averages the loss over all devices/minibatches. It is possible
   that computing the loss on each example is better, or that computing the statistics over the
   entire (global) batch would have been better."*
   (`goombalab/hnet` `hnet/utils/train.py:20–22`). Dynamic LCMs (§1.4) tested the global
   variant in a different architecture and found it better; nobody has tested it in H-Net
   proper.
7. **The relationship between "N_compress appearing nowhere in the official repo" and the
   reproducibility of these results.** The official code ships `N_compress` only as a config
   field with defaults `([1], [1,5], [1,3,9])` by hierarchy depth
   (`main-horse/hnet-impl` `src/hnet_impl/config_hnet.py:42–50`), and the training script
   that chose the paper's `N^0 = 6` was never released. So the *exact* target compression
   used in the paper's headline runs is not recoverable from the code we have.
8. **No published source combines LR modulation with a ratio-loss weight sweep.** The paper
   fixes α = 0.03 and sweeps LR; SOMBRERO fixes α = 1.0 and reports one LR schedule. Nobody
   has jointly tuned them, and main-horse's observation that *"lower muon LR → slower ascent
   of compression to target"* makes the interaction likely to matter.
9. **We could not verify the exact "All Roads Lead to Likelihood" citation** in our existing
   notes: arXiv:2503.01067 is *"All Roads Lead to Likelihood: The Value of Reinforcement
   Learning in Fine-Tuning"* and contains nothing about H-Net or compression. The
   boundary-ratio-in-low-resource finding RESEARCH.md cites is real but belongs to
   arXiv:2608.27658. **Recommend fixing that citation in `docs/RESEARCH.md` before it
   propagates.**
10. **No source addresses the "bpb destabilises *after* step 3200 while the ratio is still
    stuck" combination.** The late-training instability that main-horse attributes to
    "spikes in compression ratio are not proof of a failed run" is about compression, not
    about bpb. Our bpb going 2.27 → 2.46 while the ratio does not move is a specific
    combination we have not found characterised in the literature.
11. **No source reports the *F-versus-G gap* as a diagnostic.** The paper states the expected
    healthy regime (F converges toward G as the router becomes confident, §2.2.3) but nobody
    publishes F and G separately at convergence, so we cannot tell whether our 39% F-vs-G
    gap is anomalous for H-Nets in general or specific to our implementation. This is the
    single cheapest missing measurement: log F and G separately, per batch, from any
    reference checkpoint (the official `hnet_1stage_L`/`hnet_1stage_XL` checkpoints are
    public on HuggingFace) and compare.
12. **No source tests the interaction between confidence sharpening and the ratio loss.**
    SOMBRERO's byte-level smoothing (§1.2) and ConceptMoE's boundary sharpening (§1.13) both
    shape the score distribution, and both report a *better* ratio — but neither reports the
    F-vs-G gap before and after, so we cannot tell whether they fixed the ratio by moving
    G toward 1/N or by closing the F/G gap.
13. **Our own α = 0.1 probe is inside the window where the published guidance says results
    are unreliable.** See §4.3: main-horse is explicit that sweeps under 5000 steps are
    *"misleading"*, and our three probes run 3000–4800 steps. Whatever they show about
    *settled* values, they cannot show *oscillation* behaviour, which is the thing we are
    trying to observe.

---

## 6. Citation index (paths and URLs)

Local, read directly:

- `~/.hermes/cache/scratch/hnet-impl/src/hnet_impl/config_hnet.py` — `lambda_s` rule, lines 52–58;
  `N_compress` defaults, lines 42–50.
- `~/.hermes/cache/scratch/hnet-impl/src/hnet_impl/modeling_hnet.py` — RoutingModule (identity
  `W_q`/`W_k` init) lines 70–88; DeChunkLayer / EMA scan lines 91–148; STE lines 25–36;
  `n = N[s+1]/N[s]` lines 161–164; ratio loss lines 210–217; `split_params_by_hierachy` lines
  317–329.
- `~/.hermes/cache/scratch/hnet-impl/src/hnet_impl/lin.py` — `HighPrecLinear` (fp32 residual
  projection, zero-init weight) lines 14–27.
- `~/.hermes/cache/scratch/hnet-impl/README.md` — line 43 (`l_ratio.item()=1.064` at step 0),
  line 45 (`l_ratio.item()=1.016` at step 990): the reference implementation's own ratio-loss
  trace.
- `~/.hermes/cache/web/main-horse.github.io-3fab8748cc681977.cache.md` — intuitions, summary
  extract (ratio-loss reading at lines 101–102).
- `~/.hermes/cache/web/main-horse.github.io-5263ce99d2a79784.cache.md` — intuitions, full
  extract (hockey stick lines 200–230; spike retraction lines 265–281; "compression is a
  contest" line 299).
- `~/.hermes/cache/web/main-horse.github.io-ef49a51748c8c34b.cache.md` — param-s1, full extract
  (init-collapse line 196; 500-step stabilisation lines 205–214; residual zero-init /
  hockey-stick line 110; forced compression N1 lines 265–286; aux-free N2 lines 290–316;
  balanced α N3 lines 318–334; kth-value threshold N10 lines 336–384; routing LR N11 lines
  387–394; dynamic LR modulation N9 lines 395–402; pretrained encoder N8 lines 404–418).
- `~/.hermes/cache/web/main-horse.github.io-7224514f832ad472.cache.md` — param-s1 alt extract,
  residual zero-init at line 153.
- `~/.hermes/cache/web/main-horse.github.io-2e8af310f855453a.cache.md` — scaling-laws byte
  (line 37: LR modulation enabled for S>0, disabled for S=0; line 39: fixed N = [1,5] or
  [1,3,9]; increasing target with scale N6 line 114; abandoned architectural interventions N7
  line 126).
- `~/.hermes/cache/web/main-horse.github.io-b60016f2d5aab871.cache.md` — param-s0 summary.
- `~/.hermes/cache/web/main-horse.github.io-e20b8ec0445db197.cache.md` — engineering (1gpu).
- `~/.hermes/cache/web/arxiv.org-722ba141a12d907b.cache.md` — H-Net paper, heavily elided
  extract; α = 0.03 statement at line 86.
- `~/.hermes/cache/web/arxiv.org-90fe21e7839959dc.cache.md`, `arxiv.org-9481b08701932775.cache.md` —
  H-Net paper abstract/HTML stubs.
- `~/.hermes/cache/web/goombalab.github.io-1fb08049e39362ed.cache.md`,
  `goombalab.github.io-cde0c3fcd414daad.cache.md` — authors' blog.
- `~/.hermes/cache/web/www.cartesia.ai-1d0d6794bc015eca.cache.md` — Cartesia Post I; line 118:
  *"The H-Net's compression ratio is matched to the BPE compression rate (4.57) by using a
  strong auxiliary loss."* — i.e. Cartesia's own 2026 ladder used a **strong** auxiliary loss,
  not α = 0.03. Not quantified further in the post.

Fetched from the web during this session (text saved under
`~/.hermes/cache/scratch/`):

- arXiv:2507.07955v2 full HTML → `hnet.txt` (152 KB); v1 → `hnet_v1.txt`; §2.2.3, §2.3,
  §3.1, §3.3, App. C, App. E.1. URL: `https://arxiv.org/html/2507.07955v2`
- arXiv:2601.22805v1 → `sombrero.txt`; §3.2–3.5, §4.1, Table 1, App. A. URL:
  `https://arxiv.org/html/2601.22805v1`
- arXiv:2608.27658v1 → `bytechunk.md`; §3.2, §6, Tables 3–4. URL:
  `https://arxiv.org/html/2608.27658`
- arXiv:2605.30080v1 → `atdc.txt`; §IV-C, §V-C, Tables I/III. URL:
  `https://arxiv.org/html/2605.30080v1`
- arXiv:2508.05628 → `2508.05628.txt`; Curriculum Learning §. URL:
  `https://arxiv.org/html/2508.05628`
- arXiv:2512.24617 → `2512.24617.txt`; §8.1–8.3, Table 4. URL:
  `https://arxiv.org/html/2512.24617`
- arXiv:2603.06351v1 → `2603.06351.txt`; ratio-loss λ grid search. URL:
  `https://arxiv.org/html/2603.06351v1`
- arXiv:2601.21420 → `2601.21420.txt`; §3.2 (auxiliary loss, device-aggregated statistics),
  §3.3 (random flip boundary), §4.5.1 (λ sweep), §4.5.3 (router), §4.6 (boundary noise). URL:
  `https://arxiv.org/html/2601.21420`
- arXiv:2503.01067 → abstract, to confirm the "All Roads Lead to Likelihood" mis-citation.
  URL: `https://arxiv.org/abs/2503.01067`
- `github.com/anshulk-cmu/H-Net-Mamba-ASR` `docs/experimental_plan.md` → `asr_plan.md`;
  lines 147, 533 (realised 0.4224 vs 0.5).
- Official code, raw files pulled for line-accurate citation: `goombalab/hnet`
  `hnet/utils/train.py` → `goomba_train.py`; `hnet/models/hnet.py` → `goomba_hnet.py`;
  `hnet/modules/dc.py` → `goomba_dc.py`; `hnet/models/mixer_seq.py` → `goomba_ms.py`;
  `hnet/modules/utils.py` → `goomba_utils.py`; `configs/hnet_1stage_L.json`.
  Base URLs: `https://raw.githubusercontent.com/goombalab/hnet/main/…`

Repo-local references (ours):

- `kernel_q1/main.py:229–239` — outer/main LR-mult parameter-group split (already present).
- `kernel_q1/main.py:256–258` — zero-init `residual_proj`.
- `kernel_q1/main.py:271–291` — dechunk EMA + STE.
- `kernel_q1/main.py:309` — `alpha` default 0.03 in `HNetConfig`.
- `kernel_q1/main.py:371–372` — **per-sequence** ratio-loss statistics:
  `Fs, G = bmask.float().mean(dim=1), p.mean(dim=1)` then
  `ratio_loss = ((N/(N-1)) * ((N-1)*Fs*G + (1-Fs)*(1-G))).mean()`. This is the "Normal"
  (per-sequence) variant that Dynamic LCMs measured at realised ratio 3.15 vs target 4.0
  (§1.4).
- `kernel_q1/main.py:3` — the probe config (`alpha: 0.1`, `max_steps: 3000`).

---

## 7. Provenance of this document

Written 2026-10-31 from the sources in §0. §1–§3 are attributions to published work with
sources; §4 and §5 are labelled as our synthesis and our gaps respectively. Where a source is
silent, the document says so rather than inferring. Numbers quoted from web sources were read
from the fetched HTML/PDF in this session (text cached under `~/.hermes/cache/scratch/`), not
from abstracts or search snippets, except where a snippet is the only available form (the
main-horse blog pages are cached in summary-extract form, and the `main-horse.github.io-*`
files with the full extract are the ones cited for the longer passages).

Two corrections to existing repo notes are recorded above and should be applied to
`docs/RESEARCH.md` separately (this document does not edit that file):

1. The arXiv ID `2608.27658` is *"When Tokenizers Fail: Byte-Level Chunking for Zero-Shot
   Transfer to Low-Resource Languages"*, not "All Roads Lead to Likelihood" (which is
   arXiv:2503.01067, about RL fine-tuning). The boundary-ratio finding RESEARCH.md quotes is
   correctly attributed to 2608.27658 in substance; only the title is wrong.
2. RESEARCH.md §3's claim *"Spikes in compression ratio mid-run are not failures; forcing them
   down made runs worse at equal compute"* should be read carefully: the first half is
   main-horse's retracted-hypothesis correction (intuitions, line 273), and the second half
   is his result that runs with spiking compression *outperformed* the non-spiking ones in
   compute-equal CE even after he *"implemented clones of the runs above with ratio loss
   weight cranked up"* (intuitions, lines 275–281). The paper does not state "forcing them
   down made runs worse"; it states that cranking the weight did not stop the spiking runs
   from winning.

