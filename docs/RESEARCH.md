# From Tokens to Bytes — what the claims actually are, and what survives at T4 scale

Research date: 2026-10-29. Sources: Cartesia Post I & II (2026-10-28), H-Net paper
arXiv:2507.07955v2, official code `goombalab/hnet`, independent implementation +
replication `main-horse/hnet-impl`, practitioner notes `main-horse.github.io/hnet/*`,
"All Roads Lead to Likelihood"/`2608.27658` (boundary-ratio failure in low resource),
plus the local `mbdfa/` results from the 2026-10-24 session.

Every number below is attributed. Where two sources disagree, both are recorded.

---

## 0. Correcting the framing from last week

Last week's project (`~/Coding/kaggle/mbdfa`) tested a different lever than H-Nets do.

| | mbdfa (ours, Oct 2026) | H-Net (Cartesia) |
|---|---|---|
| lever | make *byte handling* cheaper (fused Triton scan, DFA mask) | make the *model* need fewer bytes (learned chunking) |
| result | 5.53x end-to-end from the fused scan; DFA mask ≈ null for likelihood | 1.28x compute multiplier vs BPE on validation bpb (their claim) |
| where the win was | memory traffic in the scan kernel | compression ratio handed to the main network |

So "H-Net proves the byte-manipulation idea worked" is not the right read. H-Net does
not touch byte ops at all — it removes 4-8x of the *sequence* the expensive network
sees. The prior project's conclusion ("byte manipulation is nearly free, so it cannot
be what makes the model fast") is consistent with H-Net, not contradicted by it.

The one place the two projects genuinely connect: **H-Net's outer encoder/decoder are
SSMs**, which is exactly what `mbdfa` built and made 5.5x faster.

---

## 1. What the two posts claim

### Post I (scaling ladders, 1-stage controlled comparison)

- Baseline ladder: BPE Transformer, in-corpus 128k tokenizer, 4.57 bytes/token,
  70/20/5/5 English/code/math/multilingual by **bytes**. Rungs 221M → 2.62B params,
  D/L ≈ 75, head dim 128, SwiGLU 3.6D. Muon, WSD.
- Each rung converted to a **1-stage byte H-Net**: inner network identical, outer
  networks FLOP- and parameter-matched to (embedding + unembedding), compression
  matched to 4.57, BPE hyperparameters transferred unchanged.
- Result: **1.28x** compute multiplier on validation bpb, **1.83x** Paloma,
  **2.06x** OLMES; code 1.49x (Markdown) → 2.40x (Go), median 2.04x; OOD code
  2.19x (Haskell) → 5.37x (Prolog).
- 2-stage H-Net (3.67B params, 4.44 GFLOPs/byte) matches SmolLM3-3B's final average
  score after ~1T bytes instead of ~11T — but note the compression means the main
  network sees ~1 position per 7.6 bytes, i.e. ~40% fewer units than SmolLM3, while
  outer stages take **43% of FLOPs** (vs 8% for embedding+head in the baseline).

### Post II (why)

- **Loss-to-loss laws differ by tokenizer.** Contrary to "LLMs on the Line"
  (arXiv:2502.12120, Fig. 5), input encoding changes downstream transfer by 2-4x.
- Byte Transformer has the best validation→OLMES transfer; BPE-NP (no pretokenization)
  the worst; pretokenization buys back 1.26x; BPE still 1.5x off the byte ceiling.
- H-Net recovers the byte ceiling on loss-to-loss while keeping the compression win.
- BLT-like (entropy-heuristic chunking, proxy model not counted in FLOPs) is worse than
  H-Net on both axes → **end-to-end learning of the boundaries is load-bearing**, not
  just "being hierarchical".

---

## 2. The mechanism, precisely (what we would have to build)

Five modules, per stage: encoder → **chunk** → main network → **dechunk** → decoder,
with a residual around the main network.

**Routing** (Eq. 4). `q_t = W_q x̂_t`, `k_t = W_k x̂_t` (initialised to identity),
`p_t = (1 − cos(q_t, k_{t−1})) / 2 ∈ [0,1]`, `b_t = 1{p_t ≥ 0.5}`, `p_1 = 1`.
Boundary = low similarity between adjacent encoder states. Causality is structural.

**Downsample**: keep `x̂_t` where `b_t = 1`; drop the rest. (Ablated vs mean/max/cross-attn
pooling in paper App. E.1 — direct selection wins.)

**Smoothing** (Eq. 5): `z̄_t = P_t z_t + (1 − P_t) z̄_{t−1}` — an EMA over the *chunk*
sequence, with `P` the boundary probs of the retained chunks. The official code runs it
through Mamba-2's chunked scan with `A = −1`, `dt = log(1/(1−p))`, `b = p`, `c = 1`.

**Upsample** (Eq. 6-9): `c_t = p_t` if `b_t = 1` else `1 − p_t`; `STE(c_t) = c_t + sg(1 − c_t)`;
`z̃_t = z̄_{index of last boundary ≤ t}`; output `STE(c_t) · z̃_t`, added to a
`residual_proj(x̂_t)` initialised near zero.

**Ratio loss** (Eq. 10): `L_ratio = N/(N−1) · [(N−1)F·G + (1−F)(1−G)]`, `F` = selected
fraction, `G` = mean `p_t`, `N` = target compression. Total `L = L_AR + α Σ_s L_ratio`,
**α = 0.03** fixed.

**Stability tricks** (Sec 2.3), each independently ablatable:
RMSNorm at the end of every component; projection on the residual path only; outer-stage
LR multipliers **λ₀ = 2.0, λ₁ = 1.5, λ₂ = 1.0**; zero-init residual projection.

**Configs** (Table 1 / `configs/hnet_1stage_L.json`): `arch_layout = ["m4", ["T22"], "m4"]`,
`d_model = [1024, 1536]`, `d_intermediate = [0, 4096]`, `d_state 128`, `expand 2`,
`d_conv 4`, 16 heads, SWA window 1023 on the outer Transformer layers, full attention
in the main network.

---

## 3. The problem: an independent replication says it doesn't hold at our scale

`main-horse/hnet-impl` (MIT, pure-torch + mamba-ssm, flash-attn optional) ran
compute-optimal sweeps at **1e16 – 1e18 FLOPs** on FineWeb-10BT, sweeping width/depth/
LR/batch per (architecture, budget) — a stronger protocol than the paper's fixed-LR,
fixed-batch, FLOPs-matched-by-layer-count design. Best end-of-run bpb:

| FLOPs | BPE (llama2) | byte H-Net S=1 |
|---|---|---|
| 1e16 | **1.52477** | (no H-Net rung reported better) |
| 3e16 | **1.42522** | 1.43770 |
| 1e17 | 1.30214 | **1.29588** (parity) |
| 3e17 | **1.20598** | 1.21250 |
| 1e18 | **1.11295** | 1.12724 |

Their verdict: "none of the H-Net (S>0) runs ever achieve more compute efficiency than
L2 [llama2]"; parity at 1e17, BPE pulls ahead at 1e18. Their methodological critique of
the paper, which matters for how we design our control:

1. **Batch size normalised by bytes, not units.** Optimal batch appears to be similar in
   *chunks/tokens at L₀*, so a byte-normalised batch under-trains the BPE baseline.
2. **LR modulation raises H-Net's effective LR** (λ₀=2.0) while the baseline keeps one
   flat LR, and their own sweeps suggest a byte H-Net wants a *lower* base LR than a
   BPE transformer of the same size. Both biases favour H-Net.
3. **BPIC-dependent FLOPs matching is circular** — the layer count for a FLOPs-matched
   H-Net depends on the compression it will only achieve after training.
4. Spikes in compression ratio mid-run are *not* failures; forcing them down made runs
   worse at equal compute.

Caveats on the replication, stated fairly: English-only, FineWeb only, single dataset,
no downstream evals (bpb only), and their own hparam sweeps are noisy at these scales.

**What this means for us:** the paper's 1.28x is measured at 1e19–1e21 FLOPs with a
protocol that has since been criticised; the one public test at *our* budget found no
win on English. A faithful re-run of Post I's headline is not available to us and would
not be believed if it were — 9.72 GPU-hours this week, 30/week after.

**What is still untested, and is ours to test:** the replication is English-only. The
paper's *largest* multipliers are exactly where a fixed tokenizer is worst — Chinese
(0.43x relative compression), Arabic (0.64x), Russian (0.65x) — and nobody has published
a compute-matched test of "does the H-Net advantage grow as the tokenizer degrades?"
That is a cheap, falsifiable question, and it is the one place a 1e16–1e17 budget can
still produce a real result, because the effect (if it exists) should be biggest there.

---

## 4. Known failure modes we must instrument for

1. **The chunker may simply not learn at small scale.** `2608.27658` reports that under
   the standard ratio loss the boundary ratio sits at 0.43-0.49 for the whole run against
   a 0.056 target, i.e. chunks of 200-1200 bytes instead of ~18. A related ASR port
   reports 15.5% over-compression at N=2 (0.4224 realised vs 0.5 target).
2. **The "hockey stick" is the health signal.** A correct H-Net's ratio loss collapses,
   spikes above init, then settles to target within ~500 steps. Runs that never enter the
   stable regime have bad loss. Ratio loss ≫ 1 = network wants to send everything to the
   main net (dying); ≪ N/(2(N−1)) = network is shutting the main net down.
3. **N and α are domain hyperparameters, not constants.** `N` appears nowhere in the
   released repo (no training script shipped); α=0.03 is reported, not derived.
4. **Boundary placement is systematically "hard"**: at word starts, not ends, which means
   the decoder must predict each word's first byte without main-network help. Expect that
   in our visualisations; it is behaviour, not a bug.
5. **fp16 on T4.** No bf16 on sm_75. The EMA/scan needs fp32 accumulation or it will drift;
   use autocast fp16 + fp32 master weights + GradScaler, and gate on a bf16-vs-fp16 loss
   match the way `mbdfa` gated its Triton scan (fwd 9.5e-07, grad 3.8e-06).

---

## 5. Tooling that exists, and what runs on a T4

| asset | status | T4? |
|---|---|---|
| `goombalab/hnet` (official) | MIT, needs `mamba_ssm` + `flash_attn==2.8.0.post2` | **no** — flash-attn ≥2.5 requires sm_80+; T4 is sm_75 |
| `main-horse/hnet-impl` | MIT, 2.1k lines, NJT varlen, FLOPs accounting, flash-attn *optional*, mamba-ssm required (git fork) | probably — Triton SSM kernels JIT on sm_75; mamba-ssm CUDA ext compiles with Kaggle's nvcc (~10 min) |
| `mbdfa/src/model.py` | our byte SSM + fused Triton scan, bit-exact, 5.53x | yes, validated |
| `mbdfa` DFA state stream | per-byte UTF-8 state for 6 languages | yes — gives us a free, exact measure of "do chunk boundaries land on character starts" |

Two implementation routes:

- **Route A — vendor `hnet-impl`.** Highest fidelity (NJT varlen, LR modulation, FLOPs
  accounting already written by someone who verified it against Cartesia). Risk: mamba-ssm
  build on Kaggle, NJT API churn, no training script (we write the loop).
- **Route B — our own minimal H-Net in pure PyTorch.** ~400 lines: SDPA with block-diagonal
  masks for the varlen main network, conv+SSM or conv+attention outer layers, our own
  chunk/dechunk. No compiled deps at all. Less faithful, fully under our control, and it is
  the thing that becomes "our model".

Recommendation: **Route B for the model, Route A as the reference to diff against** (same
init, same data, one batch, compare loss and chunking on a fixed seed). That mirrors the
`mbdfa` pattern that worked: build it, gate it against a reference, then push to Kaggle.

---

## 6. Compute reality

- T4 fp16 tensor-core peak 65 TFLOPS; realistic 20-40% MFU on 5-30M-param models with
  SDPA and dynamic sequence lengths → **13-26 TFLOPS**.
- Quota: 30 GPU-h/week, resets 2026-10-31T00:00Z. Today: **9.72 h remaining**.
- Therefore: 1e16 ≈ 0.1-0.2 h, 3e16 ≈ 0.3-0.7 h, 1e17 ≈ 1.1-2.2 h, 1e18 ≈ 11-22 h.
- A 3-rung × 2-arm × 2-language ladder ≈ **9-18 GPU-h**. Fits in one quota week with
  margin; a 1e18 rung does not fit at all.

Data: at 1e17 FLOPs with N ≈ 1e7 params, a run consumes ~1.7e9 bytes = 1.7 GB. Our
Tatoeba corpus is 48.6 MB train (eng 108 MB, deu 48 MB, rus 91 MB, jpn 16.5 MB,
cmn 4.2 MB, ara 3.9 MB decompressed). So either the budget drops to ~1e16 per language,
or the corpus grows. Public Kaggle mirrors exist for fineweb-edu (2.6-3.1 GB) and
wikipedia; fineweb-2 covers zh/ja/ar. Kernel internet is available.

---

## 7. Sources

- Cartesia Research, *From Tokens to Bytes I / II*, 2026-10-28.
  `cartesia.ai/blog/from-tokens-to-bytes-part-1`, `-part-2`.
- Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.*
  arXiv:2507.07955v2. Code: `github.com/goombalab/hnet` (MIT).
- `github.com/main-horse/hnet-impl` (MIT) + `main-horse.github.io/hnet/{intuitions,
  scaling-laws-byte,param-s0,param-s1,eng-1gpu}` — replication, negative result at
  1e16-1e18, and the protocol critique quoted in §3.
- `arXiv:2608.27658` — boundary-ratio loss fails to find word boundaries in low resource.
- `github.com/anshulk-cmu/H-Net-Mamba-ASR` — realised 0.4224 vs 0.5 target at N=2.
- `arXiv:2502.12120` (*LLMs on the Line*) — the "tokenizer doesn't matter for loss-to-loss"
  claim Post II refutes.
- Local: `mbdfa/RESULTS.md`, `mbdfa/NOTES.md` (Kaggle CLI + quota + push-over-running).
