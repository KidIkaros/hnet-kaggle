# Byte-Level LM Scaling: Compute-Optimal Model Size for a Fixed FLOPs Budget

Research note, compiled 2026-11-01. Question: does the ~20 tokens/param Chinchilla rule transfer to
byte-level models, and what model size is compute-optimal for a byte-level H-Net-style run at
~3e16 FLOPs? All claims are cited to primary sources (arXiv papers, official repos). Where a source
does not state a number, that is said explicitly.

## Bottom line

Per the literature, the most defensible answer for a ~3e16 FLOPs byte-level run is a model in the
**roughly 10M–25M parameter** range. The team's 24M guess is one coherent point in that range, not
an outlier — but it rests on one specific, unproven assumption. Breaking it down:

- **Chinchilla arithmetic, token units**: C ≈ 6·N·D with D ≈ 20·N (Hoffmann et al. 2022,
  https://arxiv.org/abs/2203.15556) → N ≈ √(3e16/120) ≈ **15.8M params, ~316M tokens**. This is an
  extrapolation: Chinchilla's own compute-optimal table stops at 1.84e19 FLOPs (~600x above 3e16).
- **Chinchilla ratio, byte units, project-measured FLOPs constant**: the team's H-Net-style model
  costs 2.53 FLOPs/param/byte (measured), so C ≈ 2.53·N·(r·N) with r = 20 bytes/param →
  N ≈ √(3e16/50.6) ≈ **24.3M params, ~490M bytes**. This is exactly the team's guess; it assumes
  the token-level D/N ≈ 20 rule transfers to *bytes* unchanged, which no published work has tested.
- **BLT's compute-optimal byte anchor**: the only compute-optimal data quantities expressed in
  bytes in the byte-model literature are BLT's BPE baselines: 50B bytes for a 450M-param model and
  400B bytes for a 3.6B model — both ≈ **111 bytes/param** (arXiv 2412.09871, Table 2). Applying
  that ratio: at BPE-equivalent cost (~1.3 FLOPs/param/byte) N ≈ **14.4M**; at the project's
  2.53 FLOPs/param/byte N ≈ **10.3M**; at plain-transformer 6 FLOPs/param/byte N ≈ **6.7M**.
- **The spread (10M–24M) is dominated by the assumed bytes/param ratio (20 vs ~111), not by the
  FLOPs arithmetic.** Both published anchor points are token-level-derived: Chinchilla's 20 was
  fitted on BPE-token models; BLT's 111 bytes/param is Llama-3's token-optimal ratio converted to
  bytes (~4.4 bytes/token × ~25 tokens/param — derived, slightly above Chinchilla's ~20; the
  underlying Llama-3 ratios are not restated in the BLT paper itself). No published work fits a
  scaling law on byte-level
  models' loss-vs-(N, D) surface (see §6), so **there is no literature-grounded way to prefer 20
  over 111 bytes/param for a byte model — this is the key open uncertainty**, and it spans a ~2.3x
  range in optimal parameter count at fixed FLOPs.
- Practical recommendation implied by the sources: pick a size in the 10–25M range, and if the run
  is meant to be compute-optimal rather than overtrained, prefer the *lower* end unless there is
  independent evidence the 20-bytes/param ratio transfers; BLT's own byte models at compute-optimal
  ratios were trained to ~111 bytes/param, and every byte-level model in §2 that trained near 20
  bytes/param did so only in its largest, overtrain-free configurations by accident of budget, not
  by design.

## 1. Does the ~20 tokens/param Chinchilla rule transfer to bytes?

**Chinchilla baseline.** Hoffmann et al. (2022) ("Training Compute-Optimal Large Language Models",
arXiv 2203.15556): from 400+ models (70M–16B params, 5B–400B+ tokens) they conclude that "model
size and the number of training tokens should be scaled in equal proportions" — i.e. D/N roughly
constant as compute grows (https://arxiv.org/abs/2203.15556, abstract and §3). Their FLOPs
accounting is FLOPs(N,D) ≈ 6·N·D, following Kaplan et al. 2020 (§3, Approach 3; also its FLOPs
appendix, which compares against "the common approximation C = 6DN", Table A4). Their own
compute-optimal table (Appendix D.3, Table A3) shows D/N of **19.25–29.2 tokens/param** under
Approach 2 across 400M–10T-param models (400M ↔ 7.7B tokens at 1.84e19 FLOPs; 1B ↔ 20B; 10B ↔
219.5B), and 23.0–142.6 tokens/param under Approach 3 (their Approach-3 ratios grow with scale),
which is where the "~20 tokens/param" shorthand comes from. Caveat: the smallest budget in that
table is 1.84e19 FLOPs; a 3e16 budget is ~600x below it, so using the rule there is
an extrapolation, not an interpolation of their data.

**The unit problem.** In a byte-level model the "D" in C = 6·N·D would be counted in bytes, not
subword tokens. Since BPE tokenizers compress English text ~4–5x (mT5's mC4 averages 4.1 bytes per
SentencePiece token, ByT5 paper §2, https://arxiv.org/abs/2105.13626; H-Net's GPT-2-tokenized
baseline processes 8192 bytes as 1792 tokens ≈ 4.6 bytes/token, https://arxiv.org/abs/2507.07955,
§5 experimental setup), a naive byte-for-token substitution would suggest a byte-level model sees
~4x more "units" per unit of text and should therefore train on ~4x more units per param — i.e.
~80–90 bytes/param rather than 20. Whether that is right is an empirical question the classic
scaling-law papers do not answer directly: neither Chinchilla nor Kaplan et al. ran sub-100M-param
byte-level models.

**Published byte-level models do not follow a 20-bytes/param rule in their training ratios**
(details per paper in §2): ByT5 trained at ~3.5k bytes/param (small) down to ~81 bytes/param (XXL);
MEGABYTE's compute-controlled comparisons gave every model 80B bytes — roughly 0.5–130 bytes/param
depending on which of its listed model sizes (0.6B–173B total) is used; BLT's compute-optimal BPE
anchors imply ~111 bytes/param; H-Net's headline runs trained ~290 bytes/param. None of these were
designed as compute-optimal studies at Chinchilla ratios except BLT's BPE baselines, and the byte
models generally trained far longer (in bytes) than 20 bytes/param. So the honest answer is: **the
rule does not transfer unchanged, and no published work has pinned down the byte-level
compute-optimal ratio** — but the one compute-optimal byte-family anchor point (BLT) is consistent
with ~110 bytes/param, i.e. ~4–5x the token count per param, matching the bytes-per-token
conversion.

**ByT5's 4x statement.** The ByT5 paper states explicitly that "given a fixed input sequence length
and number of training steps, the model will be exposed to roughly 4× less actual text during
pre-training" when training on bytes instead of SentencePiece tokens (arXiv 2105.13626, §1/§2).
This is the cleanest published acknowledgment that byte training consumes compute ~4x faster per
unit of text.

**Vocabulary-scaling view.** Tao et al. ("Scaling Laws with Vocabulary", arXiv 2407.13623) modify
the Chinchilla loss to include vocabulary parameters and measure training data in **characters**
(H) rather than tokens, training 33M–3B-param models on up to 500B characters with vocabularies
4K–96K (https://arxiv.org/abs/2407.13623, abstract & §4.1). Their framework implies the
tokens/param rule is *vocabulary-dependent* (the same text is more tokens when the vocab is
smaller), and they state directly: "Our scaling laws suggest that the limited vocabulary (i.e.,
256 in byte-level language models) may constrain their performance, especially for larger models.
The insight provides a potential explanation for the challenges in scaling byte-level models and
implies that successful scaling of language models may require proportional increases in
vocabulary size" (§ Related Work, "Byte-level language models"). They do not fit a byte-vocab
(V=256) scaling law themselves. A useful token-level anchor from their paper: Llama-2-7B's 2T
tokens was "significantly more than the compute-optimal allocation of a 7 billion parameter model
of around 150B tokens" — i.e. ~21 tokens/param at 7B, consistent with Chinchilla's table.

**H-Net's scaling observation** (not a fitted law): "byte-level H-Net methods start off worse than
the token-level Transformer, but scale better after enough data. H-Net (space), H-Net (1-stage),
and H-Net (2-stage) cross over the tokenized Transformer after just 200B bytes, 100B bytes, and
30B bytes respectively" (arXiv 2507.07955, §5). This is evidence that byte-level models' loss-vs-
data curve has a different slope than tokenized models', but it is not a parametric scaling law.

## 2. Bytes-per-parameter ratios the published byte-level models actually trained at

All ratios below are *derived from* the papers' stated numbers (marked "derived"); the papers
mostly report bytes and params separately, not the ratio.

| Model (source) | Params | Training data seen | bytes/param (derived) | Source |
|---|---|---|---|---|
| ByT5-Small (300M) | 300M | ~1.05e12 bytes (1M steps × 2^20 bytes/batch, seq len 1024 bytes) | ~3,500 | https://arxiv.org/abs/2105.13626 §2 ("train for 1 million steps over batches of 2^20 tokens"; seq length 1024 bytes) + Table 1 |
| ByT5-XXL (12.9B) | 12.9B | same total bytes | ~81 | same |
| MEGABYTE text models | global/local 452M/151M, 5.8B/604M, 170B/3.2B (baselines "based on GPT3" sizes) | 80e9 bytes per model (compute- and data-controlled experiments) | not stated; ~0.5–130 depending on model | https://arxiv.org/abs/2305.07185 Table 1, §6.1 ("Each model is trained for 80 billion bytes") |
| BLT BPE baselines (compute-optimal, per Llama-3 ratios) | 450M / 3.6B (inference-flop-matched classes) | 50e9 / 400e9 bytes | ~111 | https://arxiv.org/abs/2412.09871 Table 2 |
| BLT 8B (beyond compute-optimal) | 8B total (global latent transformer 6.4B per Table 10) | 4e12 bytes (1T tokens/4T bytes) | ~500 | https://arxiv.org/abs/2412.09871 §1, §5.1, Table 10 |
| H-Net (2-stage) XL | 1.6B | ~460e9 bytes (100B-token FineWeb-Edu subset at ~4.6 bytes/token) | ~290 | https://arxiv.org/abs/2507.07955 §5 setup, Table 1 |
| SpaceByte sweep | ~200M–800M (non-embedding, per Table 3/5) | varied (Figure 4 plots the full 10–1000 bytes/param axis) | 10–1000 (sweep range) | https://arxiv.org/abs/2404.14408 Figure 4, Table 3 |

Notes and caveats on this table:

- **SpaceByte**: the 10–1000 bytes/param range is the axis of its Figure 4 Pareto plot, not a
  single stated ratio; its headline compute-controlled comparison trains all architectures with
  10^19 FLOPs per run (Table 1 caption), with batch size 64 and byte context lengths of 1024–2048
  bytes (Appendix B.2, Table 3).

- **ByT5**: parameter counts from Table 1 (Small 300M ... XXL 12.9B, all "Vocab ~0%"). The total
  pretraining bytes figure (~1.05e12) is derived from "1 million steps over batches of 2^20
  tokens" combined with the byte sequence length; the paper does not state a single "total bytes
  seen" number, and mC4's multilingual byte/token ratio averages 4.1 (§2), so the *text* covered
  is ~4x less than mT5's 1.05T tokens.
- **MEGABYTE**: the "80 billion bytes" figure comes from the controlled experiments (Table 2), where
  "each model is trained for 80 billion bytes, and models are scaled to use the same compute budget".
  For the large PG-19 scale-up (Table 3) the paper states sequence length 8192 bytes and batch 256
  but (in the main text) does not state total training bytes.
- **BLT**: Table 2 reports, for two inference-FLOP classes, the BPE model's *compute-optimal*
  training data quantity in bytes: 50B bytes for the 450M-param class and 400B bytes for the 3.6B
  class — both ≈111 bytes/param. BLT's own 8B model was deliberately trained beyond compute-optimal
  (1T tokens / 4T bytes, following the Llama-3-style overtrained regime; the paper notes Llama 3.1
  8B was "trained on two orders of magnitude more data than what is compute-optimal for that model
  size", §5.2). BLT models in the scaling study were trained "for the optimal number of tokens as
  computed by Dubey et al. (2024)" — the Llama-3-determined ratio — i.e. the byte models were
  matched to the same byte budgets as the BPE baselines (§5.1: "All the models see the same number
  of bytes in each sequence during training and inference in expectation").

**Conclusion of §2:** published byte-level works trained at wildly different bytes/param ratios
(~0.5 to ~3,500), and none of them did a compute-optimal *byte-level* sweep that would directly
replace Chinchilla's 20. The only compute-optimal numbers expressed in bytes are BLT's BPE
baselines: **~111 bytes/param**.

## 3. Chunk-based architectures: FLOPs per parameter per byte, and does that shift the optimal ratio?

- **MEGABYTE** (arXiv 2305.07185) is the original statement of the argument: with patch size P, "a
  baseline transformer would use the same feedforward inputs a contextualized patch representation
  ... with m parameters P times, MEGABYTE can use a layer with mP parameters once for the same
  cost" — i.e. chunking frees compute to spend on a larger model at the same byte cost. Their
  headline consequence: "a MEGABYTE model with 1.5B parameters can generate sequences 40% faster
  than a standard 350M Transformer, whilst also improving perplexity when trained with the same
  compute" (§2.2/§3). So a chunked model has *lower FLOPs per parameter per byte*, and can be
  larger at fixed FLOPs.
- **BLT** (arXiv 2412.09871) quantifies this at scale: "When growing total parameters 20x from
  400M to 8B, we only roughly double BLT's local model parameters" (§5.1) — the local byte-level
  models stay small and the global (latent) transformer runs per-patch. BLT follows Chinchilla's
  FLOPs equations for the transformer blocks and provides the exact per-component equations
  (encoder/decoder/latent/cross-attention) in Appendix B, with patch size and bytes-per-patch as
  parameters (Table 1 reports 4.4–6.1 bytes/patch on their train mix).
- **H-Net** (arXiv 2507.07955) reports measured **GFLOPs/Byte** at evaluation for all its models
  (Table 2): BPE-tokenized Transformer (760M) 0.42 GFLOPs/byte; SpaceByte 0.41; H-Net (1-stage,
  680M) 0.43; H-Net (2-stage, 870M) 0.43; at XL scale, Transformer (1.3B) 0.69 and H-Net (2-stage,
  1.6B) 0.69. Interpretation (derived): at *matched total FLOPs*, H-Net spends about the same
  FLOPs/byte as the BPE transformer while being byte-level — the win is equivalence, not savings —
  and it packs ~1.5–3x more parameters into the same FLOP envelope than a byte-level plain
  transformer could (e.g. H-Net 1-stage 680M vs LlamaByte 210M vs MambaByte 190M at the same
  FLOP budget, Table 1).
- **Does the optimal data-to-parameter ratio shift?** In Chinchilla's framework the optimal D/N
  ratio depends on the loss decomposition's exponents (N^α vs D^β terms), while the FLOPs constant
  (6) only sets the C = s·N·D constraint. **No published scaling-law study estimates α, β for
  byte-level or chunk-based models** — Chinchilla fitted token-level BPE models; BLT fits
  scaling *trends* (BPB vs FLOPs curves) for byte models but does not publish a Chinchilla-style
  (N_opt, D_opt) parameterization (§5.1). Therefore any shift in the *ratio* for chunk-based
  architectures is unsupported by direct evidence; what the literature does support is a shift in
  the **FLOPs-per-parameter-per-byte constant** (s ≈ 6 FLOPs/param/token ÷ ~4.6 bytes/token ≈ 1.3
  for a BPE transformer; ~0.6–0.9 for H-Net-style chunked models in training-FLOPs terms, derived
  from the 0.43 eval GFLOPs/byte × 3 for backward+optimizer). For the team's own measured 2.53
  FLOPs/param/byte (H-Net-like 18.5M-param model), the arithmetic in §5 applies Chinchilla's
  *ratio* under the project's *own* measured constant — a defensible but explicit assumption.

## 4. Batch sizes, sequence lengths, and MFU reported by reference implementations

- **BLT** (arXiv 2412.09871, §4.7/§5.1): batch size held at **16M bytes per batch** (average);
  context **8k bytes** on Llama-2 data, **16k bytes** on the BLT-1T dataset; dynamic patching
  changes the bytes-to-patches ratio per batch. BLT's paper reports no MFU figure.
- **H-Net** (arXiv 2507.07955, §5): **batch size 256 sequences**; byte models process **8192
  utf-8 bytes per sequence**, the BPE baseline 1792 tokens (~8192 bytes), total batch ≈0.5M tokens
  (~2.1M bytes) "roughly matching protocols from prior work (Gu and Dao 2024)". The paper does not
  report MFU, GPU model, or wall-clock training time in the extracted text.
- **MEGABYTE** (arXiv 2305.07185): controlled experiments 80B bytes; PG-19 scale-up: sequence
  length 8192 bytes, batch 256 (Table 3 context); audio experiment: sequence length 524,288,
  patch size 32, batch size 32 (§6.5/Table 5). Trained with the Metaseq codebase. No MFU reported.
- **ByT5** (arXiv 2105.13626): sequence length **1024 bytes** (vs 1024 tokens for mT5), 1M
  pretraining steps, batches of 2^20 tokens(≈bytes at seq 1024 × 1024 seqs); fine-tuning batch
  2^17 tokens default. No MFU reported.
- **Small-batch/SGD study** (Marek, Lotfi & Somasundaram, "Small Batch Size Training for Language
  Models", arXiv 2507.07101): trains a 30M-param GPT-2-style model on **600M tokens of
  FineWeb-Edu** (Chinchilla 20 tokens/param) at context 512, and reports that batch sizes 1–2
  caused "roughly a 30–70% drop in MFU" vs larger batches on TPU v4; recommends "the smallest
  batch size that maximizes model throughput (… MFU)" with "at least several hundred tokens per
  device" (§4, Table 2). This is the closest published small-scale anchor to the team's regime,
  though it is token-level and TPU-based.
- **T4-class MFU measurement** (first-party reproduction, not a paper): a public reproducible
  experiment training a nanoGPT-shaped ~30M-param model on a single Tesla T4 measured **MFU ≈ 12%
  at batch 8, ranging ~2% (batch 1) → ~17% (batch 48)**, with GPU-Util at ~97–100% throughout, and
  fp16 2.6x faster per step than fp32 (https://github.com/sanmati1997/gpu-utilization-vs-mfu,
  "Findings (Tesla T4 reference run)"). This is consistent with the project's 11.3%-MFU anchor and
  the project's own 2.8% measurement (2.8% sits between the batch-1 and batch-8 points of that
  sweep; the repo attributes the spread to configuration, not hardware).
- **Reference-implementation configs**: the official H-Net repo (https://github.com/goombalab/hnet)
  releases configs only for the paper's Large (760M) and XL (1.3B) scales
  (`configs/hnet_{1,2}stage_{L,XL}.json`) — no small-model configs. The official BLT repo
  (https://github.com/facebookresearch/blt) states models "up to 8B parameters and 8T training
  bytes" in its README. Neither repo publishes MFU or T4-class throughput numbers.
- **T4-class hardware:** none of the arXiv papers above report MFU on T4s (BLT/H-Net/MEGABYTE used
  larger clusters). The 11.3% T4 MFU figure quoted in the project brief was not located in any of
  the papers read; the closest published-style measurement is the T4 experiment cited above
  (~12% at batch 8). The team's own 2.8% MFU is well below every reference point found.

## 5. Compute-optimal model size at ~3e16 FLOPs for byte-level training (cited range)

Arithmetic (derived, shown so it can be checked; the *inputs* are cited):

- Chinchilla: FLOPs ≈ 6·N·D with D_opt ≈ 20·N → N_opt ≈ sqrt(C/120). At C = 3e16: **N ≈ 15.8M
  params, D ≈ 316M tokens**. (Chinchilla's own table stops at 1.84e19 FLOPs, so this is an
  extrapolation; the paper itself notes "significant uncertainty extrapolating out many orders of
  magnitude", §4.1.)
- If the byte-level budget is counted in *bytes* at a Chinchilla-like 20 bytes/param and the
  architecture's FLOPs constant is s = 2.53 FLOPs/param/byte (project measurement for the H-Net-style
  model): N_opt ≈ sqrt(3e16 / (2.53·20)) ≈ **24.3M params, ~490M bytes**. This is exactly the team's
  24M guess — it is the Chinchilla *ratio* applied under the project's *own* FLOPs constant, which
  is defensible but rests on assuming the token-level D/N ≈ 20 transfers to bytes.
- If instead the byte ratio follows BLT's BPE compute-optimal anchor (~111 bytes/param; Table 2 of
  arXiv 2412.09871), at the BPE-equivalent constant s ≈ 1.3 FLOPs/param/byte: N ≈ sqrt(3e16/(1.3·111))
  ≈ **14.4M params**; at the project's s = 2.53: N ≈ sqrt(3e16/(2.53·111)) ≈ **10.3M params**.
- So the literature-supported **range is roughly 10M–25M parameters** for 3e16 FLOPs byte-level
  training: ~10–14M if you trust the BLT-derived ~111 bytes/param, ~16M at plain Chinchilla
  arithmetic, ~24M if you combine the 20-units/param rule with the chunked architecture's lower
  FLOPs/param/byte. The spread is dominated by the bytes/param ratio assumption, not the FLOPs
  arithmetic.

## 6. Is there a published scaling law specific to byte-level or patch-based models?

- **BLT** is the closest: "the first flop-controlled scaling study of byte-level models up to 8B
  parameters and 4T training bytes" (arXiv 2412.09871, abstract). It fits BPB-vs-FLOPs scaling
  trends for byte models at compute-optimal ratios and finds BLT overtakes BPE beyond a crossover
  (Table 2: crossover at 150B bytes for the small class, 1T bytes for the large class — 2.5–3x the
  BPE compute-optimal byte budget). But it publishes trends and crossover points, **not** a
  fitted (N_opt, D_opt) scaling law.
- **SpaceByte** (Slagle, arXiv 2404.14408) treats bytes-per-parameter as a first-class axis: its
  Figure 4 plots "bits-per-byte vs the number of bytes used for training divided by the number of
  non-embedding parameters" on a log axis spanning ~10–1000 bytes/param, across byte-level and
  subword baselines (https://arxiv.org/abs/2404.14408, Appendix A/Figure 4 caption). It compares
  Pareto frontiers at matched training and inference FLOPs but does not fit an optimal ratio.
- **Scaling Laws with Vocabulary** (arXiv 2407.13623) fits a Chinchilla-style loss with an extra
  vocabulary-parameter term and data in characters — the closest published parametric law that
  treats near-character-level data units — but its smallest vocabulary is 4K, and it does not fit
  a V=256 byte-level case (§4.1, Appendix A.8).
- **Hourglass / hierarchical-compute allocation** (Nawrot et al., "Hierarchical Transformers Are
  More Efficient Language Models", arXiv 2110.13711) shows pooled downsampling/upsampling between
  layers beats a flat Transformer "given the same amount of computation" (abstract, v2) — the
  architectural premise H-Net/BLT build on — but it predates compute-optimal-ratio analysis and
  reports no bytes/param optimum.
- **No published Chinchilla-style parametric scaling law (A/N^α + B/D^β with fitted exponents) for
  byte-level or patch-based models was found** in the sources read (checked: Chinchilla and its
  cited predecessor Kaplan et al. 2020, BLT, MEGABYTE, H-Net, ByT5, SpaceByte, Hourglass, and the
  vocabulary-scaling paper). This appears to be a genuine gap in the literature.

## Sources

Primary papers (arXiv):

1. Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), arXiv:2203.15556 — https://arxiv.org/abs/2203.15556
2. Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens* (BLT), arXiv:2412.09871 — https://arxiv.org/abs/2412.09871
3. Yu et al., *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*, arXiv:2305.07185 — https://arxiv.org/abs/2305.07185
4. Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (H-Net), arXiv:2507.07955 — https://arxiv.org/abs/2507.07955
5. Xue et al., *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models*, arXiv:2105.13626 — https://arxiv.org/abs/2105.13626
6. Slagle, *SpaceByte: Towards Deleting Tokenization from Large Language Modeling*, arXiv:2404.14408 — https://arxiv.org/abs/2404.14408
7. Tao et al., *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*, arXiv:2407.13623 — https://arxiv.org/abs/2407.13623
8. Marek, Lotfi & Somasundaram, *Small Batch Size Training for Language Models*, arXiv:2507.07101 — https://arxiv.org/abs/2507.07101
9. Nawrot et al., *Hierarchical Transformers Are More Efficient Language Models* (Hourglass), arXiv:2110.13711 — https://arxiv.org/abs/2110.13711

Official repositories:

10. goombalab/hnet (official H-Net code, configs L/XL only) — https://github.com/goombalab/hnet
11. facebookresearch/blt (official BLT code) — https://github.com/facebookresearch/blt
12. sanmati1997/gpu-utilization-vs-mfu (reproducible Tesla-T4 MFU measurements, ~30M-param GPT) — https://github.com/sanmati1997/gpu-utilization-vs-mfu

Note on versions: all papers were read as the arXiv PDFs linked above at their then-latest
versions: Chinchilla 2203.15556v1, BLT 2412.09871v1, MEGABYTE 2305.07185v2, H-Net 2507.07955v2,
ByT5 2105.13626v3, SpaceByte 2404.14408v3, vocabulary paper 2407.13623v3, small-batch paper
2507.07101v4. Quotes were verified against pdftotext extraction of these PDFs.
