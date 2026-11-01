# hnet-kaggle

**What does a byte-level H-Net's chunker actually learn, and does dynamic chunking buy
compute efficiency?**

A Kaggle-only study of learned chunking in byte-level H-Nets, in two parts:

1. **Mechanism** — where does the chunker place boundaries? A replicated alignment study
   across four scripts, two encodings, images and audio.
   **Read: [`docs/WRITEUP.md`](docs/WRITEUP.md)**
2. **Efficiency** — does the compute-efficiency claim (1.28×, "22% less compute") hold at
   1e16 FLOPs? A two-point, two-seed scaling measurement, checked against an independent
   replication. **Read: [`docs/RESULTS_ENG.md`](docs/RESULTS_ENG.md)**

## Part 1 in one paragraph

At 0.84 M parameters, trained 5000 steps on raw bytes, the chunker places boundaries on UTF-8
character starts above chance in Arabic (+0.250) and Russian (+0.169) — positive in every seed —
and below chance in Chinese (−0.165). English and German are structurally unmeasurable on this
metric (99.6% / 98.5% of their bytes are already lead bytes). **And it does not survive
re-encoding**: the same text as UTF-16LE moves the model to the wrong side of the character in
exactly the scripts where UTF-8 aligned (rus −0.228, arb −0.326), and the cross-script ordering
reverses. Bucketing every byte by its offset within its character shows why: the chunker places
boundaries at a **byte phase** that flips between encodings. **Learned chunking is a
byte-phase segmenter, and the router keys on byte-value discontinuities rather than on
characters.** Images and audio agree: boundary structure follows the serialization (RGB vs BGR
agree at chance; PCM little-endian vs big-endian agree only when shifted by exactly one byte —
Jaccard 0.036 against chance 0.114, 0.217 shifted), while alignment to content (edges, onsets)
stays at zero.

## Part 2 in one paragraph

Training the same 18.6 M-parameter byte H-Net at two compute budgets on FineWeb-Edu
(1.92e16 → 3.01e16 realised FLOPs), held-out bpb moved 1.4900 → 1.4173 (two seeds, spread
0.0007): **doubling compute bought 4.9%, not the ~13% the efficiency claim implies**, measured
exponent b = 0.111 vs BLT's 0.206. The verdict lands in the pre-registered FLAT band. An
independent replication (main-horse, 1e16–1e18 on the same corpus family) published curves our
point sits squarely on — implementation validated — and reaches the same conclusion: H-Net
never beats a plain baseline under compute-optimal training, and the blog's 1.28× multiplier
rests on a batch-size (and possibly LR) confound in its BPE baseline. **The architecture works;
the efficiency claim is a large-scale claim that does not replicate at small scale.** Next phase designed in
[AUDIO_PHASE.md](docs/AUDIO_PHASE.md) (LibriSpeech ladder, run after the 2026-11-07 quota
reset).

## What is here

```
src/           model.py (1-stage byte H-Net, pure PyTorch) · corpus.py (UTF-8 DFA packer)
               eng_corpus.py (FineWeb-Edu byte packer) · flops.py · scan_triton.py
kernel/        dev-loop drivers: audio.py · vision.py · eng.py (scaling driver, guards,
               TrainStep) · eng_par.py (parallel launcher, one job per T4) · batch_probe2.py
kernel_q1/     the Q1 / sweep / ablation / replication drivers
kernel_eng/    English scaling kernel (config baked, KERNEL_ID-driven)
kernel_audio/, kernel_vision/   the multimodal alignment arms
tests/         test_gates.py (8 correctness gates) · test_guards.py (10 early-stop tests)
               test_gate_injections.py (re-injects the 4 real bugs, asserts the gates fire)
               test_alignment_probe.py · eval_noise.py · power_analysis.py · report.py
configs/       frozen run configurations (q1, eng-rung, eng-3e16-seeds, …)
tools/         facts.sh (pre-plan fact sheet: devices, quota, corpus — run before any plan
               that spends quota) · poll_run.sh · eval_ckpt.py (canonical 40-batch scorer)
docs/          WRITEUP.md · RESULTS_ENG.md · RESEARCH_SCALING.md · ANATOMY.md · ENCODING.md
               RESEARCH.md · PLAN.md
.scratch/      the decision record: tickets and the map, with the amendments
```

## Reproducing

```bash
# gates (no GPU, no compiled extensions) — must exit 0
python tests/test_gates.py
python tests/test_alignment_probe.py
python tests/test_guards.py
python tests/test_gate_injections.py

# assemble and push the kernels Kaggle will run
python kernel_q1/build.py configs/q1.json     # alignment study
kaggle kernels push -p kernel_eng             # English scaling (config baked from
                                              # configs/eng-3e16-seeds.json)

# score a checkpoint on the canonical eval sample (CPU, zero quota)
python tools/eval_ckpt.py <ckpt> <corpus_dir>
```

Datasets (all public, ODC-BY): [`mauricew/hnet-multiscript-bytes`](https://www.kaggle.com/datasets/mauricew/hnet-multiscript-bytes)
(7 languages + 2 space-ablated variants), [`mauricew/hnet-english-bytes`](https://www.kaggle.com/datasets/mauricew/hnet-english-bytes)
(2.35 GB FineWeb-Edu bytes), [`mauricew/hnet-eng-ckpt`](https://www.kaggle.com/datasets/mauricew/hnet-eng-ckpt)
(best 18.6 M checkpoint), plus the audio and vision byte sets.

## Why the gates look like this

The reference implementation (`main-horse/hnet-impl`) needs a `mamba-ssm` git fork and
`goombalab/hnet` needs `flash_attn==2.8.0.post2`; neither builds for sm_75, so there is no
reference-diff gate here. It is replaced by an **independent second implementation** of the two
components most likely to be silently wrong — the router and the dechunk EMA — plus a
causal-mask check. That is a weaker gate and is recorded as one. The gate-injection tests
compensate from the other direction: each bug this project actually shipped is re-injected and
asserted to be caught.

## Licence

Code: MIT. Corpora: ODC-BY, per the sources (`FineWeb2-HQ-Plus`, `FineWeb-HQ`, FineWeb-Edu,
LibriVox recordings).
