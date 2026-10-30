# hnet-kaggle

**Where does a byte-level H-Net's chunker actually put boundaries?**

A Kaggle-only study of learned chunking in a 1-stage byte-level H-Net: a replicated measurement
of character alignment, the falsifier that was fixed in advance, and the four interpretations the
data killed along the way.

**Read this first: [`docs/WRITEUP.md`](docs/WRITEUP.md).**

## The result in one paragraph

At 0.84 M parameters, trained 5000 steps on raw bytes, the chunker places boundaries on UTF-8
character starts above chance in Arabic (+0.266) and Russian (+0.178) — 8/8 seeds positive in
both, p = 0.004 — and below chance in Chinese (-0.147, 7/8 seeds negative). Japanese is
unresolved (6/8 negative, p = 0.14). English and German are structurally unmeasurable on this
metric: 99.6% and 98.5% of their bytes are already lead bytes. The split tracks **bytes per
character** (aligned at 1.78-1.80, not at 2.67) — not whitespace, and not the chunk-length
period. The mechanism is still open.

This is a small-scale result. It is **not** evidence about compute multipliers, the large-byte
regimes, multi-stage H-Nets, or downstream quality.

## What is here

```
src/          model.py (1-stage byte H-Net, pure PyTorch) · corpus.py (sources, UTF-8 DFA,
              ranged parquet reader, packer) · flops.py
tests/        test_gates.py (8 correctness gates) · test_alignment_probe.py (oracle-mask gate)
              eval_noise.py (full-val recomputation + bootstrap CI) · power_analysis.py
              report.py (replication report) · width_vs_delim.py · repl_preview.py
kernel/       dev-loop driver + kernel builders (assembles the single file Kaggle uploads)
kernel_q1/    the Q1 / sweep / ablation / replication drivers
configs/      frozen run configurations (q1.json and its variants)
docs/         WRITEUP.md · ANATOMY.md · ENCODING.md · RESEARCH.md · PLAN.md
.scratch/     the decision record: 16 tickets and the map, with the amendments
```

## Reproducing

```bash
# gates (no GPU, no compiled extensions) — must exit 0
python tests/test_gates.py
python tests/test_alignment_probe.py

# assemble and push the kernels Kaggle will run
python kernel/build_kernel.py            # dev loop
python kernel_q1/build.py configs/q1.json
python kernel_gates/build.py
kaggle kernels push -p kernel_q1
```

The dataset the kernels mount is public: **[`mauricew/hnet-multiscript-bytes`](https://www.kaggle.com/datasets/mauricew/hnet-multiscript-bytes)**
(7 languages + 2 space-ablated variants, per-language `train.bin`, document offsets and 8-state
UTF-8 DFA streams; ODC-BY).

## Why the gates look like this

The reference implementation (`main-horse/hnet-impl`) needs a `mamba-ssm` git fork and
`goombalab/hnet` needs `flash_attn==2.8.0.post2`; neither builds for sm_75, so there is no
reference-diff gate here. It is replaced by an **independent second implementation** of the two
components most likely to be silently wrong — the router and the dechunk EMA — plus a causal-mask
check. That is a weaker gate and is recorded as one.

## Licence

Code: MIT. Corpus: ODC-BY, per the source corpora (epfml `FineWeb2-HQ-Plus`, `FineWeb-HQ`).
