# Artifact scope: what "reusable" means

Type: grilling
Status: resolved
Blocked by: 09

## Question

The destination says "verdict first, then harden the model into something reusable". What
is the artifact, concretely, and what must it do?

Candidates, not mutually exclusive:

- a **Kaggle notebook** that trains, generates and reports boundary alignment in one run;
- a **checkpoint + inference kernel** with a documented generation path (the decode path is
  the interesting part — H-Net skips the inner network when no boundary is drawn, so
  per-token compute is data-dependent);
- a **packaged repo** others can point at, with the gates runnable;
- a **public dataset** (the packed corpus) as the reusable part, model aside.

Decide:

1. Which artifact, and what "works" means for it — the acceptance test.
2. Whether DFA-constrained decoding is in scope (guaranteed-valid UTF-8, free at decode
   time, and the one durable result from `mbdfa`), or a nice-to-have.
3. What gets published vs kept private, and under what licence.
4. What the model card must say about the negative results — that this is an 8M-parameter
   model trained on ~50 MB per language, and is not evidence about scaling.

## Answer

**The artifact is the harness and the corpus, not the checkpoint. The checkpoint is a result,
not a deliverable, and the model as built is not fit to be the reusable thing.**

Decided by what survived scrutiny:

**1. What gets built.** Three artifacts, in priority order:

- **(a) A packaged repo** — `~/Coding/kaggle/hnet` as it stands: `src/model.py` (1-stage byte
  H-Net, pure PyTorch, no compiled extensions), `src/corpus.py` (sources, 8-state UTF-8 DFA,
  ranged parquet reader, packer), `src/flops.py`, `tests/test_gates.py` (8 gates, all green),
  `tests/test_alignment_probe.py` (the oracle-mask gate that catches the off-by-one),
  `tests/eval_noise.py` (full-val recomputation + bootstrap CI), `tests/power_analysis.py`,
  `tests/report.py`, `kernel/` (dev loop, Q1, sweep, ablation drivers) and `kernel_*/build.py`
  (single-file kernel assembly, since Kaggle uploads only `code_file`).
  **Acceptance test:** `python tests/test_gates.py` exits 0 on a CPU box with no GPU, and a fresh
  clone can reproduce `mauricew/hnet-multiscript-bytes` byte-for-byte from `src/corpus.py`.
- **(b) The public dataset** — `mauricew/hnet-multiscript-bytes` (v3: 7 languages + 2
  space-ablated variants), with the README stating sources, shard ids, byte counts, the packing
  format and the known issues. ODC-BY, same as the sources. Currently private; make it public
  when the write-up lands, since a dataset nobody can mount is not a reusable artifact.
- **(c) A one-command reproduction kernel** — `mauricew/hnet-q1-repl` already is it: it mounts
  the dataset, trains N seeds, and emits per-language alignment with CIs. **Acceptance test:**
  it completes unattended and its `q1.json` regenerates the write-up table.

**Not the artifact: the checkpoint.** A 0.84 M-parameter model trained on ~50 MB per language
for 5000 steps, whose ratio control never locked (realised 2.4-7.1 bytes/chunk against a
target of 6) and whose behaviour is mixture-sensitive. Shipping that as a reusable model would
be shipping the least trustworthy part of the work. The checkpoints stay as evidence, not as a
product.

**2. DFA-constrained decoding: out of scope, and the reasoning matters.** It is the one durable
result from `mbdfa` — guaranteed-valid UTF-8 at zero decode-time cost — and the state stream it
needs is already shipped in the dataset. But this effort's claim is about *where the chunker puts
boundaries*, and a constrained decoder would change the boundary distribution that is being
measured. It also needs a decode path we never built (the paper's interesting property is that
the inner network is skipped when no boundary is drawn; that is a generation claim, and we have
no generation result). **Deferred**, with the state stream shipped so it can be added without
re-packing. If it is built later, it belongs on top of this repo, not inside the measurement.

**3. Public vs private.** Dataset and repo public (ODC-BY); kernels private until the write-up
lands, then public. Model checkpoints stay private — see above.

**4. What the model card must say.** There is no model card, because there is no model release.
The write-up must carry these lines instead:

- 0.84 M parameters, 1-stage, byte-level, pure PyTorch, trained on ~50 MB per language
  (≈7% of one epoch), 5000 steps, on a single T4. **Not** evidence about the 1.28x/2.06x
  compute multipliers, the 100B-byte crossover, 2-stage H-Nets, or downstream quality.
- Per-language alignment numbers are **not stable across training mixtures** (Hindi moved 0.27
  between two runs at the same config), so every reported number carries its mixture and seed
  count, and single-seed numbers are reported as unresolved rather than as findings.
- The metric (boundary alignment to UTF-8 character starts) is **structurally unmeasurable on
  near-ASCII scripts** — English and German sit at zero because 99.6% and 98.5% of their bytes
  are already lead bytes. Do not read their zeros as failures.
- Three interpretations were overturned by later experiments within this effort (an off-by-one
  in the probe, a chunk-width hypothesis, a whitespace hypothesis). The write-up should say so
  rather than presenting the final story as if it were the first one.

