# Kaggle-only dev loop

Type: prototype
Status: resolved
Blocked by: —

## Question

What is the cheapest end-to-end path that proves the whole pipeline works **entirely on
Kaggle**, with no local training run and no GPU quota spent?

The user's constraint: the laptop may not handle it, so everything executes on Kaggle. The
useful fact is that **CPU kernels do not draw on GPU quota**, which makes a CPU-only
prototype effectively free. This ticket produces that prototype and settles what it proves.

It must demonstrate, in one script kernel on a CPU-only session:

1. Data acquisition: stream one shard of `FineWeb2-HQ-Plus` (and `FineWeb-HQ` for English),
   **selecting only the `text` column**, and confirm the text is the expected language and
   is not truncated or HTML-mangled. This is the first real validation of the corpus choice
   made in *Corpus: which datasets, how much, how packed* — HQ-Plus is 17 days old.
2. Packing: documents to a byte stream with explicit document boundaries, written as `.bin`
   plus an index, sized to ~50 MB/language.
3. A 200-step train of the H-Net skeleton on CPU, printing loss, realised compression ratio
   and boundary-probability statistics per step.
4. The output as a Kaggle dataset (or kernel output) that later GPU kernels mount instead of
   re-downloading.

Deliberately out of scope here: correctness gates, real training, any quality claim. A CPU
step is ~100x slower than a T4, so 200 steps is a liveness test, not a learning test.

## Answer

**The path works end to end on Kaggle, for free, and it already paid for itself: three real
bugs surfaced in the first two live runs.**

Kernel: `mauricew/hnet-kaggle-devloop`, CPU-only (`enable_gpu: false` → no GPU quota
touched; 9.72 h left unchanged), internet on, v4 COMPLETE. ~5 min of push latency per
cycle, no wall clock on the session.

**What it demonstrates**

1. *Data acquisition, text column only.* 50 MB × 6 languages streamed from
   `epfml/FineWeb2-HQ-Plus` (`deu_Latn`, `rus_Cyrl`, `cmn_Hani`, `jpn_Jpan`, `arb_Arab`) and
   `epfml/FineWeb-HQ` (eng), via **HTTPS range reads on the parquet footer + column chunks**
   (~60 MB transferred per 460-700 MB shard instead of a full download). Schema read
   successfully; the `embeddings` vector is never fetched.
2. *Corpus validation, and HQ-Plus passes.* Language-score medians 0.936-1.000 with the
   in-language sample correct in all six. HTML tag rate ≤ 2e-5. The packer's invariants are
   asserted every run, not just eyeballed: document offsets strictly increasing and in
   range, state-stream length == byte length, and `train.bin` decodes as strict UTF-8 (so
   the val cut never splits a character). Reported per run in `corpus_report.json`.
   Caveat for the corpus ticket: `eng` from `FineWeb-HQ` is dirty at the document level —
   it opens on cookie-consent boilerplate ("We use the information stored using cookies…")
   and median language score 0.945 is the weakest of the six. Its `2026-04` dump does not
   exist (the latest is `CC-MAIN-2024-51`), so English is a 2024 corpus while the other five
   are 2026 — a recency asymmetry the corpus ticket should record.
3. *Packing.* Per language `train.bin` + `train_docs.u32` (LE u32 offsets) + `train_state.u8`
   (the mbdfa 8-state UTF-8 DFA), plus a 500 KB `val.*` tail. 49.5 MB train per language,
   54k documents total. Reused `mbdfa`'s DFA rather than re-deriving it.
4. *200 steps, health curve, artifacts.* Loss falls 5.54 → 3.07; realised compression
   0.46 → 0.29 chunks/byte (`F`), i.e. the ratio term engages inside 200 steps.
   `health.json` (21 rows + 4 per-language evals), `corpus_report.json`, `smoke.pt`,
   and the packed corpus (573 MB) are all kernel output.

**Bugs the live runs caught** (each fixed, gated, re-verified)

- `bytes_per_chunk` divided `L` by the *batch-wide* boundary count, under-reporting the
  realised ratio by a factor of B. The first run printed ratios 0.5-3.2 where the true value
  was ~2-13 — i.e. it made the compression control look far healthier than it was.
- The alignment probe indexed `bmask[t]` (byte t starts a chunk) against the lead-ness of
  byte **t+1**, shifting a "is this boundary on a character start" measure onto the next
  byte. Signature: rus read **−0.420** and arb −0.446 (boundaries apparently *avoiding*
  character starts), eng read −0.000, because pure-ASCII English has ~100% lead bytes and is
  shift-invariant. `tests/test_alignment_probe.py` reproduces it exactly — an oracle
  boundary set scores +0.530 correctly and −0.033 under the shift — so it cannot silently
  come back.
- The per-language eval was nested inside the `log_every` block, so every scheduled eval was
  skipped except when it collided with a log step (1 eval instead of 4). Now on its own
  schedule.

**What the metrics show, on a liveness basis only**

Alignment lift (`P(lead | boundary) − P(lead)`, per-language on held-out `val.bin`):

| step | eng | deu | rus | cmn | jpn | arb |
|---|---|---|---|---|---|---|
| 49 | +0.003 | −0.015 | +0.343 | +0.173 | +0.512 | +0.296 |
| 199 | −0.002 | −0.014 | +0.404 | −0.078 | +0.438 | +0.372 |

- **eng ≈ 0 by construction**, not as a null: nearly every ASCII byte is a lead byte, so
  there is almost nothing to lift above. Any Q1 design that averages languages unweighted
  will dilute the signal with English. Read English by bpb, not by alignment.
- **deu ≈ 0 for a different reason** — 2-byte chars are ~2% of bytes, so bytes-per-chunk
  (2.9) sits below the character length and boundaries mostly fall inside words.
- **rus, jpn, arb are strongly and stably positive** (+0.37 to +0.51), i.e. the chunker is
  finding character starts on multi-byte scripts within 200 CPU steps.
- **cmn is the interesting one**: +0.173 at step 49, decaying to −0.078 at 199 while
  bytes-per-chunk collapsed 11.6 → 5.1. It starts on whole characters and then fragments
  past character resolution as the ratio target pulls chunks shorter. That is a lead for Q1
  (and for the N choice in *Experiment config*), not yet a finding.

**What this does and does not prove.** It proves the acquisition → pack → validate → train →
artifacts path is executable entirely on Kaggle at zero quota, and that the skeleton trains
and moves all three control quantities. It does not prove quality: 0.447 M params, 200 steps,
one seed, 8 fixed eval windows per language, and a CPU step is ~100x a T4 step. Every number
above is a liveness read.

**Reusable artifacts.** `src/corpus.py` (sources + DFA + range reader + packer),
`src/model.py`, `kernel/train.py`, `kernel/build_kernel.py` (inlines src into the single
`code_file` Kaggle uploads), `kernel/kernel-metadata.json`, `tests/test_alignment_probe.py`.
`kernel/pack.py` is superseded by `src/corpus.py`.

**Follow-on facts for later tickets**

- The packed corpus is kernel output only. *Pack the corpus as a Kaggle dataset* still has
  to push it as a mountable dataset before a GPU kernel can use it — the dev loop proves the
  bytes but does not publish them.
- A T4 kernel needs `pyarrow` too; the kernel already falls back to `pip install` when the
  image lacks it, and the CPU image had to.
- Interpreter on the CPU image is 3.13 with torch 2.11 and **4 CPUs** — plan CPU smoke
  timings at 4 cores, not the local 12.

