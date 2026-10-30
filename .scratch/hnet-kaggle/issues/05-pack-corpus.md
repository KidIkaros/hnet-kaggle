# Pack the corpus as a Kaggle dataset

Type: task
Status: resolved
Blocked by: 01, 03

## Question

Produce the packed corpus every later kernel mounts — nothing to decide, but the work must
land before any training can be planned against it.

Work:

1. Stream `epfml/FineWeb2-HQ-Plus` configs `deu_Latn`, `rus_Cyrl`, `cmn_Hani`, `jpn_Jpan`,
   `arb_Arab` and `epfml/FineWeb-HQ` for English, selecting **only `text`** (the HQ-Plus
   schema carries a per-document embedding vector that would otherwise multiply download
   size).
2. Take ~50 MB of UTF-8 per language (200 MB if Q2 is later greenlit), preserving document
   boundaries — the packing must record where documents start, not just concatenate.
3. Emit per language: `train.bin`, `val.bin` (held out, ~1 MB), and the UTF-8 DFA state
   stream required by *Alignment metric and DFA stream*.
4. Upload as a Kaggle dataset under `mauricew/`, with a `dataset-metadata.json` and a short
   README stating exact sources, shard ids, byte counts and the packing format.

Resolve with: the dataset ref, per-language byte counts, and the mount path that later
kernels should glob for (remember `/kaggle/input/datasets/<owner>/<slug>/`).

## Answer

**Dataset ref: `mauricew/hnet-multiscript-bytes` (private, ODC-BY). Mount:
`/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/corpus/<lang>/`.**

Verified by mounting it in its own kernel (`mauricew/hnet-mount-check`, COMPLETE) rather than
trusting the upload: all six language directories present, `train.bin` and `val.bin` decode as
strict UTF-8, `train_state.u8` length == `train.bin` length and same for val, document offsets
strictly increasing and in range, and — a check that the offsets and the DFA agree — **every
document offset lands on DFA state 0** (a lead byte). Path resolution uses
`glob.glob("/kaggle/input/**/corpus")`, not a hardcoded path.

| lang | source | train bytes | val bytes | docs in train |
|---|---|---|---|---|
| arb | FineWeb2-HQ-Plus `arb_Arab/000_00000.parquet` | 49,511,655 | 499,999 | 5,390 |
| cmn | FineWeb2-HQ-Plus `cmn_Hani/000_00000.parquet` | 49,506,956 | 499,999 | 10,072 |
| deu | FineWeb2-HQ-Plus `deu_Latn/000_00000.parquet` | 49,502,045 | 500,000 | 11,366 |
| eng | FineWeb-HQ `data/CC-MAIN-2024-51/000_00000.parquet` | 49,502,186 | 500,000 | 12,422 |
| jpn | FineWeb2-HQ-Plus `jpn_Jpan/000_00000.parquet` | 49,500,123 | 500,000 | 12,156 |
| rus | FineWeb2-HQ-Plus `rus_Cyrl/000_00000.parquet` | 49,503,822 | 500,000 | 3,134 |

Total 297 MB train + 3 MB val, 54,540 documents, 573 MB on disk including state streams. Packed
inside `mauricew/hnet-kaggle-devloop` v4 from **one shard per language** via HTTPS range reads,
`text` column only. Document boundaries are a trailing `\n` plus an offset table; the DFA stream
is mbdfa's 8-state table.

**Deviations from the ticket as written, and why**

- Out of the dev-loop kernel, not a separate one: the packer already ran there, so re-running it
  would have doubled the transfer for identical bytes. The dev loop's output *is* the dataset
  source.
- English dump is `CC-MAIN-2024-51`, not 2026: `FineWeb-HQ` has no `CC-MAIN-2026-*` directory
  (latest is 2024-51) and `FineWeb2` has no `eng_Latn` config. English is therefore a 2024 crawl
  while the other five are 2026 — a recency asymmetry now recorded in the dataset README.
- `val` is 500 KB per language, not the 1 MB the ticket says, because the dev loop sized it that
  way. Immaterial for a curve read; raise it if any verdict hinges on val-set variance.

**Known issues carried forward**

- `eng` documents commonly open on cookie-consent boilerplate and have the weakest median
  language score (0.945). Filter or swap before Q2; Q1 can live with it because English is read
  by bpb, not by alignment.
- `arb` has very long documents (5.4k docs / 49.5 MB) versus `cmn`/`jpn` (10-12k). Any
  per-document statistic will be noisier on `arb`.
- Q1-sized only (50 MB/language). Q2 needs a 200 MB repack, which is a new dataset version.

