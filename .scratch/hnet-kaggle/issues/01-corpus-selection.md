# Corpus: which datasets, how much, how packed

Type: research
Status: resolved
Blocked by: —

## Question

Which corpus do we train on, given: six scripts (eng, deu, rus, cmn, jpn, arb), a
controlled cross-script comparison, ~50-200 MB per language, download and packing must
happen inside Kaggle, and a stated preference for recent (2026) datasets?

## Answer

**Chosen: epfml `FineWeb2-HQ-Plus` for deu/rus/cmn/jpn/arb + epfml `FineWeb-HQ` for eng.**

Why this pair:

- **One filtering family across all six scripts.** Both are epfml HQ corpora built on the
  same quality pipeline, so a cross-language difference is not a corpus-quality artefact.
  Mixing pipelines across languages would confound exactly the variable we are measuring.
- **Newest usable multilingual corpus.** `FineWeb2-HQ-Plus` last modified 2026-10-12 (17
  days before charting), 100 languages, per-language configs (`deu_Latn`, `rus_Cyrl`,
  `cmn_Hani`, `jpn_Jpan`, `arb_Arab`), ungated, ODC-BY. Sizes: deu 142.6 GB, rus 326.7 GB,
  cmn 187.4 GB, jpn 103.1 GB, arb 19.8 GB across 337/840/680/315/45 shards.
- **Cheap to sample.** Shards are ~461 MB and per-language, so one shard per language
  (~2.3 GB total) covers 50-200 MB of packed text with room to spare.
- **Schema is clean**: `text, id, date, dump, language, language_score, quality_score, url`.
  Note it also carries a per-document `embeddings` vector — **select only `text`** when
  streaming, or downloads inflate several-fold.

`FineWeb-HQ` (2025-09-30, 1.4M downloads, ODC-BY) supplies English: `FineWeb2` has **no
`eng_Latn` config** (verified: zero eng configs among its 3,740 splits), and FineWeb-HQ is
the same family's English corpus. Shards are ~2.1 GB organised by CC dump
(`data/CC-MAIN-YYYY-NN/`), so take one dump's early shards.

**Rejected:**

| candidate | why not |
|---|---|
| Tatoeba 6-language 48 MB (already on Kaggle as `mauricew/mbdfa-multiscript-bytes`) | too small for anything but a smoke test; sentence-per-line, translation-flavoured, no document structure |
| `HuggingFaceFW/fineweb-2` (2025-10) | the paper's own multilingual source, but older pipeline, 3,740 configs, and no English — superseded by HQ-Plus for our purpose |
| `openbmb/Ultra-FineWeb` (2026-08) | en + zh only; cannot span six scripts |
| `Helsinki-NLP/fineweb-edu-translated` (2026-04) | 37 European languages only — no cmn/jpn/arb. **Keep as a possible control** for eng/deu/rus: identical documents translated, which isolates script effects from content effects |
| `The Stack v2` | gated; code is out of scope for now |

**Fallbacks, in order:** (1) if HQ-Plus is malformed or truncated, use `HuggingFaceFW/fineweb-2`
configs for the same five languages; (2) if the epfml pair proves inconsistent, use
`fineweb-edu` + `fineweb-2` — the exact pair the Cartesia paper used, at the cost of recency.

**Sizing rule:** 50 MB/language for Q1 (mechanism; 300 MB total), 200 MB/language for Q2 if
it runs. Pack once to `.bin` and ship as a Kaggle dataset — never re-download per arm.

**Open risk:** HQ-Plus is 17 days old and unvalidated in public. Verify by reading one
shard's text and checking language purity before committing (see the dev-loop ticket).
