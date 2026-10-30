# Dataset scouting for the modality expansion

Type: task
Status: resolved
Blocked by: none

## What was checked

Every candidate was queried live against the Hugging Face API on 2026-10-30: last-modified date,
gating, licence, declared size, actual file listing, and -- where it mattered -- a byte-range
probe to confirm the file format is what the card claims. Recency is treated as a first-class
criterion: nothing older than 2025 is proposed as a primary, and the stale classics are named
explicitly so they are not silently reached for.

## Audio (ticket 25) -- needs raw samples we can re-encode ourselves

| dataset | last modified | licence | format | verdict |
|---|---|---|---|---|
| `espnet/yodas3` | **2026-10-07** | cc-by-3.0 | tar audio + parquet metadata | **primary** |
| `disco-eth/WorldSpeech` | 2026-09-25 | cc-by-**nc**-4.0 | parquet | easiest engineering, NC licence |
| `google/fleurs` | 2026-05-15 | cc-by-4.0 | tar.gz | fallback |
| ~~LibriSpeech~~ (2015) | stale | -- | -- | not proposed |

`espnet/yodas3` is the strongest pick: ten days old at time of writing, 416k downloads, permissive,
and it is organised **per language**, which matters for cost -- the Abkhazian shard
(`data/ab/audio/0000.tar`) is a small tar for a low-resource language, so a full tar download is
cheap where an English shard would not be. The tar was probed directly: the first 4 KB contains a
JSON sidecar (`{"id":..., "length":437.0, "audio_bitrate":145.584, "channels":2, ...}`) followed
by the audio entries, so the metadata needed to select clips is available without decoding.

The experiment needs only ~2 h of speech, decoded to int16 and re-emitted as four byte streams
(PCM-LE, PCM-BE, mu-law, float32) by our own packer. **The dataset is only a source of samples;
the encodings are ours**, so dataset licence governs the samples, not the artifact we publish.

## Vision I (ticket 23) -- needs images, not masks

| dataset | last modified | licence | format | verdict |
|---|---|---|---|---|
| `athkot/coco-20-synthetic-50k` | **2026-10-30** | cc-by-4.0 | parquet, 11 shards | **primary** |
| `Travor278/whoego-synth-animals` | 2026-10-30 | cc-by-4.0 | tar, n<1K | small, has a segmentation track |
| ~~CIFAR-10~~ (2009), ~~ImageNet~~ | stale | -- | -- | not proposed |

`athkot/coco-20-synthetic-50k` is a day old, permissive, 10-100k images, COCO-style with bboxes,
and its parquet footer is **range-readable** -- verified by fetching the last 8 bytes and finding
`PAR1`, so it fits the same `Ranged` reader the text corpus uses and needs no full download.
Caveat: it carries **bboxes, not per-pixel masks**.

## Vision II (ticket 24) -- needs masks

**Primary: generate the data ourselves.** The 2D chunker needs ground truth that is unambiguous,
and synthetic shapes on plain backgrounds give exact per-pixel masks with no licence, no download
and no annotation noise -- the same reasoning that put a synthetic oracle behind the alignment
probe test. `Travor278/whoego-synth-*` (2026-10-30, cc-by-4.0, n<1K) has a segmentation track and
is usable for a qualitative real-ish check, but at under 1K images it cannot carry a statistic.
The medical segmentation sets that dominate the recent image-segmentation list (`BodyMaps/CancerVerse`,
`MedOtter/LUNA16`) have real masks but the wrong domain -- chunking behaviour on CT slices would
not transfer to natural images.

## Code (ticket 26)

| dataset | last modified | licence | format | verdict |
|---|---|---|---|---|
| `code-search-net/code_search_net` | 2026-02-23 | other (per-repo) | parquet | usable |
| `openbmb/UltraData-Code` | 2026-10-16 | apache-2.0 | **declared parquet, absent** | **not usable** |
| ~~bigcode/the-stack-smol~~ (2023), ~~The Stack v2~~ | stale / gated | -- | -- | not proposed |

`openbmb/UltraData-Code` looked ideal on its card -- apache-2.0, 100M-1B, updated 2026-10-16 -- but
its file listing contains **only** `README.md`, `LICENSE`, `README_ZH.md` and an image. The card
declares `data/UltraData-Code-L2/cpp/*.parquet` and the repo has no `data/` directory at all. The
dataset card is a placeholder, not a dataset. Recorded because the card would otherwise have been
believed.

## What this changes about the plan

1. All four modality arms can start without waiting for GPU quota, and three of them can use
   parquet range-reading, which is already built.
2. The publishable artifact should be **prepared byte-serialized datasets** -- the same pattern as
   `hnet-multiscript-bytes`: one Kaggle dataset per modality holding the same content in several
   encodings, so every run mounts rather than downloads and the encoding comparison is exact by
   construction.
3. Audio is the cheapest arm to start (one small tar, ~2 h of samples, no new architecture) and
   the LE/BE pair is the sharpest test on the roadmap.
