# 03 — English corpus at 2-3 GB, byte-length sorted

**What to build:** **What to build:** An English training corpus of two to three gigabytes, documents sorted by byte
length and packed, that a training kernel can mount and stream from.

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] The corpus is fetched from a fineweb-edu mirror or fineweb-2 English, and its provenance and licence are recorded.
- [ ] Documents are sorted by byte length and packed with document boundaries preserved, so a batch can be formed without slicing a document across a boundary.
- [ ] A mount-check kernel asserts the files exist, that lengths agree and that offsets are in range.
- [ ] The corpus is published as a Kaggle dataset, so runs mount rather than download.
- [ ] A held-out split is carved out before packing and is never trained on.

## Result: DONE

Corpus: `mauricew/hnet-english-bytes` (private), 2.35 GB train / 47.9 MB val.

- Source: `HuggingFaceFW/fineweb-edu` `sample/10BT/000_00000.parquet`, ODC-By-1.0. The shard is
  2.15 GB compressed; the leading documents up to the 2.4 GB target were taken.
- 492,363 train documents and 10,048 val documents, sorted by byte length within each split
  (median 2,936 B), with offsets and lengths in LE u32 files so a batch can be built from whole
  documents.
- The held-out split was carved by document before packing with seed 20261031, and 589
  duplicate documents were dropped by sha1 **before** the split. That dedup was not cosmetic:
  the first pack, split by index alone, left 18 identical documents on both sides, which the
  mount check caught.
- `kernel/eng_mount_check.py` asserts the files exist, that offsets are in range and increasing,
  that the lengths sum to the file size, that documents are sorted by length, and that the two
  splits share no document in content. All 20 checks pass on the published corpus.

Two publishing notes worth keeping: the Kaggle CLI's `--dir-mode` defaults to **skip** for
directories, so a corpus laid out as `train/` and `val/` uploads as metadata only — the layout
is flat instead; and `datasets version` takes `-p` for the folder (`-d` means delete old
versions).
