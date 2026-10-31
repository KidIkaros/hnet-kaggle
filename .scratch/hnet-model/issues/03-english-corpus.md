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
