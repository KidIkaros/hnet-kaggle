"""English corpus packer: fineweb-edu, length-sorted, document boundaries preserved.

Ticket 03. The multilingual corpus in src/corpus.py is packed in shard order and the
training driver reads contiguous byte windows, which slices documents. The comparable rung
needs a different property: documents sorted by byte length and packed so that a batch can be
formed out of whole documents, with a held-out split carved BEFORE packing so that no
document can appear in both splits.

The DFA state stream that the multilingual corpus carries is not written here. It exists for
the byte-phase alignment analysis, and paying one extra byte per byte for it would double a
3 GB upload to answer a question the rung does not ask.

Usage:
    python src/eng_corpus.py --out DIR --target-bytes 2500000000
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq
from corpus import Ranged

SOURCE = {
    "repo": "HuggingFaceFW/fineweb-edu",
    "path": "sample/10BT/000_00000.parquet",
    "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/"
           "sample/10BT/000_00000.parquet",
    "licence": "ODC-By-1.0",
    "note": "FineWeb-Edu sample/10BT, shard 000_00000. Educational-quality filter over "
            "Common Crawl. The shard is 2.15 GB compressed; we take the leading documents "
            "up to the byte target.",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def open_source():
    req = urllib.request.Request(SOURCE["url"], method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    return pq.ParquetFile(Ranged(SOURCE["url"], size)), size


def collect(target_bytes, max_docs=None):
    """Read documents until the byte target, keeping the text and its byte length."""
    pf, size = open_source()
    log(f"shard {SOURCE['path']} ({size/1e9:.2f} GB), {pf.metadata.num_row_groups} row groups")
    docs, nbytes = [], 0
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["text", "language_score"])
        texts = tbl.column("text").to_pylist()
        scores = tbl.column("language_score").to_pylist()
        for t, ls in zip(texts, scores):
            if not t:
                continue
            b = t.encode("utf-8", "replace")
            if not b:
                continue
            docs.append((b, ls))
            nbytes += len(b)
        log(f"  row group {rg}: {len(texts)} docs, {nbytes/1e6:.1f} MB total")
        if nbytes >= target_bytes or (max_docs and len(docs) >= max_docs):
            break
    return docs, nbytes, pf.metadata.num_row_groups


def write_split(docs, out_dir, name):
    """Write one split: bytes, document offsets, and document lengths.

    Documents are sorted by byte length, so a batch of consecutive documents has near-uniform
    length and padding stays small. Offsets make the boundaries recoverable, which is what
    lets the driver form a batch out of whole documents.
    """
    docs_sorted = sorted(docs, key=lambda d: len(d[0]))
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    # Written straight to disk, document by document. Building the whole split in memory
    # first would double the corpus in RAM, and 2.5 GB of documents does not fit twice.
    offsets, lengths = [], []
    pos = 0
    with open(os.path.join(d, f"{name}.bin"), "wb") as f:
        for b, _ in docs_sorted:
            f.write(b)
            offsets.append(pos)
            lengths.append(len(b))
            pos += len(b)
    blob_len = pos
    with open(os.path.join(d, f"{name}_docs.u32"), "wb") as f:
        f.write(b"".join(o.to_bytes(4, "little") for o in offsets))
    with open(os.path.join(d, f"{name}_lens.u32"), "wb") as f:
        f.write(b"".join(v.to_bytes(4, "little") for v in lengths))
    return blob_len, offsets, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-bytes", type=int, default=2_500_000_000)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    docs, nbytes, rgs = collect(args.target_bytes)
    log(f"collected {len(docs)} documents, {nbytes/1e6:.1f} MB")

    # Duplicates are removed before the split, not after. The shard contains repeated
    # documents, and a split by index alone leaves identical text on both sides: the mount
    # check found 18 such documents in a 2.3 GB corpus. Hashing the documents first makes
    # the two splits disjoint in content, not merely in index.
    import hashlib
    seen = set()
    unique = []
    n_dupes = 0
    for d in docs:
        h = hashlib.sha1(d[0]).digest()
        if h in seen:
            n_dupes += 1
            continue
        seen.add(h)
        unique.append(d)
    if n_dupes:
        log(f"dropped {n_dupes} duplicate documents ({len(unique)} remain)")
    docs = unique

    # The held-out split is carved BEFORE packing, by document, with a fixed seed. Splitting
    # the byte stream instead would put a document on both sides of the boundary.
    import random
    idx = list(range(len(docs)))
    random.Random(20261031).shuffle(idx)
    n_val = max(1, int(len(docs) * args.val_fraction))
    val_idx = set(idx[:n_val])
    val_docs = [docs[i] for i in sorted(val_idx)]
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_idx]
    log(f"split: {len(train_docs)} train docs, {len(val_docs)} val docs")

    scores = sorted(s for _, s in docs)
    n_train_docs, n_val_docs = len(train_docs), len(val_docs)
    log(f"deduplicated: {n_dupes} dropped")
    del docs
    tr_bytes, tr_off, tr_len = write_split(train_docs, args.out, "train")
    del train_docs
    va_bytes, va_off, va_len = write_split(val_docs, args.out, "val")
    del val_docs

    meta = {
        "source_repo": SOURCE["repo"],
        "source_shard": SOURCE["path"],
        "source_shard_bytes": None,
        "licence": SOURCE["licence"],
        "provenance": SOURCE["note"],
        "split_seed": 20261031,
        "duplicate_documents_dropped": n_dupes,
        "dedup_method": "sha1 of the document bytes, before the split",
        "split_method": "held out by document before packing; the two splits share no document",
        "ordering": "documents sorted by byte length within each split",
        "train_docs": n_train_docs,
        "val_docs": n_val_docs,
        "train_bytes": tr_bytes,
        "val_bytes": va_bytes,
        "train_doc_bytes_min": min(tr_len) if tr_len else 0,
        "train_doc_bytes_median": tr_len[len(tr_len) // 2] if tr_len else 0,
        "train_doc_bytes_max": max(tr_len) if tr_len else 0,
        "val_doc_bytes_max": max(va_len) if va_len else 0,
        "language_score_median": scores[len(scores) // 2] if scores else None,
        "dfa_states": None,
        "format": "train.bin + train_docs.u32 (LE u32 offsets) + train_lens.u32 (LE u32 lengths)",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"train {tr_bytes/1e6:.1f} MB over {len(tr_off)} docs "
        f"(median {meta['train_doc_bytes_median']} B, max {meta['train_doc_bytes_max']} B)")
    log(f"val   {va_bytes/1e6:.1f} MB over {len(va_off)} docs")
    log("done")


if __name__ == "__main__":
    main()
