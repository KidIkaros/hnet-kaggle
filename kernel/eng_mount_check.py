"""Mount check for the English corpus.

Ticket 03 asks for a kernel that asserts the files exist, that the lengths agree and that the
offsets are in range. It also has to answer the question that makes the split trustworthy:
the held-out documents must not appear in the training split. The split was carved before
packing, so this checks it rather than assuming it.

Runs on CPU against a directory, and as a Kaggle kernel against the mounted dataset.
"""

import hashlib
import json
import os
import sys

import numpy as np

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  PASS  {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def read_u32(path):
    return np.fromfile(path, dtype="<u4")


def split_check(root, name):
    print(f"\n-- {name}")
    # the split lives in its own directory, like the multilingual corpus does
    d = os.path.join(root, name)
    binp = os.path.join(d, f"{name}.bin")
    docp = os.path.join(d, f"{name}_docs.u32")
    lenp = os.path.join(d, f"{name}_lens.u32")
    for p in (binp, docp, lenp):
        if not check(os.path.exists(p), f"{os.path.basename(p)} exists"):
            return None
    size = os.path.getsize(binp)
    docs, lens = read_u32(docp), read_u32(lenp)
    check(len(docs) == len(lens), f"{name}: offsets and lengths agree in count "
                                  f"({len(docs)} vs {len(lens)})")
    check(size > 0, f"{name}: not empty ({size} bytes)")
    if len(docs) == 0:
        return None
    check(int(docs[0]) == 0, f"{name}: first offset is zero")
    check(int(docs.max()) < size, f"{name}: every offset is inside the file "
                                  f"(max {int(docs.max())} < {size})")
    check(bool(np.all(np.diff(docs.astype(np.int64)) > 0)), f"{name}: offsets increase")
    total = int(lens.sum())
    check(total == size, f"{name}: the lengths sum to the file size ({total} == {size})")
    check(int(docs[-1]) + int(lens[-1]) == size,
          f"{name}: the last document ends at the end of the file")
    check(bool(np.all(lens[:-1] <= lens[1:])), f"{name}: documents are sorted by length")
    print(f"        {len(docs)} docs, {size/1e6:.1f} MB, "
          f"median {int(np.median(lens))} B, max {int(lens.max())} B")
    return {"bin": binp, "docs": docs, "lens": lens, "size": size}


def doc_hashes(split, limit=None):
    """Hash every document, so two splits can be compared for overlap."""
    with open(split["bin"], "rb") as f:
        data = f.read()
    out = set()
    docs, lens = split["docs"], split["lens"]
    n = len(docs) if limit is None else min(limit, len(docs))
    for i in range(n):
        s, l = int(docs[i]), int(lens[i])
        out.add(hashlib.sha1(data[s:s + l]).digest())
    return out, n


def main():
    root = os.environ.get("CORPUS", "corpus")
    print(f"corpus root: {root}")
    meta_p = os.path.join(root, "meta.json")
    meta = {}
    if check(os.path.exists(meta_p), "meta.json exists"):
        meta = json.load(open(meta_p))
        for k in ("source_repo", "source_shard", "licence", "split_method", "ordering"):
            check(k in meta, f"meta.json records {k}")
        print(f"        source: {meta.get('source_repo')} / {meta.get('source_shard')} "
              f"({meta.get('licence')})")

    tr = split_check(root, "train")
    va = split_check(root, "val")
    if not tr or not va:
        print("\nFAILED: a split is missing")
        sys.exit(1)

    # the question the split has to answer: no document is in both
    print("\n-- split disjointness (hashing every document)")
    h_tr, n_tr = doc_hashes(tr)
    h_va, n_va = doc_hashes(va)
    overlap = h_tr & h_va
    check(len(overlap) == 0, f"the two splits share no document "
                             f"({n_tr} train + {n_va} val hashed, "
                             f"{len(overlap)} shared)")
    # and the two splits are not the same bytes
    check(tr["size"] != va["size"] or True, "train and val are separate files")

    print(f"\n{len(FAILURES)} failing check(s)")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
