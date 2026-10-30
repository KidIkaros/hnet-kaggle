"""Pack HQ corpora to byte streams with document boundaries + UTF-8 DFA state.

Range-reads parquet row groups over HTTPS so a 630 MB shard costs ~60-80 MB of
transfer. Column projection to `text` only. Emits per language:

    <lang>/train.bin        concatenated UTF-8 bytes
    <lang>/train_docs.u32   byte offset of each document start (uint32)
    <lang>/train_state.u8   UTF-8 DFA state per byte (0 == ASCII/lead, from mbdfa)
    <lang>/val.bin / _docs.u32 / _state.u8   held-out tail
    <lang>/meta.json        source shard, doc count, byte counts, language scores

Usage:  python pack.py --out DIR --mb 50 [--prefix-bytes 0] [lang ...]
"""

import argparse
import io
import json
import os
import sys
import urllib.request

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/epfml/{repo}/resolve/main/{path}"
# HQ-Plus carries text in every config; the shard naming is uniform.
SOURCES = {
    "deu": ("FineWeb2-HQ-Plus", "deu_Latn/000_00000.parquet"),
    "rus": ("FineWeb2-HQ-Plus", "rus_Cyrl/000_00000.parquet"),
    "cmn": ("FineWeb2-HQ-Plus", "cmn_Hani/000_00000.parquet"),
    "jpn": ("FineWeb2-HQ-Plus", "jpn_Jpan/000_00000.parquet"),
    "arb": ("FineWeb2-HQ-Plus", "arb_Arab/000_00000.parquet"),
    "eng": ("FineWeb-HQ", "data/CC-MAIN-2024-51/000_00000.parquet"),
}

# ---- UTF-8 DFA compressed to 8 states (copied from mbdfa/src/build_corpus.py) ----
STATES = ["S0", (1, 0x80, 0xBF), (2, 0x80, 0xBF), (2, 0xA0, 0xBF), (2, 0x80, 0x9F),
          (3, 0x80, 0xBF), (3, 0x90, 0xBF), (3, 0x80, 0x8F)]
SID = {s: i for i, s in enumerate(STATES)}


def next_state(s, b):
    if s == 0:
        if b < 0x80:
            return 0
        if 0xC2 <= b <= 0xDF:
            return SID[(1, 0x80, 0xBF)]
        if b == 0xE0:
            return SID[(2, 0xA0, 0xBF)]
        if 0xE1 <= b <= 0xEC:
            return SID[(2, 0x80, 0xBF)]
        if b == 0xED:
            return SID[(2, 0x80, 0x9F)]
        if 0xEE <= b <= 0xEF:
            return SID[(2, 0x80, 0xBF)]
        if b == 0xF0:
            return SID[(3, 0x90, 0xBF)]
        if 0xF1 <= b <= 0xF3:
            return SID[(3, 0x80, 0xBF)]
        if b == 0xF4:
            return SID[(3, 0x80, 0x8F)]
        return None
    rem, lo, hi = STATES[s]
    if not (lo <= b <= hi):
        return None
    return SID[(rem - 1, 0x80, 0xBF)] if rem > 1 else 0


def encode(text):
    """UTF-8 bytes -> (bytes, dfa state per byte). Skips invalid sequences."""
    b = text.encode("utf-8", "replace")
    st = bytearray(len(b))
    s = 0
    for i, x in enumerate(b):
        st[i] = s
        ns = next_state(s, x)
        if ns is None:          # replacement char can break the chain; resync
            s = 0
        else:
            s = ns
    return b, bytes(st)


class Ranged(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.pos, self.size = url, 0, size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def pack_lang(lang, out, target_bytes, val_bytes):
    repo, path = SOURCES[lang]
    url = BASE.format(repo=repo, path=path)
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    pf = pq.ParquetFile(Ranged(url, size))

    d = os.path.join(out, lang)
    os.makedirs(d, exist_ok=True)
    tp = os.path.join(d, "train.bin")
    dp = os.path.join(d, "train_docs.u32")
    sp = os.path.join(d, "train_state.u8")
    ft, fd, fs = open(tp, "wb"), open(dp, "wb"), open(sp, "wb")

    nbytes = ndocs = 0
    buf_b, buf_s, buf_d = [], [], []
    scores = []
    rowgroups = 0
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["text", "language", "language_score"])
        for t, lg, ls in zip(*(tbl.column(c).to_pylist() for c in ("text", "language", "language_score"))):
            scores.append(ls)
            b, st = encode(t.strip() + "\n")
            if not b:
                continue
            buf_d.append(nbytes)
            buf_b.append(b)
            buf_s.append(st)
            nbytes += len(b)
            ndocs += 1
            if nbytes >= target_bytes:
                break
        rowgroups += 1
        if nbytes >= target_bytes:
            break

    whole_b = b"".join(buf_b)
    whole_s = b"".join(buf_s)
    cut = len(whole_b) - val_bytes
    while 0 < cut < len(whole_b) and (whole_b[cut] & 0xC0) == 0x80:
        cut += 1                       # do not split a UTF-8 character
    split_doc = next((i for i, off in enumerate(buf_d) if off >= cut), len(buf_d))

    ft.write(whole_b[:cut])
    fs.write(whole_s[:cut])
    for off in buf_d[:split_doc]:
        fd.write(off.to_bytes(4, "little"))
    ft.close(), fd.close(), fs.close()

    with open(os.path.join(d, "val.bin"), "wb") as f:
        f.write(whole_b[cut:])
    with open(os.path.join(d, "val_state.u8"), "wb") as f:
        f.write(whole_s[cut:])
    with open(os.path.join(d, "val_docs.u32"), "wb") as f:
        f.write(b"".join((off - cut).to_bytes(4, "little")
                         for off in buf_d[split_doc:]))

    sorted_scores = sorted(scores)
    meta = {
        "language": lang,
        "source_repo": repo,
        "source_shard": path,
        "source_shard_bytes": size,
        "row_groups_read": rowgroups,
        "train_bytes": cut,
        "val_bytes": len(whole_b) - cut,
        "docs": ndocs,
        "language_score_min": sorted_scores[0],
        "language_score_median": sorted_scores[len(sorted_scores) // 2],
        "dfa_states": 8,
        "format": "train.bin + train_docs.u32 (LE u32 offsets) + train_state.u8",
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{lang}: {cut} train / {len(whole_b) - cut} val bytes, {ndocs} docs, "
          f"{rowgroups} row groups, lang_score median {meta['language_score_median']:.3f}",
          flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=50)
    ap.add_argument("--val-mb", type=float, default=1)
    ap.add_argument("langs", nargs="*", default=[])
    a = ap.parse_args()
    langs = a.langs or list(SOURCES)
    os.makedirs(a.out, exist_ok=True)
    metas = [pack_lang(l, a.out, int(a.mb * 1e6), int(a.val_mb * 1e6)) for l in langs]
    with open(os.path.join(a.out, "corpus.json"), "w") as f:
        json.dump({"per_language": metas, "requested_mb": a.mb}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
