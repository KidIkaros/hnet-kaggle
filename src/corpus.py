"""Corpus sources, UTF-8 DFA encoding and ranged parquet reads.

Inlined verbatim into the Kaggle kernel (only `code_file` is uploaded), and
imported by the local packer, so there is one copy of the encoding logic.
"""

import io
import json
import os
import urllib.request

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/epfml/{repo}/resolve/main/{path}"

# HQ-Plus carries the same schema in every config; FineWeb-HQ is the English arm
# of the same filtering family. One shard per language is ~460-700 MB raw and
# covers a 50-200 MB sample with room to spare.
SOURCES = {
    "deu": ("FineWeb2-HQ-Plus", "deu_Latn/000_00000.parquet"),
    "rus": ("FineWeb2-HQ-Plus", "rus_Cyrl/000_00000.parquet"),
    "cmn": ("FineWeb2-HQ-Plus", "cmn_Hani/000_00000.parquet"),
    "jpn": ("FineWeb2-HQ-Plus", "jpn_Jpan/000_00000.parquet"),
    "arb": ("FineWeb2-HQ-Plus", "arb_Arab/000_00000.parquet"),
    # 3-byte AND whitespace-delimited: separates character width from word delimitation,
    # which are confounded across the six Q1 languages.
    "hin": ("FineWeb2-HQ-Plus", "hin_Deva/000_00000.parquet"),
    "eng": ("FineWeb-HQ", "data/CC-MAIN-2024-51/000_00000.parquet"),
}

# ---- UTF-8 DFA compressed to 8 states (from mbdfa/src/build_corpus.py) ----
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
    """UTF-8 bytes -> (bytes, dfa state per byte). state 0 == ASCII or lead byte."""
    b = text.encode("utf-8", "replace")
    st = bytearray(len(b))
    s = 0
    for i, x in enumerate(b):
        st[i] = s
        ns = next_state(s, x)
        if ns is None:            # U+FFFD can break the chain; resync at ASCII
            s = 0
        else:
            s = ns
    return b, bytes(st)


class Ranged(io.RawIOBase):
    """Seekable file-like over HTTPS range requests: a 630 MB shard costs ~60 MB."""

    def __init__(self, url, size):
        self.url, self.pos, self.size = url, 0, size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1
                                            else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url,
                                     headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def open_shard(lang):
    repo, path = SOURCES[lang]
    url = BASE.format(repo=repo, path=path)
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    return pq.ParquetFile(Ranged(url, size)), url, path, repo, size


def pack_lang(lang, out, target_bytes, val_bytes, log=print):
    """Stream one shard's `text` column into train/val byte streams + DFA states."""
    pf, url, path, repo, size = open_shard(lang)
    d = os.path.join(out, lang)
    os.makedirs(d, exist_ok=True)

    whole_b, whole_s, offsets = [], [], []
    nbytes = ndocs = rowgroups = 0
    scores = []
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["text", "language", "language_score"])
        for t, lg, ls in zip(*(tbl.column(c).to_pylist()
                               for c in ("text", "language", "language_score"))):
            scores.append(ls)
            b, st = encode((t or "").strip() + "\n")
            if not b:
                continue
            offsets.append(nbytes)
            whole_b.append(b)
            whole_s.append(st)
            nbytes += len(b)
            ndocs += 1
            if nbytes >= target_bytes:
                break
        rowgroups += 1
        if nbytes >= target_bytes:
            break

    wb, ws = b"".join(whole_b), b"".join(whole_s)
    cut = max(0, len(wb) - val_bytes)
    while 0 < cut < len(wb) and (wb[cut] & 0xC0) == 0x80:
        cut += 1                               # never split a UTF-8 character
    split_doc = next((i for i, o in enumerate(offsets) if o >= cut), len(offsets))

    for tr_name, va_name, tr, va in (("train.bin", "val.bin", wb[:cut], wb[cut:]),
                                     ("train_state.u8", "val_state.u8", ws[:cut], ws[cut:])):
        with open(os.path.join(d, tr_name), "wb") as f:
            f.write(tr)
        with open(os.path.join(d, va_name), "wb") as f:
            f.write(va)
    for name, vals in (("train_docs.u32", offsets[:split_doc]),
                       ("val_docs.u32", [o - cut for o in offsets[split_doc:]])):
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"".join(v.to_bytes(4, "little") for v in vals))

    ss = sorted(scores)
    meta = {"language": lang, "source_repo": repo, "source_shard": path,
            "source_shard_bytes": size, "row_groups_read": rowgroups,
            "train_bytes": cut, "val_bytes": len(wb) - cut, "docs": ndocs,
            "language_score_min": ss[0] if ss else None,
            "language_score_median": ss[len(ss) // 2] if ss else None,
            "dfa_states": 8,
            "format": "train.bin + train_docs.u32 (LE u32) + train_state.u8"}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  packed {lang}: {cut} train / {len(wb) - cut} val bytes, {ndocs} docs, "
        f"{rowgroups} row groups, lang_score {meta['language_score_median']}")
    return meta
