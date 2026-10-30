"""Spaced variants: insert an ASCII space every K characters.

The one remaining question the cross-linguistic comparisons cannot answer: is the
Chinese/Japanese/Thai failure caused by the *absence of word delimiters* (and the byte
phase that a 1-byte landmark would let the model recover), or by something specific to
those scripts?

Adding a space every K characters makes the text look, in byte terms, like a
delimited script -- 1-byte landmarks recur every K characters, so character phase is
recoverable from the byte stream alone. If the spaced variant aligns, then the failure
was the missing delimiter (or the missing landmark); if it does not, the failure is
specific to those scripts and no delimiter story can explain it.

K is swept (3 and 6) because the landmark period is the thing being tested.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from corpus import encode

SRC = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
DST = "/home/ikaaros/.hermes/cache/scratch/ds_hnet/corpus"


def space_every(text, k):
    """Insert a space after every k characters."""
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if (i + 1) % k == 0 and not ch.isspace():
            out.append(" ")
    return "".join(out)


def build(lang, k):
    src = os.path.join(SRC, lang)
    out = os.path.join(DST, f"{lang}_sp{k}")
    os.makedirs(out, exist_ok=True)
    meta = {}
    for split in ("train", "val"):
        raw = open(os.path.join(src, f"{split}.bin"), "rb").read()
        offs = list(np.frombuffer(
            open(os.path.join(src, f"{split}_docs.u32"), "rb").read(), np.uint32))
        bounds = offs + [len(raw)]
        pieces, new_offsets, n = [], [], 0
        for i in range(len(offs)):
            doc = raw[bounds[i]:bounds[i + 1]].decode("utf-8")
            b = space_every(doc, k).encode("utf-8")
            new_offsets.append(n)
            pieces.append(b)
            n += len(b)
        wb = b"".join(pieces)
        open(os.path.join(out, f"{split}.bin"), "wb").write(wb)
        # the DFA state stream: examples() reads it to annotate characters
        _, st = encode(wb.decode("utf-8"))
        open(os.path.join(out, f"{split}_state.u8"), "wb").write(st)
        open(os.path.join(out, f"{split}_docs.u32"), "wb").write(
            b"".join(int(o).to_bytes(4, "little") for o in new_offsets))
        plain = wb.decode("utf-8")
        meta[split] = {"bytes": len(wb), "docs": len(new_offsets),
                       "bytes_per_char": len(wb) / max(len(plain.replace(" ", "")), 1),
                       "space_rate": plain.count(" ") / max(len(plain), 1)}
    m = {"language": lang, "variant": f"spaced_every_{k}",
         "derived_from": f"corpus/{lang}",
         "transform": f"inserted an ASCII space after every {k} characters",
         "detail": meta}
    json.dump(m, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"{lang}_sp{k}: train {meta['train']['bytes']}B "
          f"({meta['train']['bytes_per_char']:.2f} B/char, "
          f"space rate {meta['train']['space_rate']:.3f})")


if __name__ == "__main__":
    langs = sys.argv[1:] or ["cmn"]
    for lang in langs:
        for k in (3, 6):
            build(lang, k)
