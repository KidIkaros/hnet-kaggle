"""Build spaceless variants of a packed language directory.

Same script, same characters, same byte width — only the 0x20 word separators are
deleted, so the DFA state stream has to be recomputed. This is the causal version of
the cross-language comparison: hold the language fixed and remove the whitespace prior.
"""
import json
import os
import sys

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from corpus import encode

SRC = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
DST = "/home/ikaaros/.hermes/cache/scratch/ds_hnet/corpus"


def strip_spaces(raw):
    return raw.replace(b" ", b"")


def build(lang, target_bytes=49_000_000):
    src = os.path.join(SRC, lang)
    out = os.path.join(DST, f"{lang}_nospace")
    os.makedirs(out, exist_ok=True)

    tr = open(os.path.join(src, "train.bin"), "rb").read()
    va = open(os.path.join(src, "val.bin"), "rb").read()
    tr_s, va_s = strip_spaces(tr), strip_spaces(va)

    def rebuild(raw):
        text = raw.decode("utf-8")
        b, st = encode(text)
        return b, st

    tr_b, tr_st = rebuild(tr_s)
    va_b, va_st = rebuild(va_s)

    open(os.path.join(out, "train.bin"), "wb").write(tr_b)
    open(os.path.join(out, "train_state.u8"), "wb").write(tr_st)
    open(os.path.join(out, "val.bin"), "wb").write(va_b)
    open(os.path.join(out, "val_state.u8"), "wb").write(va_st)

    # document offsets no longer apply (spaces were the separators), so emit none;
    # nothing downstream of Q1 uses them.
    for name in ("train_docs.u32", "val_docs.u32"):
        open(os.path.join(out, name), "wb").write(b"")

    meta = {"language": lang, "variant": "nospace",
            "derived_from": f"corpus/{lang}",
            "transform": "deleted all 0x20 bytes, recomputed UTF-8 DFA states",
            "train_bytes": len(tr_b), "val_bytes": len(va_b),
            "space_rate_before": tr.count(b" ") / len(tr),
            "space_rate_after": tr_b.count(b" ") / len(tr_b)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"{lang}_nospace: {len(tr_b)} train / {len(va_b)} val bytes, "
          f"space {meta['space_rate_before']:.3f} -> {meta['space_rate_after']:.3f}")


for lang in sys.argv[1:] or ["hin", "rus"]:
    build(lang)
