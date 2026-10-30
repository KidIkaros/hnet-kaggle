"""Width manipulation: hold script and text fixed, change bytes per character.

The replication left one correlation untested causally: alignment tracks bytes per
character (1.78-1.80 aligned, 2.67 not). Cross-language comparisons confound width
with script, encoding, corpus and delimiters.

This removes that confound. Appending U+0301 COMBINING ACUTE ACCENT to every non-space
character of the *same text* turns a 1.0 B/char stream into ~2.7 and a 1.8 B/char stream
into ~3.8, without changing the script, the words, the whitespace, the documents or the
model. If width drives alignment, the expanded variants must lose it; if they keep it,
width is not the driver.

Document boundaries are preserved by transforming each document separately and tracking
the new offsets. The UTF-8 DFA state stream is recomputed for the new bytes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from corpus import encode

MARK = "́"
SRC = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
DST = "/home/ikaaros/.hermes/cache/scratch/ds_hnet/corpus"


def widen(text):
    """Append a combining acute to every non-space character."""
    out = []
    for ch in text:
        out.append(ch)
        if not ch.isspace():
            out.append(MARK)
    return "".join(out)


def docs(raw, offsets):
    """Split a byte stream into documents using the offset table."""
    bounds = list(offsets) + [len(raw)]
    return [raw[bounds[i]:bounds[i + 1]] for i in range(len(offsets))]


def build(lang, suffix="_comb"):
    src = os.path.join(SRC, lang)
    out = os.path.join(DST, f"{lang}{suffix}")
    os.makedirs(out, exist_ok=True)
    meta_out = {}

    for split in ("train", "val"):
        raw = open(os.path.join(src, f"{split}.bin"), "rb").read()
        offs = np.frombuffer(open(os.path.join(src, f"{split}_docs.u32"), "rb").read(),
                             np.uint32)
        pieces, states, new_offsets, nbytes = [], [], [], 0
        for doc in docs(raw, offs):
            text = doc.decode("utf-8")
            b, st = encode(widen(text))
            new_offsets.append(nbytes)
            pieces.append(b)
            states.append(st)
            nbytes += len(b)
        wb, ws = b"".join(pieces), b"".join(states)
        # Truncate to the ORIGINAL byte count: the same amount of text-bytes as the
        # plain variant, so the two arms differ in bytes-per-character and nothing else.
        # (Left whole, the widened corpus is 3x the bytes and would confound width
        # with corpus size.)
        wb, ws = wb[: len(raw)], ws[: len(raw)]
        # Cut on a character boundary using the DFA stream itself: state 0 marks a
        # character start, so the last state-0 position is a safe cut point. Trimming
        # trailing continuation bytes is not enough -- a truncated lead byte would
        # leave invalid UTF-8 behind.
        st = np.frombuffer(ws, np.uint8)
        starts = np.nonzero(st == 0)[0]
        cut = int(starts[-1]) if len(starts) else 0
        wb, ws = wb[:cut], ws[:cut]
        new_offsets = [o for o in new_offsets if o < cut]
        open(os.path.join(out, f"{split}.bin"), "wb").write(wb)
        open(os.path.join(out, f"{split}_state.u8"), "wb").write(ws)
        open(os.path.join(out, f"{split}_docs.u32"), "wb").write(
            b"".join(o.to_bytes(4, "little") for o in new_offsets))
        plain = raw.decode("utf-8")
        meta_out[split] = {
            "bytes": len(wb), "docs": len(new_offsets),
            "bytes_per_char_before": len(raw) / max(len(plain), 1),
            "bytes_per_char_after": len(wb) / max(len(wb.decode("utf-8").replace(MARK, "")), 1),
            "truncated_to_original_bytes": True,
        }

    meta = {"language": lang, "variant": "widened",
            "derived_from": f"corpus/{lang}",
            "transform": "appended U+0301 COMBINING ACUTE ACCENT to every non-space character",
            **{f"{k}_bytes_per_char": round(v["bytes_per_char_after"], 3)
               for k, v in meta_out.items()},
            "detail": meta_out}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"{lang}{suffix}: train {meta_out['train']['bytes_per_char_before']:.2f} -> "
          f"{meta_out['train']['bytes_per_char_after']:.2f} B/char, "
          f"val {meta_out['val']['bytes_per_char_before']:.2f} -> "
          f"{meta_out['val']['bytes_per_char_after']:.2f} B/char")


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["eng", "rus"]:
        build(lang)
