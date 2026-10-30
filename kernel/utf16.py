"""UTF-16LE variants of the packed corpus, for the encoding-invariance test.

Same text, different byte representation: every character becomes 2 bytes (BMP), so
character starts sit at even byte offsets. Ticket 11's amendments apply: match on
characters rather than bytes, use offset parity instead of the UTF-8 DFA, and compare
boundary *positions in character space* between the two encodings.

Document boundaries are preserved (the same documents, doubled in length).
"""
import json
import os
import sys

SRC = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
DST = "/home/ikaaros/.hermes/cache/scratch/ds_hnet/corpus"


def build(lang):
    src = os.path.join(SRC, lang)
    out = os.path.join(DST, f"{lang}_u16")
    os.makedirs(out, exist_ok=True)
    meta = {}
    for split in ("train", "val"):
        raw = open(os.path.join(src, f"{split}.bin"), "rb").read()
        offs = [int(o) for o in
                __import__("numpy").frombuffer(
                    open(os.path.join(src, f"{split}_docs.u32"), "rb").read(),
                    __import__("numpy").uint32)]
        bounds = offs + [len(raw)]
        pieces, new_offsets, n = [], [], 0
        for i in range(len(offs)):
            doc = raw[bounds[i]:bounds[i + 1]]
            b = doc.decode("utf-8").encode("utf-16-le")
            new_offsets.append(n)
            pieces.append(b)
            n += len(b)
        wb = b"".join(pieces)
        open(os.path.join(out, f"{split}.bin"), "wb").write(wb)
        open(os.path.join(out, f"{split}_docs.u32"), "wb").write(
            b"".join(o.to_bytes(4, "little") for o in new_offsets))
        meta[split] = {"bytes": len(wb), "chars": len(wb) // 2, "docs": len(new_offsets)}
    m = {"language": lang, "variant": "utf16le",
         "derived_from": f"corpus/{lang}",
         "transform": "same text, encoded UTF-16LE; character starts at even byte offsets",
         "detail": meta}
    json.dump(m, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"{lang}_u16: train {meta['train']['bytes']}B "
          f"({meta['train']['chars']} chars), val {meta['val']['bytes']}B")


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["eng", "deu", "rus", "cmn", "jpn", "arb"]:
        build(lang)
