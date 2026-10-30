"""Where inside a character does the boundary fall?

For every (script, encoding) pair, bucket each byte by its offset within its
character (0 = character start, 1 = second byte, 2 = third byte, ...) and compare
the boundary distribution to the byte distribution. This is unit-free and comparable
across scripts and encodings, unlike a residue-class histogram, because it is keyed
to the actual character boundaries rather than to a fixed modulus.

    concentration(offset o) = p(boundary | offset o) / p(offset o)

> 1 means boundaries avoid that offset; < 1 means they favour it.
"""
import glob
import json
import os
import re
import statistics as st
from collections import Counter

import numpy as np
import torch

import sys
sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import HNetConfig, HNetLM

CORPUS = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
LANGS = ["eng", "rus", "cmn", "jpn", "arb"]
SEQ = 256
WINDOWS = 24


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = HNetLM(HNetConfig(**blob["config"]))
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def utf8_offsets(raw):
    """offset within the character for each byte of a UTF-8 buffer.

    A lead byte starts a character (offset 0); each continuation byte advances the
    offset. Getting this wrong by one silently shifts every continuation byte into
    the previous class and inverts the reading, so it is checked below.
    """
    off = []
    cur = 0
    for x in raw:
        if (x & 0xC0) != 0x80:
            cur = 0
        else:
            cur += 1
        off.append(cur)
    return off


def windows(lang, utf16, n=WINDOWS):
    data = open(os.path.join(CORPUS, lang, "val.bin"), "rb").read()
    starts = [i for i, x in enumerate(data) if (x & 0xC0) != 0x80]
    step = max(1, len(starts) // (n + 1))
    out = []
    for k in range(n):
        s = starts[k * step]
        piece = data[s:s + SEQ]
        if len(piece) < SEQ // 2:
            continue
        if utf16:
            piece = piece.decode("utf-8", "ignore").encode("utf-16-le")
            if len(piece) < SEQ // 2:
                continue
        out.append(piece)
    return out


def measure(model, pieces, utf16):
    b_off = Counter()
    p_off = Counter()
    for piece in pieces:
        offs = ([i % 2 for i in range(len(piece))] if utf16
                else utf8_offsets(piece))
        idx = torch.from_numpy(np.frombuffer(piece, np.uint8).astype(np.int64))[None]
        with torch.no_grad():
            _, aux = model(idx)
        bm = aux["bmask"][0].numpy()
        for i in range(len(piece)):
            p_off[offs[i]] += 1
            if bm[i]:
                b_off[offs[i]] += 1
    nb, np_ = sum(b_off.values()), sum(p_off.values())
    if nb == 0:
        return None
    out = {}
    for o in sorted(p_off):
        share_b = b_off.get(o, 0) / nb
        share_p = p_off[o] / np_
        out[str(o)] = round(share_b / share_p, 3) if share_p else None
    return out


def main():
    u8 = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/replres/ckpt-s*-n4.pt")
                + glob.glob("/home/ikaaros/.hermes/cache/scratch/replres2/ckpt-s*-n4.pt"),
                key=lambda p: int(re.search(r"ckpt-s(\d+)", p).group(1)))[:4]
    u16_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/ikaaros/.hermes/cache/scratch/invres"
    u16 = sorted(glob.glob(os.path.join(u16_dir, "ckpt-u16-s*.pt")))
    print("utf16 checkpoints from", u16_dir)
    print(f"{len(u8)} UTF-8 checkpoints, {len(u16)} UTF-16 checkpoints")
    print("concentration by offset-within-character (1.0 = no preference)\n")
    res = {}
    for lang in LANGS:
        res[f"u8_{lang}"] = [measure(load(c), windows(lang, False), False) for c in u8]
        res[f"u16_{lang}"] = [measure(load(c), windows(lang, True), True) for c in u16]
        for tag, vals in (("utf8", res[f"u8_{lang}"]), ("utf16", res[f"u16_{lang}"])):
            good = [v for v in vals if v]
            if not good:
                continue
            keys = sorted(set(k for v in good for k in v))
            mean = {k: round(st.mean([v[k] for v in good if k in v]), 3) for k in keys}
            print(f"  {lang:>4} {tag:>6}: " +
                  "  ".join(f"off{k}={mean[k]:.3f}" for k in keys) +
                  f"   ({len(good)} models)")
    print("\n  off0 = character start, off1 = second byte of the character, etc.")
    json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/char_offset.json", "w"), indent=2)


if __name__ == "__main__":
    main()
