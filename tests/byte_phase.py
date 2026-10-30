"""Byte-phase analysis: what does the chunker learn, if not characters?

Two measurements, no training, from checkpoints that already exist.

(a) Offset-phase concentration. For each language and encoding, where do boundaries fall
    relative to byte offsets modulo 2, 3, 4? In UTF-8, Russian and Arabic characters are two
    bytes, so their character starts coincide with even offsets -- "aligns to characters" and
    "aligns to even offsets" are indistinguishable there, and only separate in Chinese, whose
    three-byte characters start at multiples of three.

(b) Zero-training transfer. Feed the UTF-8 model the UTF-16LE bytes of the same text, an
    encoding it has never seen. If the chunker is a byte-stream segmenter it should still place
    boundaries at some consistent rate and spacing; if it learned characters it cannot.

Windows start on character boundaries so the phase is well defined.
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import HNetConfig, HNetLM

CORPUS = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
LANGS = ["eng", "rus", "cmn", "jpn", "arb"]
KS = (2, 3, 4)
WINDOWS = 24
SEQ = 256


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = HNetLM(HNetConfig(**blob["config"]))
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def char_start_positions(raw):
    """Byte offsets that begin a UTF-8 character."""
    return [i for i, x in enumerate(raw) if (x & 0xC0) != 0x80]


def windows_utf8(lang, n=WINDOWS):
    data = open(os.path.join(CORPUS, lang, "val.bin"), "rb").read()
    starts = char_start_positions(data)
    step = max(1, len(starts) // (n + 1))
    out = []
    for k in range(n):
        s = starts[k * step]
        piece = data[s:s + SEQ]
        if len(piece) < SEQ // 2:
            continue
        out.append(piece)
    return out


def windows_utf16(lang, n=WINDOWS):
    """Same text, UTF-16LE bytes; character starts at even offsets."""
    data = open(os.path.join(CORPUS, lang, "val.bin"), "rb").read()
    starts = char_start_positions(data)
    step = max(1, len(starts) // (n + 1))
    out = []
    for k in range(n):
        s = starts[k * step]
        text = data[s:s + SEQ].decode("utf-8", "ignore")
        b16 = text.encode("utf-16-le")
        if len(b16) >= SEQ // 2:
            out.append(b16)
    return out


def phase_stats(boundaries, npos, ks=KS):
    """Concentration of boundaries per offset residue class, per modulus k."""
    out = {}
    for k in ks:
        tot_b = Counter(p % k for p in boundaries)
        tot_p = Counter(p % k for p in range(npos))
        nb = sum(tot_b.values())
        np_ = sum(tot_p.values())
        if nb == 0 or np_ == 0:
            continue
        rate = nb / np_
        per = {}
        for r in range(k):
            share_b = tot_b.get(r, 0) / nb
            share_p = tot_p.get(r, 0) / np_
            per[r] = (share_b / share_p) if share_p else None
        vals = [v for v in per.values() if v is not None]
        best = max(range(k), key=lambda r: per[r] if per[r] is not None else -1)
        out[k] = {"per_residue": {str(r): (round(v, 3) if v is not None else None)
                                   for r, v in per.items()},
                  "max_concentration": round(per[best], 3) if per[best] is not None else None,
                  "best_residue": best}
    return out


def run(model, pieces, utf16):
    """Boundary positions for each window, plus chunk spacing."""
    all_b, npos_total, gaps, nb_total = [], 0, [], 0
    for piece in pieces:
        idx = torch.from_numpy(np.frombuffer(piece, np.uint8).astype(np.int64))[None]
        with torch.no_grad():
            _, aux = model(idx)
        bm = aux["bmask"][0].numpy()
        pos = np.flatnonzero(bm)
        all_b.extend(pos.tolist())
        npos_total += len(bm)
        nb_total += len(pos)
        if len(pos) > 1:
            gaps.extend(np.diff(pos).tolist())
    return all_b, npos_total, nb_total, gaps


def summarise(name, model, pieces, utf16, label):
    b, npos, nb, gaps = run(model, pieces, utf16)
    st_ = phase_stats(b, npos)
    rec = {"encoding": label, "boundaries_per_position": round(nb / npos, 4),
           "mean_chunk_bytes": round(float(np.mean(gaps)), 3) if gaps else None,
           "sd_chunk_bytes": round(float(np.std(gaps)), 3) if gaps else None,
           "phase": st_}
    print(f"  {label:>10} {name:>5}: rate {rec['boundaries_per_position']:.4f}  "
          f"chunk {rec['mean_chunk_bytes']} +- {rec['sd_chunk_bytes']} B  "
          + "  ".join(f"k={k}: best r={v['best_residue']} x{v['max_concentration']}"
                      for k, v in st_.items()))
    return rec


def main():
    u8_ckpts = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/replres/ckpt-s*-n4.pt")
                      + glob.glob("/home/ikaaros/.hermes/cache/scratch/replres2/ckpt-s*-n4.pt"),
                      key=lambda p: int(re.search(r"ckpt-s(\d+)", p).group(1)))
    u16_ckpts = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/invres/ckpt-u16-s*.pt"))
    print(f"{len(u8_ckpts)} UTF-8 checkpoints, {len(u16_ckpts)} UTF-16 checkpoints")
    out = {}

    print("\n(a) native encoding, offset-phase concentration")
    for lang in LANGS:
        out[f"u8_{lang}"] = []
        for cp in u8_ckpts[:3]:
            out[f"u8_{lang}"].append(summarise(lang, load(cp), windows_utf8(lang),
                                               False, "utf8"))
        out[f"u16_{lang}"] = []
        for cp in u16_ckpts:
            out[f"u16_{lang}"].append(summarise(lang, load(cp), windows_utf16(lang),
                                                True, "utf16"))

    print("\n(b) zero-training transfer: the UTF-8 model fed UTF-16LE bytes")
    out["transfer"] = []
    for lang in LANGS:
        for cp in u8_ckpts[:3]:
            out["transfer"].append(summarise(lang, load(cp), windows_utf16(lang),
                                             True, "xfer"))

    json.dump(out, open("/home/ikaaros/.hermes/cache/scratch/phase.json", "w"), indent=2)


if __name__ == "__main__":
    main()
