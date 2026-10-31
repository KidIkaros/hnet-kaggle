"""LE vs BE at identical byte offsets: the test that needs no semantic reference.

PCM-LE and PCM-BE are the same samples with every byte pair swapped. A boundary before sample k
sits at byte 2k in *both* streams, so:

  - a chunker that follows the signal puts boundaries at the same byte offsets in both;
  - a chunker that follows byte values cannot, because the byte at 2k is the low byte in LE and
    the high byte in BE.

Chance is computed from each model's own boundary rate, so the two arms having different rates
does not inflate the comparison.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import HNetConfig, HNetLM

DATA = "/home/ikaaros/.hermes/cache/scratch/aud/packed"
RES = "/home/ikaaros/.hermes/cache/scratch/audres"
META = json.load(open(os.path.join(DATA, "meta.json")))
SR, N = META["sample_rate"], META["n_samples"]
SEQ = 512
N_WIN = 40


def load(arm):
    blob = torch.load(os.path.join(RES, f"ckpt-{arm}-s0.pt"), map_location="cpu",
                      weights_only=False)
    m = HNetLM(HNetConfig(**blob["config"]))
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def bmask(model, raw, base, n=SEQ):
    arr = np.frombuffer(raw[base:base + n], np.uint8).astype(np.int64)
    with torch.no_grad():
        _, aux = model(torch.from_numpy(arr)[None, :])
    return aux["bmask"][0].numpy()


def main():
    m = {a: load(a) for a in ("pcm16le", "pcm16be")}
    raw = {a: np.memmap(os.path.join(DATA, f"{a}.bin"), np.uint8, mode="r")
           for a in ("pcm16le", "pcm16be")}
    val_start = int(N * 0.85) * 2
    end = N * 2

    jac, f1, rates, shift1 = [], [], [], []
    for i in range(N_WIN):
        base = val_start + i * SEQ
        if base + SEQ + 1 > end:
            break
        le = bmask(m["pcm16le"], raw["pcm16le"], base)
        be = bmask(m["pcm16be"], raw["pcm16be"], base)
        inter, union = float(np.sum(le & be)), float(np.sum(le | be))
        jac.append(inter / union if union else 0.0)
        p, r = inter / max(1.0, le.sum()), inter / max(1.0, be.sum())
        f1.append(2 * p * r / max(1e-9, p + r))
        rates.append((float(le.mean()), float(be.mean())))
        # byte-phase probe: does shifting one arm by a byte recover agreement?
        be1 = np.roll(be, 1)
        inter1, union1 = float(np.sum(le & be1)), float(np.sum(le | be1))
        shift1.append(inter1 / union1 if union1 else 0.0)

    rl, rb = np.mean([x[0] for x in rates]), np.mean([x[1] for x in rates])
    chance = rl * rb / max(1e-9, rl + rb - rl * rb)
    print(f"windows: {len(jac)}  ({SEQ} bytes = {SEQ/2/SR*1000:.0f} ms each)")
    print(f"boundary rate: LE {rl:.3f}  BE {rb:.3f} per byte")
    print(f"Jaccard at identical offsets : {np.mean(jac):.3f}   chance {chance:.3f}")
    print(f"F1 at identical offsets      : {np.mean(f1):.3f}")
    print(f"Jaccard with BE shifted 1 B  : {np.mean(shift1):.3f}")
    print()
    if np.mean(jac) > 2.5 * chance:
        print("reading: boundaries survive the byte-order change -> signal-following")
    elif np.mean(jac) <= 1.5 * chance:
        print("reading: boundaries do not survive the byte-order change -> byte-value-following")
    else:
        print("reading: above chance but far from agreement -> partly signal, partly layout")


if __name__ == "__main__":
    main()
