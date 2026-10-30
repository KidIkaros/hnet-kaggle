"""How much can the 8-window eval be trusted?

The Q1 runs scored alignment on 8 windows of 256 bytes per language per eval. This
recomputes the same statistic on *every* non-overlapping window of the 500 KB val
stream, then bootstraps the 8-window estimator to get its sampling error.

Usage: python tests/eval_noise.py <ckpt> <corpus_dir> [batch]
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import HNetConfig, HNetLM

LANGS = ["eng", "deu", "rus", "cmn", "jpn", "arb"]
SEQ = 256


def load(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = HNetConfig(**blob["config"])
    model = HNetLM(cfg)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, cfg


def windows(data, seq=SEQ):
    n = (len(data) - 1) // seq
    return np.stack([np.frombuffer(data[i * seq:(i + 1) * seq + 1], np.uint8)
                     for i in range(n)]).astype(np.int64)


def lift_per_window(model, batch):
    """Per-window alignment lift, so the estimator's spread can be measured."""
    with torch.no_grad():
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
    lead = (batch[:, :-1].numpy() & 0xC0) != 0x80
    bm = aux["bmask"].numpy()
    out = []
    for b in range(bm.shape[0]):
        if bm[b].any():
            out.append(lead[b][bm[b]].mean() - lead[b].mean())
        else:
            out.append(0.0)
    return np.array(out), ce.item() / np.log(2)


def main():
    ckpt, corpus = sys.argv[1], sys.argv[2]
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    model, cfg = load(ckpt)
    print(f"ckpt {ckpt}  params {sum(p.numel() for p in model.parameters())/1e6:.3f}M")
    res = {}
    for lang in LANGS:
        data = np.memmap(f"{corpus}/{lang}/val.bin", np.uint8, mode="r")
        W = windows(data)
        lifts, bpbs = [], []
        for i in range(0, len(W), bs):
            l, b = lift_per_window(model, torch.from_numpy(W[i:i + bs]))
            lifts.append(l)
            bpbs.append(b)
        lifts = np.concatenate(lifts)
        true_lift = float(lifts.mean())

        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(lifts, 8, replace=False).mean()
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[lang] = {"windows": int(len(lifts)), "full_lift": true_lift,
                     "bpb": float(np.mean(bpbs)),
                     "eight_window_sd": float(boot.std()),
                     "eight_window_ci95": [float(lo), float(hi)],
                     "prob_negative_under_8win": float((boot < 0).mean())}
        print(f"{lang}: full lift {true_lift:+.3f} over {len(lifts)} windows | "
              f"8-window 95% CI [{lo:+.3f}, {hi:+.3f}] sd {boot.std():.3f} | "
              f"P(negative) {res[lang]['prob_negative_under_8win']:.3f} | bpb {np.mean(bpbs):.3f}")
    json.dump(res, open(ckpt.replace(".pt", "-evalnoise.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
