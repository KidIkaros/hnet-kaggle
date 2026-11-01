"""Evaluate a saved checkpoint on the held-out split, with the shift the model actually uses.

Needed because the first rung's evaluation was off by one: it compared logits[t] against byte
t while the model predicts byte t+1, which reported 13.2 bpb on a run whose training bpb had
reached 1.44. The weights are fine; only the measurement was wrong, so this re-scores the
existing checkpoint rather than retraining.

    python tools/eval_ckpt.py --ckpt path/to/ckpt-final.pt [--batches 100]

Runs on CPU so it costs no GPU quota.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/ikaaros/Coding/kaggle/hnet"
sys.path.insert(0, os.path.join(ROOT, "src"))
import model as M

CORPUS = os.environ.get("CORPUS", "/home/ikaaros/.hermes/cache/scratch/eng_corpus")


def log(*a):
    print(*a, flush=True)


class Docs:
    def __init__(self, split):
        d = os.path.join(CORPUS, split)
        if not os.path.exists(os.path.join(d, f"{split}.bin")):
            d = CORPUS
        self.data = np.memmap(os.path.join(d, f"{split}.bin"), dtype=np.uint8, mode="r")
        self.offsets = np.fromfile(os.path.join(d, f"{split}_docs.u32"), dtype="<u4")
        self.lengths = np.fromfile(os.path.join(d, f"{split}_lens.u32"), dtype="<u4")

    def __len__(self):
        return len(self.offsets)

    def doc(self, i):
        o, l = int(self.offsets[i]), int(self.lengths[i])
        return self.data[o:o + l]


def batch_plan(docs, batch, seq_len, rng):
    n_batches = len(docs) // batch
    starts = np.arange(n_batches) * batch
    widths = np.array([int(docs.lengths[s: s + batch].max()) for s in starts])
    return starts, widths, rng.permutation(n_batches)


def make_batch(docs, start, width, batch, seq_len):
    out = np.zeros((batch, seq_len), dtype=np.uint8)
    mask = np.zeros((batch, seq_len), dtype=np.float32)
    for r in range(batch):
        i = start + r
        if i >= len(docs):
            continue
        raw = docs.doc(i)
        n = min(len(raw), width, seq_len)
        if n < 2:
            continue
        out[r, :n] = raw[:n]
        mask[r, :n - 1] = 1.0
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = M.HNetConfig(**ck["config"])
    m = M.HNetLM(cfg)
    m.load_state_dict(ck["state"])
    m.eval()
    M.SCAN_IMPL = "reference"
    log(f"checkpoint step {ck.get('step')}, {sum(p.numel() for p in m.parameters())/1e6:.2f} M params")

    docs = Docs(args.split)
    seq_len, batch = 2048, 4
    rng = np.random.default_rng(0)
    starts, widths, order = batch_plan(docs, batch, seq_len, rng)
    log(f"{args.split}: {len(docs)} documents, scoring {min(args.batches, len(order))} batches")

    tot, nbytes = 0.0, 0
    with torch.no_grad():
        for k in order[:args.batches]:
            out, mask = make_batch(docs, int(starts[k]), int(widths[k]), batch, seq_len)
            x = torch.from_numpy(out.astype(np.int64))
            mk = torch.from_numpy(mask)
            logits, aux = m(x)
            # the model predicts byte t+1 from byte t, so shift both target and mask
            ls = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1),
                                 reduction="none")
            mm = mk[:, 1:].reshape(-1)
            tot += float((ls * mm).sum())
            nbytes += float(mm.sum())
    bpb = (tot / max(nbytes, 1)) / math.log(2)
    log(f"held-out bpb {bpb:.4f} over {int(nbytes)} bytes")
    print(json.dumps({"ckpt": args.ckpt, "step": ck.get("step"), "bpb": bpb,
                      "bytes": nbytes}))


if __name__ == "__main__":
    main()
