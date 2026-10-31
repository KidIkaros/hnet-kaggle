"""Cross-encoding test in pixel space: does chunking follow the image or the bytes?

`rgb` and `bgr` are the same pixels with the channel order permuted inside each pixel. The pixel
grid is identical, the byte count is identical, and the layout is identical -- only the byte
*values* differ. So:

  - if the chunker follows the image, its boundaries land on the same pixels in both, and the
    pixel-space Jaccard is well above chance;
  - if it follows byte values, the boundaries move and the Jaccard sits at chance.

This is the audio arm's LE/BE test in image form, and it needs no semantic reference at all.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import HNetConfig, HNetLM

DATA = "/home/ikaaros/.hermes/cache/scratch/vis/packed"
RES = "/home/ikaaros/.hermes/cache/scratch/visres"
META = json.load(open(os.path.join(DATA, "meta.json")))
SIZE, NIMG = META["size"], META["n_images"]
EDGES = np.fromfile(os.path.join(DATA, "edges.bin"), np.uint8).reshape(NIMG, SIZE, SIZE)
OBJ = np.fromfile(os.path.join(DATA, "objmask.bin"), np.uint8).reshape(NIMG, SIZE, SIZE)
SEQ = 256


def load(arm):
    blob = torch.load(os.path.join(RES, f"ckpt-{arm}-s0.pt"), map_location="cpu",
                      weights_only=False)
    m = HNetLM(HNetConfig(**blob["config"]))
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def boundaries(model, raw, base, n=SEQ):
    arr = np.frombuffer(raw[base:base + n], np.uint8).astype(np.int64)
    with torch.no_grad():
        _, aux = model(torch.from_numpy(arr)[None, :])
    return aux["bmask"][0].numpy()


def main():
    models = {a: load(a) for a in ("rgb", "bgr")}
    raw = {a: np.memmap(os.path.join(DATA, f"{a}.bin"), np.uint8, mode="r")
           for a in ("rgb", "bgr")}
    bpi = META["bytes_per_image"]["rgb"]
    val = list(range(int(NIMG * 0.85), NIMG, 8))   # subsample: 28k forwards is too slow on CPU

    jac, f1, edge_hit, obj_hit, rates = [], [], [], [], []
    for i in val:
        s = i * bpi
        for base in range(s, s + bpi - SEQ - 1, SEQ * 4):
            bm = {a: boundaries(models[a], raw[a], base) for a in ("rgb", "bgr")}
            # byte offset -> pixel index, so the two arms are compared in the same space
            px = {a: np.unique(np.flatnonzero(bm[a]) // 3) for a in ("rgb", "bgr")}
            A, B = set(px["rgb"].tolist()), set(px["bgr"].tolist())
            if not A or not B:
                continue
            inter = len(A & B)
            jac.append(inter / len(A | B))
            p, r = inter / len(A), inter / len(B)
            f1.append(2 * p * r / max(1e-9, p + r))
            rates.append((len(A) / (SEQ / 3), len(B) / (SEQ / 3)))
            # per-arm semantic check on the same windows
            for a in ("rgb", "bgr"):
                pos = np.flatnonzero(bm[a])
                y = ((pos + base - s) // META["row_bytes"][a]) % SIZE
                x = ((pos + base - s) % META["row_bytes"][a]) // 3
                allpos = np.arange(base - s, base - s + SEQ)
                y2 = (allpos // META["row_bytes"][a]) % SIZE
                x2 = (allpos % META["row_bytes"][a]) // 3
                if a == "rgb":
                    edge_hit.append(EDGES[i].ravel()[y * SIZE + x].mean()
                                    - EDGES[i].ravel()[y2 * SIZE + x2].mean())
                    obj_hit.append(OBJ[i].ravel()[y * SIZE + x].mean()
                                   - OBJ[i].ravel()[y2 * SIZE + x2].mean())

    rle, rbe = np.mean([r[0] for r in rates]), np.mean([r[1] for r in rates])
    chance = rle * rbe / max(1e-9, rle + rbe - rle * rbe)
    print(f"windows compared: {len(jac)}")
    print(f"pixel-space Jaccard  rgb vs bgr: {np.mean(jac):.3f}   chance {chance:.3f}")
    print(f"pixel-space F1       rgb vs bgr: {np.mean(f1):.3f}")
    print(f"boundary rate (pixels): rgb {rle:.3f}  bgr {rbe:.3f}")
    print(f"rgb edge lift {np.mean(edge_hit):+.3f}   object lift {np.mean(obj_hit):+.3f}")


if __name__ == "__main__":
    main()
