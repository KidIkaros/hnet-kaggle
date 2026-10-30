# ==== VISION DRIVER ====
# Images serialized to bytes. The question is the same one the text arm ended on: does the
# chunker follow the data or the layout of its encoding?
#
# A 64x64 RGB image is 12288 bytes: pixels are the "characters" (3 bytes each) and rows are the
# "words" (192 bytes each). So the two hypotheses make different predictions that can be told
# apart in one run:
#
#   layout    -> boundaries on pixel starts (offset mod 3) or row starts (offset mod 192)
#   semantics -> boundaries on edges and on object outlines
#
# `bgr` is the audio arm's LE/BE trick in image form: identical byte count, identical pixel
# grid, only the channel order inside each pixel changes.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/vision.json")))
DATA = os.environ.get("DATA")
for root in (() if DATA else ("/kaggle/input/hnet-vision-bytes",
                              "/kaggle/input/datasets/mauricew/hnet-vision-bytes")):
    if os.path.isdir(root):
        DATA = root
        break
assert DATA, "vision dataset not mounted"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


META = json.load(open(os.path.join(DATA, "meta.json")))
SIZE = META["size"]
NIMG = META["n_images"]
BPP = META["bytes_per_pixel"]
ROW = META["row_bytes"]
ARMS = CFG["arms"]
EDGES = np.fromfile(os.path.join(DATA, "edges.bin"), np.uint8).reshape(NIMG, SIZE, SIZE)
OBJ = np.fromfile(os.path.join(DATA, "objmask.bin"), np.uint8).reshape(NIMG, SIZE, SIZE)
VAL_FROM = int(NIMG * 0.85)


def load_arm(arm):
    return np.memmap(os.path.join(DATA, f"{arm}.bin"), np.uint8, mode="r")


def image_slice(arm, i):
    bpi = META["bytes_per_image"][arm]
    return i * bpi, (i + 1) * bpi


def decompose(arm, pos):
    """byte offset inside an image -> (channel index, row, column)."""
    bpp, row = BPP[arm], ROW[arm]
    y = (pos // row) % SIZE
    x = (pos % row) // bpp
    c = pos % bpp
    return c, y, x


def alignment(arm, bmask, base, img_idx):
    """Lift of boundary placement onto edges / objects / pixel starts / row starts."""
    pos = np.flatnonzero(bmask)
    if len(pos) < 8:
        return None
    c, y, x = decompose(arm, pos + base)
    e = EDGES[img_idx].ravel()[y * SIZE + x].astype(bool)
    o = OBJ[img_idx].ravel()[y * SIZE + x].astype(bool)
    # base rates over every sample position in the same window, same mapping
    allpos = np.arange(base, base + len(bmask))
    c2, y2, x2 = decompose(arm, allpos)
    e2 = EDGES[img_idx].ravel()[y2 * SIZE + x2].astype(bool)
    o2 = OBJ[img_idx].ravel()[y2 * SIZE + x2].astype(bool)
    bpp = BPP[arm]
    return {"edge_lift": float(e.mean() - e2.mean()),
            "obj_lift": float(o.mean() - o2.mean()),
            "pixel_start": float(np.mean(c == 0)) if bpp > 1 else None,
            "row_start": float(np.mean(x == 0)),
            "channel0_share": float(np.mean(c == 0)) if bpp > 1 else None}


def evaluate(model, arm, seq_len, n_images=24):
    raw = load_arm(arm)
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(VAL_FROM, min(VAL_FROM + n_images, NIMG)):
            s, e = image_slice(arm, i)
            for base in range(s, e - seq_len - 1, seq_len):
                arr = np.frombuffer(raw[base:base + seq_len + 1], np.uint8).astype(np.int64)
                x = torch.from_numpy(arr)[None, :].to(DEV)
                logits, aux = model(x[:, :-1])
                ce = F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))
                bm = aux["bmask"][0].cpu().numpy()
                a = alignment(arm, bm, base, i)
                if a:
                    rows.append({"bpb": ce.item() / math.log(2),
                                 "bytes_per_chunk": aux["bytes_per_chunk"], **a})
    model.train()
    if not rows:
        return None
    keys = ["bpb", "bytes_per_chunk", "edge_lift", "obj_lift", "row_start"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["pixel_start"] = (float(np.mean([r["pixel_start"] for r in rows
                                         if r["pixel_start"] is not None]))
                          if any(r["pixel_start"] is not None for r in rows) else None)
    out["n_windows"] = len(rows)
    return out


def train_arm(arm, seed, deadline):
    seq_len = CFG["seq_len"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"],
                            betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    raw = load_arm(arm)
    bpi = META["bytes_per_image"][arm]
    train_imgs = VAL_FROM
    history, evals = [], []
    t0, stop, step = time.time(), "max_steps", 0
    for step in range(CFG["max_steps"]):
        idx = rng.integers(0, train_imgs, size=CFG["batch"])
        starts = np.array([image_slice(arm, i)[0] + rng.integers(0, bpi - seq_len - 1)
                           for i in idx])
        b64 = np.stack([np.frombuffer(raw[s:s + seq_len + 1], np.uint8) for s in starts])
        batch = torch.from_numpy(b64.astype(np.int64)).to(DEV)
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss = ce + CFG["alpha"] * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, CFG)
        opt.step()
        if step % CFG["log_every"] == 0 or step == CFG["max_steps"] - 1:
            history.append({"step": step, "ce": ce.item(), "bpb": ce.item() / math.log(2),
                            "bytes_per_chunk": aux["bytes_per_chunk"],
                            "grad_norm": float(gn), "elapsed_s": round(time.time() - t0, 1)})
            log(f"{arm} s{seed} step {step} ce {ce.item():.4f} bpc {aux['bytes_per_chunk']:.2f}")
        if step % CFG["eval_every"] == 0 and step:
            ev = evaluate(model, arm, seq_len)
            evals.append({"step": step, **ev})
            log(f"  eval {arm} bpb {ev['bpb']:.3f} bpc {ev['bytes_per_chunk']:.2f} "
                f"edge {ev['edge_lift']:+.3f} obj {ev['obj_lift']:+.3f} "
                f"pix {ev['pixel_start'] if ev['pixel_start'] is None else round(ev['pixel_start'],3)} "
                f"row {ev['row_start']:.3f}")
        if time.time() > deadline:
            stop = f"wall_guard@{step}"
            break
    torch.save({"config": dict(cfg.__dict__), "seed": seed, "arm": arm,
                "state": model.state_dict()}, os.path.join(OUT, f"ckpt-{arm}-s{seed}.pt"))
    final = evaluate(model, arm, seq_len, n_images=60)
    json.dump({"arm": arm, "seed": seed, "stop": stop, "steps": step + 1, "final": final,
               "history": history, "evals": evals},
              open(os.path.join(OUT, f"arm-{arm}-s{seed}.json"), "w"), indent=2)
    return model, final


def main():
    log("device", DEV, "arms", ARMS, "images", NIMG, "size", SIZE)
    summary = []
    t0 = time.time()
    for i, arm in enumerate(ARMS):
        share = (CFG["wall_guard_s"] - (time.time() - t0)) / (len(ARMS) - i)
        log(f"arm {i+1}/{len(ARMS)}: {arm}")
        _, final = train_arm(arm, CFG["seeds"][0], time.time() + share)
        summary.append({"arm": arm, "final": final})
    json.dump({"config": CFG, "device": DEV, "per_arm": summary},
              open(os.path.join(OUT, "vision.json"), "w"), indent=2)
    for s in summary:
        f = s["final"] or {}
        log(f"  {s['arm']:>7}: bpb {f.get('bpb', float('nan')):.3f} "
            f"edge {f.get('edge_lift', 0):+.3f} obj {f.get('obj_lift', 0):+.3f} "
            f"row {f.get('row_start', float('nan')):.3f}")


if __name__ == "__main__":
    main()
