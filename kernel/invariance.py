"""Encoding invariance: does the chunker learn characters, or this encoding?

The pre-registered claim (ticket 02) has two stages. Stage one (alignment above chance)
is measured. Stage two -- that the alignment survives re-encoding to UTF-16LE -- is not.

Method (ticket 11's amendments): the same text is presented to two models in two
encodings. Boundaries are compared in *character space*, not byte space: a UTF-8
boundary at byte t maps to the character index that byte starts; a UTF-16LE boundary
at byte t' maps to t' // 2, since UTF-16LE puts every BMP character at an even offset.

Reported per language:
  u8_lift / u16_lift -- alignment lift in each encoding (even-offset base rate for UTF-16)
  agreement          -- fraction of the union of boundary characters that both models hit
  char_f1            -- F1 between the two boundary sets
  bpc_u16            -- realised bytes/chunk of the UTF-16 model (2 bytes per character)

Passing means the UTF-16 model aligns on characters AND lands on the same characters as
the UTF-8 model. Failing means the chunker learned this encoding's byte phase.
"""
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/cfg.json")))
LANGS = CFG["languages"]
DATA = os.environ.get("DATA")
if not DATA:
    for root in ("/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/corpus",
                 "/kaggle/input/hnet-multiscript-bytes/corpus", "corpus"):
        if os.path.isdir(root):
            DATA = root
            break
CKPT_DIR = os.environ.get("CKPT_DIR", "/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/checkpoints_utf8")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
SEQ = CFG["seq_len"]
# training reads the UTF-16LE variant; evaluation reads the UTF-8 val text so
# both models are compared on identical characters
TRAIN_SUFFIX = CFG.get("train_suffix", "_u16")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class Stream:
    def __init__(self, rng, batch, seq_len):
        self.rng, self.batch, self.seq_len = rng, batch, seq_len
        self.data = [np.memmap(os.path.join(DATA, l + TRAIN_SUFFIX, "train.bin"),
                                 np.uint8, mode="r") for l in LANGS]
        self.lens = [len(d) for d in self.data]

    def batch_bytes(self):
        out = np.empty((self.batch, self.seq_len + 1), dtype=np.int64)
        for b in range(self.batch):
            i = int(self.rng.integers(len(self.data)))
            off = int(self.rng.integers(max(1, self.lens[i] - self.seq_len - 1)))
            out[b] = np.frombuffer(self.data[i][off:off + self.seq_len + 1], np.uint8)
        return out


def char_index_utf8(raw_bytes, nbytes):
    """For each byte offset, the character index it belongs to (start of its char)."""
    starts = [0]
    for i in range(nbytes):
        if (raw_bytes[i] & 0xC0) != 0x80:
            starts.append(i + 1)
    idx = np.zeros(nbytes, dtype=np.int64)
    c = 0
    for i in range(nbytes):
        while c + 1 < len(starts) and starts[c + 1] <= i:
            c += 1
        idx[i] = c
    return idx


def boundaries_to_chars(bmask, raw_bytes, utf16):
    """Boundary byte positions -> character indices."""
    if utf16:
        pos = np.flatnonzero(bmask)
        return set((pos // 2).tolist())
    idx = char_index_utf8(raw_bytes, len(raw_bytes))
    pos = np.flatnonzero(bmask)
    return set(idx[pos].tolist())


def evaluate_u16_only(u16_model, lang, windows=12, chars=96):
    """UTF-16 alignment alone, with no UTF-8 twin.

    The scale ladder asks whether the UTF-16 anti-alignment survives at 2.4 M parameters. That is
    a single-model measurement: the pairing in evaluate_pair exists to show that the same text
    chunks differently in two encodings, which is already established at 0.84 M. Pairing a 2.4 M
    UTF-16 model against 0.84 M UTF-8 checkpoints produces agreement numbers that mean nothing,
    so this path reports the lift and nothing else.
    """
    text = open(os.path.join(DATA, lang, "val.bin"), "rb").read().decode("utf-8", "replace")
    step = max(1, (len(text) - chars) // max(windows, 1))
    lifts, bpcs = [], []
    for w in range(windows):
        piece = text[w * step:w * step + chars]
        if len(piece) < 16:
            continue
        b16 = piece.encode("utf-16-le")
        while len(b16) > SEQ - 1 and len(piece) > 8:
            piece = piece[: len(piece) // 2]
            b16 = piece.encode("utf-16-le")
        if len(b16) > SEQ - 1:
            continue
        with torch.no_grad():
            i16 = torch.from_numpy(np.frombuffer(b16, np.uint8).astype(np.int64))[None]
            _, a16 = u16_model(i16)
        m16 = a16["bmask"][0].numpy()
        even = np.arange(len(b16)) % 2 == 0
        lifts.append(float(even[m16].mean() - even.mean()) if m16.any() else 0.0)
        bpcs.append(float(a16["bytes_per_chunk"]))
    if not lifts:
        return None
    return {"u16_lift": float(np.mean(lifts)), "bpc_u16": float(np.mean(bpcs)),
            "windows": len(lifts)}


def evaluate_pair(u8_model, u16_model, lang, windows=12, chars=96):
    """Run both models on the same text and compare boundaries in character space."""
    text = open(os.path.join(DATA, lang, "val.bin"), "rb").read().decode("utf-8", "replace")
    step = max(1, (len(text) - chars) // max(windows, 1))
    u8_lift, u16_lift, agree, f1s, u8_sets, u16_sets = [], [], [], [], [], []
    for w in range(windows):
        s = w * step
        piece = text[s:s + chars]
        if len(piece) < 16:
            continue
        b8 = piece.encode("utf-8")
        b16 = piece.encode("utf-16-le")
        # shrink until BOTH encodings fit the sequence length: UTF-16 doubles the
        # bytes and UTF-8 can triple them (CJK), so one window size cannot suit both
        while (len(b8) > SEQ - 1 or len(b16) > SEQ - 1) and len(piece) > 8:
            piece = piece[: len(piece) // 2]
            b8 = piece.encode("utf-8")
            b16 = piece.encode("utf-16-le")
        if len(b8) > SEQ - 1 or len(b16) > SEQ - 1:
            continue
        with torch.no_grad():
            i8 = torch.from_numpy(np.frombuffer(b8, np.uint8).astype(np.int64))[None]
            _, a8 = u8_model(i8)
            i16 = torch.from_numpy(np.frombuffer(b16, np.uint8).astype(np.int64))[None]
            _, a16 = u16_model(i16)
        m8 = a8["bmask"][0].numpy()
        m16 = a16["bmask"][0].numpy()
        # UTF-8 lift: boundary on a lead byte
        lead8 = (np.frombuffer(b8, np.uint8) & 0xC0) != 0x80
        u8_lift.append(lead8[m8].mean() - lead8.mean() if m8.any() else 0.0)
        # UTF-16 lift: boundary on an even byte offset
        even = np.arange(len(b16)) % 2 == 0
        u16_lift.append(even[m16].mean() - even.mean() if m16.any() else 0.0)
        s8 = boundaries_to_chars(m8, b8, False)
        s16 = boundaries_to_chars(m16, b16, True)
        u8_sets.append(s8)
        u16_sets.append(s16)
        if s8 or s16:
            inter = len(s8 & s16)
            union = len(s8 | s16)
            agree.append(inter / union if union else 0.0)
            p, r = (inter / len(s8) if s8 else 0.0), (inter / len(s16) if s16 else 0.0)
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return {"u8_lift": float(np.mean(u8_lift)) if u8_lift else None,
            "u16_lift": float(np.mean(u16_lift)) if u16_lift else None,
            "agreement": float(np.mean(agree)) if agree else None,
            "char_f1": float(np.mean(f1s)) if f1s else None,
            "windows": len(agree),
            "u8_boundaries": float(np.mean([len(s) for s in u8_sets])) if u8_sets else None,
            "u16_boundaries": float(np.mean([len(s) for s in u16_sets])) if u16_sets else None}


def train_u16(seed, deadline):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    fields["d_state"] = 16
    cfg = HNetConfig(**fields)
    model = HNetLM(cfg)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"],
                            betas=(0.9, 0.95))
    stream = Stream(np.random.default_rng(seed), CFG["batch"], CFG["seq_len"])
    hist = []
    t0 = time.time()
    for step in range(CFG["max_steps"]):
        batch = torch.from_numpy(stream.batch_bytes())
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss = ce + CFG["alpha"] * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if step % CFG["log_every"] == 0 or step == CFG["max_steps"] - 1:
            hist.append({"step": step, "ce": ce.item(), "bpc": aux["bytes_per_chunk"]})
            log(f"s{seed} step {step} ce {ce.item():.4f} bpc {aux['bytes_per_chunk']:.2f}")
        if time.time() > deadline:
            log("wall guard at step", step)
            break
    return model, hist


def main():
    log("config", json.dumps(CFG))
    log("data", DATA, "ckpts", CKPT_DIR)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt-s*-n4.pt")))
    log(f"{len(ckpts)} UTF-8 checkpoints")
    if CFG.get("pair", True):
        assert ckpts, "no UTF-8 checkpoints found"
    out = []
    for seed in CFG["seeds"]:
        deadline = time.time() + CFG.get("wall_guard_s", 8 * 3600) / len(CFG["seeds"])
        u16_model, hist = train_u16(seed, deadline)
        torch.save({"config": dict(u16_model.cfg.__dict__), "seed": seed,
                    "state": u16_model.state_dict()},
                   os.path.join(OUT, f"ckpt-u16-s{seed}.pt"))
        if not CFG.get("pair", True):
            u16_model.eval()
            res = {"u16_seed": seed, "mode": "u16_only"}
            for lang in LANGS:
                res[lang] = evaluate_u16_only(u16_model, lang)
                log(f"  s{seed} {lang}: u16_lift {res[lang]['u16_lift']:+.3f} "
                    f"bpc {res[lang]['bpc_u16']:.2f}")
            out.append(res)
            continue
        for ck in ckpts:
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            u8_model = HNetLM(HNetConfig(**blob["config"]))
            u8_model.load_state_dict(blob["state"])
            u8_model.eval()
            u16_model.eval()
            res = {"u8_ckpt": os.path.basename(ck), "u16_seed": seed}
            for lang in LANGS:
                res[lang] = evaluate_pair(u8_model, u16_model, lang)
                log(f"  s{seed} {os.path.basename(ck)} {lang}: "
                    f"u8_lift {res[lang]['u8_lift']} u16_lift {res[lang]['u16_lift']} "
                    f"agree {res[lang]['agreement']} f1 {res[lang]['char_f1']}")
            out.append(res)
        json.dump(out, open(os.path.join(OUT, "invariance.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
