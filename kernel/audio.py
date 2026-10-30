# ==== AUDIO DRIVER ====
# Four byte encodings of the same speech, one model each. The question is not "does it learn
# phonemes" -- it is the same question the text arm ended with: does the chunker follow the
# signal, or the layout of its serialization?
#
# Three measurements, in increasing strength:
#   1. ms_per_chunk vs bytes_per_chunk across arms. If the arms converge on a fixed *duration*
#      the chunker tracks time; if they converge on a fixed *byte count* it tracks layout.
#   2. onset alignment: do boundaries land on energy rises above the base rate?
#   3. LE vs BE: identical samples, identical byte count, every byte pair swapped. A chunker
#      following the signal puts boundaries at the same byte offsets in both. This needs no
#      semantic reference at all and is the sharpest test available.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/audio.json")))
DATA = os.environ.get("DATA")
for root in (() if DATA else ("/kaggle/input/datasets/mauricew/hnet-audio-bytes",
                              "/kaggle/input/hnet-audio-bytes",
                              "/kaggle/input/hnet-audio-bytes/corpus",
             "/kaggle/input/datasets/mauricew/hnet-audio-bytes/corpus", "corpus")):
    if os.path.isdir(root):
        DATA = root
        break
if not DATA:
    import glob
    print("INPUT:", glob.glob("/kaggle/input/*")[:20], glob.glob("/kaggle/input/datasets/*/*")[:30], flush=True)
    print("CWD:", os.getcwd(), sorted(os.listdir("."))[:20], flush=True)
assert DATA, "audio dataset not mounted"
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
SR = META["sample_rate"]
N = META["n_samples"]
ARMS = CFG["arms"]                      # e.g. ["pcm16le", "pcm16be", "mulaw", "f32le"]
BPS = {k: v["bytes_per_sample"] for k, v in META["encodings"].items()}
SEQ = {k: v["seq_len_for_32ms"] for k, v in META["encodings"].items()}

# N is a SAMPLE count; every byte offset must be scaled by the arm's bytes-per-sample.
# Using N directly as a byte offset reads only the first half of the pcm16le stream.
def nbytes(arm):
    return N * BPS[arm]


def val_start(arm):
    return int(N * 0.85) * BPS[arm]
_onsets = np.fromfile(os.path.join(DATA, "onsets.u32"), dtype="<u4").astype(np.int64)
ONSETS = np.sort(_onsets[_onsets < N])


def load_arm(arm):
    raw = np.memmap(os.path.join(DATA, f"{arm}.bin"), np.uint8, mode="r")
    return raw


def sample_of(byte_pos, arm):
    return byte_pos // BPS[arm]


def windows(arm, n_windows, seq_len, start, end):
    """Byte windows that all begin on a sample boundary of this encoding."""
    bps = BPS[arm]
    span = end - start
    step = max(1, span // (n_windows + 1))
    out = []
    for i in range(n_windows):
        b = start + i * step
        b -= b % bps                      # align to a sample so offsets are comparable
        if b + seq_len + 1 > end:
            continue
        out.append(b)
    return out


def onset_lift(bmask, base_byte, arm, tol_ms=8):
    """Fraction of boundaries within tol of an onset, against the base rate of such positions."""
    bps = BPS[arm]
    tol = int(tol_ms * SR / 1000)                  # tolerance in SAMPLES, like ONSETS
    pos = (np.flatnonzero(bmask) + base_byte) // bps   # byte offset -> sample index
    if len(pos) == 0:
        return None
    idx = np.searchsorted(ONSETS, pos)
    lo = np.clip(idx - 1, 0, len(ONSETS) - 1)
    hi = np.clip(idx, 0, len(ONSETS) - 1)
    d = np.minimum(np.abs(pos - ONSETS[lo]), np.abs(ONSETS[hi] - pos))
    hit = float(np.mean(d <= tol))
    # base rate: same question asked of every sample-aligned position in the window
    allpos = np.arange(base_byte, base_byte + len(bmask), bps) // bps
    i2 = np.searchsorted(ONSETS, allpos)
    l2 = np.clip(i2 - 1, 0, len(ONSETS) - 1)
    h2 = np.clip(i2, 0, len(ONSETS) - 1)
    d2 = np.minimum(np.abs(allpos - ONSETS[l2]), np.abs(ONSETS[h2] - allpos))
    base = float(np.mean(d2 <= tol))
    return {"hit": hit, "base": base, "lift": hit - base}


def phase_concentration(bmask):
    """Boundary concentration by offset mod 2: 1.0 = no preference."""
    pos = np.flatnonzero(bmask)
    if len(pos) < 4:
        return None
    ev = float(np.mean(pos % 2 == 0))
    return {"even_share": ev, "concentration": ev / 0.5}


def evaluate(model, arm, seq_len, n_windows=10):
    raw = load_arm(arm)
    model.eval()
    rows = []
    with torch.no_grad():
        for b in windows(arm, n_windows, seq_len, val_start(arm), nbytes(arm)):
            arr = np.frombuffer(raw[b:b + seq_len + 1], np.uint8).astype(np.int64)
            x = torch.from_numpy(arr)[None, :].to(DEV)
            logits, aux = model(x[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))
            bm = aux["bmask"][0].cpu().numpy()
            rows.append({"bpb": ce.item() / math.log(2),
                         "bytes_per_chunk": aux["bytes_per_chunk"],
                         "onset": onset_lift(bm, b, arm),
                         "phase": phase_concentration(bm),
                         "boundaries": bm})
    model.train()
    if not rows:
        return None
    ol = [r["onset"] for r in rows if r["onset"]]
    out = {"bpb": float(np.mean([r["bpb"] for r in rows])),
           "bytes_per_chunk": float(np.mean([r["bytes_per_chunk"] for r in rows])),
           "ms_per_chunk": float(np.mean([r["bytes_per_chunk"] for r in rows]))
                            / BPS[arm] / SR * 1000,
           "onset_lift": float(np.mean([o["lift"] for o in ol])) if ol else None,
           "onset_hit": float(np.mean([o["hit"] for o in ol])) if ol else None,
           "onset_base": float(np.mean([o["base"] for o in ol])) if ol else None,
           "even_share": float(np.mean([r["phase"]["even_share"] for r in rows
                                        if r["phase"]])) if any(r["phase"] for r in rows) else None}
    return out


def train_arm(arm, seed, deadline):
    seq_len = SEQ[arm]
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
    train_end = val_start(arm)
    history, evals = [], []
    t0, stop = time.time(), "max_steps"
    for step in range(CFG["max_steps"]):
        starts = rng.integers(0, max(1, train_end - seq_len - 2), size=CFG["batch"])
        starts -= starts % BPS[arm]
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
                            "ms_per_chunk": aux["bytes_per_chunk"] / BPS[arm] / SR * 1000,
                            "grad_norm": float(gn), "elapsed_s": round(time.time() - t0, 1)})
            log(f"{arm} s{seed} step {step} ce {ce.item():.4f} bpc "
                f"{aux['bytes_per_chunk']:.2f} ({aux['bytes_per_chunk']/BPS[arm]/SR*1000:.1f} ms)")
        if step % CFG["eval_every"] == 0 and step:
            ev = evaluate(model, arm, seq_len)
            evals.append({"step": step, **ev})
            log(f"  eval {arm} bpb {ev['bpb']:.3f} bpc {ev['bytes_per_chunk']:.2f} "
                f"ms {ev['ms_per_chunk']:.1f} onset {ev['onset_lift'] or 0:+.3f}")
        if time.time() > deadline:
            stop = f"wall_guard@{step}"
            break
    torch.save({"config": dict(cfg.__dict__), "seed": seed, "arm": arm,
                "state": model.state_dict()}, os.path.join(OUT, f"ckpt-{arm}-s{seed}.pt"))
    final = evaluate(model, arm, seq_len, n_windows=40)
    json.dump({"arm": arm, "seed": seed, "stop": stop, "steps": step + 1,
               "final": final, "history": history, "evals": evals},
              open(os.path.join(OUT, f"arm-{arm}-s{seed}.json"), "w"), indent=2)
    return model, final


def cross_arm(models, seed):
    """LE vs BE at identical byte offsets -- the test that needs no semantic reference.

    Both streams have the same byte count and the same sample positions, so a chunker that
    follows the signal must place boundaries at the same offsets. Agreement is reported
    against the chance level implied by each model's own boundary rate.
    """
    out = {}
    if "pcm16le" not in models or "pcm16be" not in models:
        return out
    seq_len = min(SEQ["pcm16le"], SEQ["pcm16be"])
    raw = {a: load_arm(a) for a in ("pcm16le", "pcm16be")}
    wins = windows("pcm16le", 40, seq_len, val_start("pcm16le"), nbytes("pcm16le"))
    agree, f1s, rates = [], [], []
    cross = []
    for b in wins:
        bm = {}
        for a in ("pcm16le", "pcm16be"):
            arr = np.frombuffer(raw[a][b:b + seq_len], np.uint8).astype(np.int64)
            x = torch.from_numpy(arr)[None, :].to(DEV)
            with torch.no_grad():
                _, aux = models[a](x)
            bm[a] = aux["bmask"][0].cpu().numpy()
        le, be = bm["pcm16le"], bm["pcm16be"]
        inter = float(np.sum(le & be))
        union = float(np.sum(le | be))
        agree.append(inter / union if union else 0.0)
        p, r = (inter / max(1.0, le.sum())), (inter / max(1.0, be.sum()))
        f1s.append(2 * p * r / max(1e-9, p + r))
        rates.append((float(le.mean()), float(be.mean())))
        # zero-shot: the LE model reading BE bytes, same offsets
        arr = np.frombuffer(raw["pcm16be"][b:b + seq_len], np.uint8).astype(np.int64)
        with torch.no_grad():
            _, aux2 = models["pcm16le"](torch.from_numpy(arr)[None, :].to(DEV))
        zs = aux2["bmask"][0].cpu().numpy()
        cross.append(float(np.sum(le & zs)) / max(1.0, float(np.sum(le | zs))))
    rate_le, rate_be = float(np.mean([r[0] for r in rates])), float(np.mean([r[1] for r in rates]))
    chance = rate_le * rate_be / max(1e-9, rate_le + rate_be - rate_le * rate_be)
    out = {"jaccard": float(np.mean(agree)), "f1": float(np.mean(f1s)),
           "chance_jaccard": chance, "rate_le": rate_le, "rate_be": rate_be,
           "zero_shot_le_on_be": float(np.mean(cross)), "n_windows": len(wins)}
    log(f"LE vs BE: jaccard {out['jaccard']:.3f} vs chance {chance:.3f} | f1 {out['f1']:.3f} "
        f"| LE model on BE bytes {out['zero_shot_le_on_be']:.3f}")
    return out


def main():
    log("device", DEV, "arms", ARMS, "sr", SR, "samples", N)
    t_start = time.time()
    models, summary = {}, []
    for i, arm in enumerate(ARMS):
        share = (CFG["wall_guard_s"] - (time.time() - t_start)) / (len(ARMS) - i)
        log(f"arm {i+1}/{len(ARMS)}: {arm} seq_len {SEQ[arm]}")
        m, final = train_arm(arm, CFG["seeds"][0], time.time() + share)
        models[arm] = m
        summary.append({"arm": arm, "final": final})
    res = {"config": CFG, "device": DEV, "per_arm": summary,
           "cross": cross_arm(models, CFG["seeds"][0])}
    json.dump(res, open(os.path.join(OUT, "audio.json"), "w"), indent=2)
    log("done")
    for s in summary:
        f = s["final"] or {}
        log(f"  {s['arm']:>8}: bpb {f.get('bpb', float('nan')):.3f} "
            f"bpc {f.get('bytes_per_chunk', float('nan')):.2f} "
            f"ms {f.get('ms_per_chunk', float('nan')):.1f} "
            f"onset {f.get('onset_lift') or 0:+.3f}")


if __name__ == "__main__":
    main()
