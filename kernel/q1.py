# ==== PREAMBLE ====
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "configs/q1.json")))
WALL_GUARD_S = float(CFG["wall_guard_s"])
LANGS = CFG["languages"]
DATA = os.environ.get("DATA")
if not DATA:
    for root in ("/kaggle/input/datasets/mauricew/hnet-multiscript-bytes/corpus",
                 "/kaggle/input/hnet-multiscript-bytes/corpus", "corpus"):
        if os.path.isdir(root):
            DATA = root
            break
assert DATA, "corpus dataset not mounted"
os.makedirs(OUT, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ==== DATA ====
def mem(lang, split):
    return np.memmap(os.path.join(DATA, lang, f"{split}.bin"), dtype=np.uint8, mode="r")


class Stream:
    """Round-robin byte windows. `origin` lets alignment be read per script."""

    def __init__(self, rng, batch, seq_len):
        self.rng, self.batch, self.seq_len = rng, batch, seq_len
        self.data = [mem(l, "train") for l in LANGS]
        self.lens = [len(d) for d in self.data]

    def batch_bytes(self):
        out = np.empty((self.batch, self.seq_len + 1), dtype=np.int64)
        origin = []
        for b in range(self.batch):
            i = int(self.rng.integers(len(self.data)))
            off = int(self.rng.integers(max(1, self.lens[i] - self.seq_len - 1)))
            out[b] = np.frombuffer(self.data[i][off:off + self.seq_len + 1], np.uint8)
            origin.append(i)
        return out, origin


def alignment(batch, bmask):
    """Lift of boundary placement over the per-sequence lead-byte base rate.

    bmask[t] flags byte t, so compare byte t (batch[:, :-1]) -- not t+1.
    """
    lead = (batch[:, :-1].numpy() & 0xC0) != 0x80
    bm = bmask.numpy()
    lift, on_lead, base = [], [], []
    for b in range(bm.shape[0]):
        if bm[b].any():
            on_lead.append(lead[b][bm[b]].mean())
            base.append(lead[b].mean())
            lift.append(on_lead[-1] - base[-1])
    if not lift:
        return None, None, None
    return float(np.mean(lift)), float(np.mean(on_lead)), float(np.mean(base))


def evaluate(model, seq_len, windows=8):
    """Per-language bpb, realised bytes/chunk and alignment lift on held-out windows."""
    model.eval()
    scores = {}
    with torch.no_grad():
        for lang in LANGS:
            data = mem(lang, "val")
            n = len(data)
            if n < seq_len + 1:
                continue
            offs = np.linspace(0, n - seq_len - 2, windows).astype(int)
            arr = np.stack([np.frombuffer(data[o:o + seq_len + 1], np.uint8) for o in offs])
            batch = torch.from_numpy(arr.astype(np.int64)).to(DEV)
            logits, aux = model(batch[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
            lift, on_lead, base = alignment(batch.cpu(), aux["bmask"].cpu())
            scores[lang] = {"bpb": ce.item() / math.log(2),
                            "bytes_per_chunk": aux["bytes_per_chunk"],
                            "align_lift": lift, "align_on_lead": on_lead,
                            "lead_base": base}
    model.train()
    return scores


# ==== TRAIN ====
def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def run_seed(seed, deadline, ratio=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    if ratio is not None:
        fields["target_ratio"] = ratio
    cfg = HNetConfig(**fields)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    mo = float(CFG.get("lr_mult_outer", 1.0))
    mm = float(CFG.get("lr_mult_main", 1.0))
    groups = param_groups(model, mo, mm)
    opt = torch.optim.AdamW(groups, lr=CFG["lr"],
                            weight_decay=CFG["wd"], betas=(0.9, 0.95))
    stream = Stream(np.random.default_rng(seed), CFG["batch"], CFG["seq_len"])
    history, evals = [], []
    t0, done, stop = time.time(), 0, "max_steps"

    for step in range(CFG["max_steps"]):
        b64, origin = stream.batch_bytes()
        batch = torch.from_numpy(b64).to(DEV)
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss = ce + CFG["alpha"] * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, CFG) * g.get("lr_mult", 1.0)
        opt.step()
        done = step + 1

        if step % CFG["log_every"] == 0 or step == CFG["max_steps"] - 1:
            lift, on_lead, base = alignment(batch.cpu(), aux["bmask"].cpu())
            history.append({"step": step, "seed": seed, "ce": ce.item(),
                            "bpb": ce.item() / math.log(2),
                            "ratio_loss": aux["ratio_loss"].item(),
                            "F": aux["F"], "G": aux["G"],
                            "bytes_per_chunk": aux["bytes_per_chunk"],
                            "realised_ratio": 1.0 / max(aux["bytes_per_chunk"], 1e-9),
                            "align_lift": lift, "align_on_lead": on_lead,
                            "lead_base": base, "grad_norm": float(gn),
                            "lr": lr_at(step, CFG),
                            "langs": ",".join(LANGS[i] for i in origin),
                            "elapsed_s": round(time.time() - t0, 1)})
            log(f"s{seed} step {step} ce {ce.item():.4f} rl {aux['ratio_loss'].item():.3f} "
                f"F {aux['F']:.3f} bpc {aux['bytes_per_chunk']:.2f} "
                f"lift {'n/a' if lift is None else f'{lift:+.3f}'}")
        if step % CFG["eval_every"] == 0 and step:
            ev = evaluate(model, CFG["seq_len"])
            evals.append({"step": step, "seed": seed, "per_language": ev})
            for lang, s in ev.items():
                log(f"  s{seed} eval {lang} bpb {s['bpb']:.3f} bpc "
                    f"{s['bytes_per_chunk']:.2f} lift {s['align_lift'] or 0:+.3f}")
        if time.time() > deadline:
            stop = f"wall_guard@{step}"
            log("wall guard hit at step", step)
            break

    tag = f"s{seed}" if ratio is None else f"s{seed}-n{ratio:g}"
    torch.save({"config": dict(cfg.__dict__), "seed": seed,
                "state": model.state_dict()}, os.path.join(OUT, f"ckpt-{tag}.pt"))
    json.dump({"seed": seed, "ratio": cfg.target_ratio, "stop": stop, "steps": done,
               "history": history, "evals": evals},
              open(os.path.join(OUT, f"seed-{tag}.json"), "w"), indent=2)
    return model, {"seed": seed, "ratio": cfg.target_ratio, "stop": stop,
                   "steps": done, "evals": evals}


def examples(model, seed, per_lang=20, seq_len=192):
    """Boundary-marked samples, bytes annotated with the UTF-8 DFA state."""
    model.eval()
    out = {}
    with torch.no_grad():
        for lang in LANGS:
            data, states = mem(lang, "val"), np.memmap(
                os.path.join(DATA, lang, "val_state.u8"), np.uint8, mode="r")
            rows, n = [], len(data)
            for o in np.linspace(0, max(0, n - seq_len - 2), per_lang).astype(int):
                raw = np.frombuffer(data[o:o + seq_len], np.uint8)
                st = np.frombuffer(states[o:o + seq_len], np.uint8)
                batch = torch.from_numpy(raw.astype(np.int64))[None, :].to(DEV)
                _, aux = model(batch)
                bm = aux["bmask"][0].cpu().numpy()
                rows.append({"text": raw.tobytes().decode("utf-8", "replace"),
                             "state": st.tolist(), "boundary": bm.tolist(),
                             "on_lead": bool(((raw & 0xC0) != 0x80)[bm].all())})
            out[lang] = rows
    json.dump(out, open(os.path.join(OUT, f"examples-s{seed}.json"), "w"))
    return out


# ==== MAIN ====
def main():
    log("device", DEV, "torch", torch.__version__)
    log("config", json.dumps(CFG))
    t_start = time.time()
    summary = []
    ratios = CFG.get("sweep_ratios")
    jobs = ([(seed, r) for seed in CFG["seeds"] for r in ratios] if ratios
            else [(seed, None) for seed in CFG["seeds"]])
    for i, (seed, ratio) in enumerate(jobs):
        share = (WALL_GUARD_S - (time.time() - t_start)) / (len(jobs) - i)
        log(f"job {i + 1}/{len(jobs)} seed {seed} ratio {ratio}")
        model, s = run_seed(seed, time.time() + share, ratio)
        s["examples"] = examples(model, seed)
        summary.append(s)
    json.dump({"config": CFG, "seeds": summary, "device": DEV},
              open(os.path.join(OUT, "q1.json"), "w"), indent=2)
    log("done", [(s["seed"], s.get("ratio"), s["stop"], s["steps"]) for s in summary])


if __name__ == "__main__":
    main()
