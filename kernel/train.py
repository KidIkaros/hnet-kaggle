# ==== PREAMBLE ====
import glob
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyarrow.parquet as pq
except ImportError:                       # not preinstalled on every image
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], check=True)
    import pyarrow.parquet as pq

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CORPUS = os.path.join(OUT, "corpus")
CFG = dict(d_model=128, n_enc=2, n_dec=2, n_main=2, d_state=16, scan_chunk=64,
           seq_len=256, batch=4, lr=1e-3, wd=0.1, warmup=20, clip=1.0,
           target_ratio=6.0, alpha=0.03, max_steps=200, log_every=10)
_ov = os.environ.get("CFG_OVERRIDE")
if _ov:
    CFG.update(json.loads(_ov))
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 8.0 * 3600))
PACK_MB = float(os.environ.get("PACK_MB", 50))
LANGUAGES = os.environ.get("LANGUAGES", "eng deu rus cmn jpn arb").split()

torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ==== TRAIN ====
def prepare():
    """Pack if absent, then read the corpus back and check it before training."""
    os.makedirs(CORPUS, exist_ok=True)
    paths = [os.path.join(CORPUS, l, "train.bin") for l in LANGUAGES]
    if all(os.path.exists(p) for p in paths):
        log("corpus present, skipping download")
    else:
        log(f"streaming {PACK_MB} MB/lang x {len(LANGUAGES)} langs (text column only)")
        for lang in LANGUAGES:
            if not os.path.exists(os.path.join(CORPUS, lang, "train.bin")):
                pack_lang(lang, CORPUS, int(PACK_MB * 1e6), int(0.5 * 1e6))

    # The corpus choice is 17 days old: read it back and check it before training.
    report = {}
    for lang in LANGUAGES:
        meta = json.load(open(os.path.join(CORPUS, lang, "meta.json")))
        raw = open(os.path.join(CORPUS, lang, "train.bin"), "rb").read()
        text = raw.decode("utf-8")            # raises if a character was split
        clean = text.replace("\n", " ")
        offs = np.frombuffer(
            open(os.path.join(CORPUS, lang, "train_docs.u32"), "rb").read(), dtype=np.uint32
        )
        states = np.frombuffer(
            open(os.path.join(CORPUS, lang, "train_state.u8"), "rb").read(), dtype=np.uint8
        )
        report[lang] = {
            "bytes": len(raw), "docs": meta["docs"],
            "lang_score_median": meta["language_score_median"],
            "html_tag_rate": clean.count("<") / max(len(clean), 1),
            "doc_offsets_monotonic": bool((np.diff(offs.astype(np.int64)) > 0).all()),
            "doc_offsets_in_range": bool(len(offs) and offs.max() < len(raw)),
            "state_stream_len_ok": len(states) == len(raw),
            "lead_state_fraction": float((states == 0).mean()),
            "sample": clean[:200],
        }
        assert report[lang]["doc_offsets_monotonic"], f"{lang}: doc offsets not increasing"
        assert report[lang]["doc_offsets_in_range"], f"{lang}: doc offset past end"
        assert report[lang]["state_stream_len_ok"], f"{lang}: state stream length mismatch"
        log(f"  verify {lang}: {len(raw)}B {meta['docs']}docs "
            f"lang={meta['language_score_median']:.3f} "
            f"lead_state={report[lang]['lead_state_fraction']:.3f} :: {clean[:60]}")
    json.dump(report, open(os.path.join(OUT, "corpus_report.json"), "w"), indent=2)
    return paths


class Stream:
    """Byte windows drawn round-robin from the per-language streams."""

    def __init__(self, rng, batch, seq_len, paths):
        self.rng, self.batch, self.seq_len = rng, batch, seq_len
        self.data = [np.memmap(p, dtype=np.uint8, mode="r") for p in paths]
        self.lens = [len(d) for d in self.data]

    def batch_bytes(self):
        out = np.empty((self.batch, self.seq_len + 1), dtype=np.int64)
        origin = []
        for b in range(self.batch):
            i = int(self.rng.integers(len(self.data)))
            off = int(self.rng.integers(max(1, self.lens[i] - self.seq_len - 1)))
            out[b] = np.frombuffer(self.data[i][off:off + self.seq_len + 1],
                                   dtype=np.uint8)
            origin.append(i)
        return out, origin


def lr_at(step):
    if step < CFG["warmup"]:
        return CFG["lr"] * (step + 1) / CFG["warmup"]
    span = max(1, CFG["max_steps"] - CFG["warmup"])
    p = (step - CFG["warmup"]) / span
    return CFG["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def evaluate(model, paths, seq_len, windows=8):
    """Fixed windows from each language's val.bin: per-language bpc and alignment.

    Random batch composition makes the training-step alignment number swing with
    which scripts happened to be drawn; this is the per-language read.
    """
    scores = {}
    model.eval()
    with torch.no_grad():
        for lang, path in zip(LANGUAGES, paths):
            data = np.memmap(path, dtype=np.uint8, mode="r")
            vpath = path.replace("train.bin", "val.bin")
            if os.path.exists(vpath):
                data = np.memmap(vpath, dtype=np.uint8, mode="r")
            n = len(data)
            if n < seq_len + 1:
                continue
            offs = np.linspace(0, n - seq_len - 2, windows).astype(int)
            batch = torch.from_numpy(np.stack(
                [np.frombuffer(data[o:o + seq_len + 1], dtype=np.uint8) for o in offs]
            ).astype(np.int64))
            logits, aux = model(batch[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
            lead = ((batch[:, :-1].numpy() & 0xC0) != 0x80)
            bm = aux["bmask"].numpy()
            on_lead, base = [], []
            for b in range(bm.shape[0]):
                if bm[b].any():
                    on_lead.append(lead[b][bm[b]].mean())
                    base.append(lead[b].mean())
            scores[lang] = {
                "bpb": ce.item() / math.log(2),
                "bytes_per_chunk": aux["bytes_per_chunk"],
                "align_on_lead_byte": float(np.mean(on_lead)) if on_lead else None,
                "lead_byte_base_rate": float(np.mean(base)) if base else None,
                "align_lift": (float(np.mean(on_lead)) - float(np.mean(base)))
                if on_lead and base else None,
            }
    model.train()
    return scores


def main():
    log("config", CFG)
    log("torch", torch.__version__, "threads", torch.get_num_threads(),
        "cpus", os.cpu_count())
    paths = prepare()

    cfg = HNetConfig(**{k: v for k, v in CFG.items()
                        if k in HNetConfig.__dataclass_fields__})
    model = HNetLM(cfg)
    init_weights(model)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params {n_params / 1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["wd"], betas=(0.9, 0.95))
    stream = Stream(np.random.default_rng(0), CFG["batch"], CFG["seq_len"], paths)
    history, evals, t0 = [], [], time.time()
    eval_every = max(1, CFG["max_steps"] // 4)

    for step in range(CFG["max_steps"]):
        batch, origin = stream.batch_bytes()
        batch = torch.from_numpy(batch)
        logits, aux = model(batch[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), batch[:, 1:].reshape(-1))
        loss = ce + CFG["alpha"] * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()

        if step % CFG["log_every"] == 0 or step == CFG["max_steps"] - 1:
            # Boundary alignment, as a lift over the base rate. Plain
            # mean(lead[bmask]) tracks the sampled script (pure-ASCII English is
            # ~100% lead bytes, CJK ~33%) and so oscillates with batch
            # composition -- subtracting the base rate per sequence makes the
            # number comparable across languages and steps.
            # bmask[t] means "byte t starts a chunk", so compare byte t itself
            # (batch[:, :-1]) -- not the byte it predicts, which is t+1.
            lead = ((batch[:, :-1].numpy() & 0xC0) != 0x80)
            bm = aux["bmask"].numpy()
            lift, on_lead, base = [], [], []
            for b in range(bm.shape[0]):
                if bm[b].any():
                    on_lead.append(lead[b][bm[b]].mean())
                    base.append(lead[b].mean())
                    lift.append(on_lead[-1] - base[-1])
            row = {"step": step, "ce": ce.item(), "bpb": ce.item() / math.log(2),
                   "loss": loss.item(), "ratio_loss": aux["ratio_loss"].item(),
                   "F": aux["F"], "G": aux["G"],
                   "realised_ratio": 1.0 / max(aux["bytes_per_chunk"], 1e-9),
                   "bytes_per_chunk": aux["bytes_per_chunk"],
                   "align_on_lead_byte": float(np.mean(on_lead)) if on_lead else 0.0,
                   "lead_byte_base_rate": float(np.mean(base)) if base else 0.0,
                   "align_lift": float(np.mean(lift)) if lift else 0.0,
                   "boundary_rate": float(bm.mean()),
                   "langs_in_batch": ",".join(LANGUAGES[i] for i in origin),
                   "grad_norm": float(gn), "lr": lr_at(step),
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            log("step", step, "ce", round(ce.item(), 4), "ratio_loss",
                round(aux["ratio_loss"].item(), 4), "F", round(aux["F"], 3),
                "G", round(aux["G"], 3), "bpc", round(aux["bytes_per_chunk"], 3),
                "align", round(row["align_on_lead_byte"], 3),
                "base", round(row["lead_byte_base_rate"], 3),
                "lift", round(row["align_lift"], 3))
        # eval has its own schedule -- nesting it under log_every silently
        # skipped every scheduled eval that was not also a log step.
        if (step + 1) % eval_every == 0 or step == CFG["max_steps"] - 1:
            row_eval = evaluate(model, paths, CFG["seq_len"])
            evals.append({"step": step, "per_language": row_eval})
            for lang, s in row_eval.items():
                lift = "n/a" if s["align_lift"] is None else f"{s['align_lift']:+.3f}"
                log(f"  eval {lang} bpb {s['bpb']:.3f} bpc "
                    f"{s['bytes_per_chunk']:.2f} lift {lift}")
        if time.time() - t0 > TIME_BUDGET_S:
            log("wall-clock guard hit at step", step)
            break

    json.dump({"config": CFG, "params": n_params, "cpu": os.cpu_count(),
               "torch": torch.__version__, "languages": LANGUAGES,
               "history": history, "final_eval": evals[-1] if evals else None,
               "evals": evals},
              open(os.path.join(OUT, "health.json"), "w"), indent=2)
    torch.save({"config": dict(cfg.__dict__), "state": model.state_dict()},
               os.path.join(OUT, "smoke.pt"))
    log("done", len(history), "logged rows")


if __name__ == "__main__":
    main()
