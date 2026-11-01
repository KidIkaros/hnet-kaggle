# ==== ENGLISH RUNG DRIVER ====
# Trains the own H-Net on the English corpus for the comparable rung.
#
# Three things differ from the multilingual driver (kernel/q1.py), and all three come from
# the corpus being documents rather than a byte stream:
#   - batches are formed out of whole documents, never by slicing one across a boundary;
#   - the loss is masked, so padded positions do not contribute to it or to bpb;
#   - the run is specified by a FLOPs allowance, and the step count is derived from it.
#
# The documents are sorted by byte length within each split, so a batch of consecutive
# documents has near-uniform length and the padding waste stays small. The order of the
# batches is shuffled, so the model does not see the corpus shortest-document-first.

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/eng.json")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ==== DATA ====
CORPUS = os.environ.get("CORPUS")
if not CORPUS:
    for root in ("/kaggle/input/datasets/mauricew/hnet-english-bytes",
                 "/kaggle/input/hnet-english-bytes",
                 "/kaggle/input/hnet-english-bytes/corpus"):
        if os.path.isdir(root):
            CORPUS = root
            break
assert CORPUS, "the English corpus dataset is not mounted"
SEQ_LEN = int(CFG["seq_len"])
BATCH = int(CFG["batch"])
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class Docs:
    """One split: the bytes, and each document's offset and length."""

    def __init__(self, split):
        # the dataset is published flat: the Kaggle CLI skips directories unless asked to
        # zip them, so the split files sit at the root rather than under a split directory
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
    """Which documents go in which batch, and in what order.

    Documents are length-sorted, so consecutive runs have near-uniform length. The runs are
    then shuffled, so the model does not see the corpus ordered by length. A document longer
    than the sequence is truncated to its head; a short one is padded, and the padding is
    masked out of the loss.
    """
    n = len(docs)
    n_batches = n // batch
    starts = np.arange(n_batches) * batch
    # the longest document in each run sets the batch's width
    widths = np.array([int(docs.lengths[s: s + batch].max()) for s in starts])
    order = rng.permutation(n_batches)
    return starts, widths, order


def make_batch(docs, start, width, batch, seq_len):
    """(bytes, mask): whole documents, padded to the batch's width, capped at seq_len."""
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
        mask[r, :n - 1] = 1.0            # the loss predicts byte t+1 from byte t
    return out, mask


def lr_at(step, max_steps, lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def masked_loss(logits, targets, mask):
    """Cross entropy over the real bytes only."""
    ls = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    denom = m.sum().clamp(min=1.0)
    return (ls * m).sum() / denom, m.sum().item()


def evaluate(model, docs, seq_len, batch, n_batches=40, seed=0):
    """Held-out bpb over whole documents."""
    rng = np.random.default_rng(seed)
    starts, widths, order = batch_plan(docs, batch, seq_len, rng)
    model.eval()
    tot_nll, tot_bytes = 0.0, 0
    with torch.no_grad():
        for k in order[:n_batches]:
            start = int(starts[k])
            width = int(widths[k])
            out, mask = make_batch(docs, start, width, batch, seq_len)
            x = torch.from_numpy(out.astype(np.int64)).to(DEV)
            m = torch.from_numpy(mask).to(DEV)
            logits, aux = model(x)
            # logits[t] predicts byte t+1, so the target is x[:, 1:] and the mask shifts with
            # it. Comparing logits[t] against x[t] instead measures the model's guess for a
            # byte it is being shown -- which scored 13.2 bpb, worse than uniform, on a run
            # whose training bpb had reached 1.44.
            ls = F.cross_entropy(logits[:, :-1].reshape(-1, 256),
                                 x[:, 1:].reshape(-1), reduction="none")
            mm = m[:, 1:].reshape(-1)
            tot_nll += float((ls * mm).sum().item())
            tot_bytes += float(mm.sum().item())
    model.train()
    bpb = (tot_nll / max(tot_bytes, 1)) / math.log(2)
    return {"bpb": bpb, "bytes": tot_bytes}


def save_state(model, opt, scaler, step, history, evals, best, flops, path):
    torch.save({"config": dict(model.cfg.__dict__), "seed": CFG["seeds"][0],
                "state": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
                "history": history, "evals": evals, "best": best, "flops": flops,
                "rng": np.random.get_state()}, path)


def load_state(model, opt, scaler, path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["state"])
    opt.load_state_dict(ck["opt"])
    if ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    try:
        np.random.set_state(ck["rng"])
    except Exception:
        pass
    return ck


def self_check(model, train, seq_len, batch):
    """Prove the evaluation path measures what training measures, before spending hours.

    The first rung trained correctly and was scored by an evaluate() that compared logits[t]
    with byte t instead of t+1. Nothing in the loss curve showed it; the run looked broken
    only in a metric nobody could trust. This compares the two paths on the same batch and
    refuses to start if they disagree.
    """
    model.eval()
    with torch.no_grad():
        ev = evaluate(model, train, seq_len, batch, n_batches=1)
        rng = np.random.default_rng(0)
        starts, widths, order = batch_plan(train, batch, seq_len, rng)
        out, mask = make_batch(train, int(starts[order[0]]), int(widths[order[0]]),
                               batch, seq_len)
        x = torch.from_numpy(out.astype(np.int64)).to(DEV)
        m = torch.from_numpy(mask).to(DEV)
        logits, _ = model(x)
        ce, _ = masked_loss(logits[:, :-1], x[:, 1:], m[:, 1:])
    model.train()
    train_bpb = float(ce) / math.log(2)
    if abs(train_bpb - ev["bpb"]) > 0.05:
        raise RuntimeError(
            f"evaluation disagrees with the training loss on the same batch: "
            f"evaluate() says {ev['bpb']:.3f} bpb, the training path says {train_bpb:.3f}. "
            f"Refusing to train against a metric that is measuring something else.")
    log(f"self-check: evaluation and training agree on the same batch "
        f"({ev['bpb']:.3f} vs {train_bpb:.3f} bpb)")


class TrainStep(nn.Module):
    """The loss, computed inside forward so DataParallel can gather it.

    DataParallel gathers a module's outputs across replicas and cannot gather the dict of
    floats our model returns (F, G, bytes_per_chunk are Python floats). Returning a tuple of
    one-element tensors instead means the gather produces a vector per field, one entry per
    GPU, which the caller averages.

    The wrapper is used for training only. Evaluation and checkpointing keep the raw model, so
    the state dict has no `module.` prefix and an existing checkpoint still loads.
    """

    def __init__(self, inner, alpha):
        super().__init__()
        self.inner = inner
        self.alpha = alpha

    def forward(self, x, mask):
        logits, aux = self.inner(x)
        ce, _ = masked_loss(logits[:, :-1], x[:, 1:], mask[:, 1:])
        rl = aux["ratio_loss"]
        loss = ce + self.alpha * rl
        one = lambda v: torch.as_tensor(float(v), device=x.device).reshape(1)
        return (loss, ce.detach(), rl.detach(), one(aux["F"]), one(aux["G"]),
                one(aux["bytes_per_chunk"]))


def main():
    log("device", DEV)
    log("corpus", CORPUS)
    log("config", json.dumps(CFG))
    train = Docs("train")
    val = Docs("val")
    log(f"train {len(train)} docs {len(train.data)/1e6:.1f} MB, "
        f"val {len(val)} docs {len(val.data)/1e6:.1f} MB")

    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    torch.manual_seed(CFG["seeds"][0])
    np.random.seed(CFG["seeds"][0])
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    global SCAN_IMPL
    want = CFG.get("scan_impl") or ("compiled" if CFG.get("compile_blocks", True) else "reference")
    # Fail loudly rather than falling back: the reference scan is 3.5x slower, and a silent
    # fallback would burn the whole GPU session instead of stopping in the first second.
    if want == "triton":
        # Three ways the fused path can be quietly unavailable. Each would turn an 8 h run
        # into a 30 h one, so each refuses rather than falling back.
        if scan_triton is None:
            raise RuntimeError("config asks for the fused scan but scan_triton did not import")
        if DEV != "cuda":
            raise RuntimeError("config asks for the fused scan but there is no CUDA device")
        if cfg.d_model % 32:
            raise RuntimeError(f"config asks for the fused scan but d_model {cfg.d_model} is "
                               f"not a multiple of the kernel's block size")
    SCAN_IMPL = want
    if want == "compiled":
        compile_ssm_blocks(model)
    log(f"scan implementation: {SCAN_IMPL}")
    ndev = torch.cuda.device_count() if DEV == "cuda" else 1
    log(f"devices visible: {ndev}"
        + "".join(f", cuda:{i} {torch.cuda.get_device_name(i)}" for i in range(ndev)))
    step_model = TrainStep(model, cfg.alpha)
    if CFG.get("data_parallel", True) and ndev > 1 and BATCH % ndev != 0:
        raise SystemExit(f"batch {BATCH} is not divisible by {ndev} devices: DataParallel "
                         f"would split it unevenly and the averaged loss would weight the "
                         f"shards unequally. Use a batch that divides, or set "
                         f"data_parallel=false.")
    if CFG.get("data_parallel", True) and ndev > 1:
        step_model = nn.DataParallel(step_model)
        log(f"training on {ndev} GPUs via DataParallel; evaluation and checkpoints use the "
            f"raw model so the state dict keeps its keys")
    else:
        log("training on a single device")
    mo = float(CFG.get("lr_mult_outer", 1.0))
    mm = float(CFG.get("lr_mult_main", 1.0))
    groups = param_groups(model, mo, mm)
    opt = torch.optim.AdamW(groups, lr=CFG["lr"], weight_decay=CFG["wd"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(CFG.get("amp") == "fp16" and DEV == "cuda"))
    nparams = sum(p.numel() for p in model.parameters())
    log(f"model {nparams/1e6:.2f} M params, {cfg.n_enc}+{cfg.n_dec} SSM blocks, "
        f"{cfg.n_main} main blocks")

    # the budget becomes the schedule
    budget = float(CFG["flops_budget"])
    assumed_chunk = float(cfg.target_ratio)
    max_steps, per_step = steps_for_budget(cfg, SEQ_LEN, BATCH, assumed_chunk, budget)
    flops = flops_report(cfg, SEQ_LEN, BATCH, assumed_chunk)
    flops.update({"budget_flops": budget, "derived_steps": max_steps,
                  "training_flops_per_step": per_step,
                  "budget_total_flops": max_steps * per_step,
                  "chunk_len_assumed": assumed_chunk})
    log(f"budget {budget:.4e} training FLOPs -> {max_steps} steps "
        f"({per_step:.4e} FLOPs/step, {flops['training_flops_per_byte']:.4e} FLOPs/byte)")

    self_check(model, train, SEQ_LEN, BATCH)

    rng = np.random.default_rng(CFG["seeds"][0])
    starts, widths, order = batch_plan(train, BATCH, SEQ_LEN, rng)
    log(f"{len(order)} batches of {BATCH} documents")

    history, evals = [], []
    best = None
    resume_from = os.path.join(OUT, "ckpt-resume.pt")
    start_step = 0
    if CFG.get("resume", True) and os.path.exists(resume_from):
        ck = load_state(model, opt, scaler, resume_from)
        start_step = int(ck["step"])
        history = ck.get("history", [])
        evals = ck.get("evals", [])
        best = ck.get("best")
        log(f"resumed from {resume_from} at step {start_step}")
    t0 = time.time()
    wall_guard = float(CFG.get("wall_guard_s", 36000))
    done = 0
    for step in range(start_step, max_steps):
        k = order[step % len(order)]
        start = int(starts[k])
        width = int(widths[k])
        out, mask = make_batch(train, start, width, BATCH, SEQ_LEN)
        x = torch.from_numpy(out.astype(np.int64)).to(DEV)
        m = torch.from_numpy(mask).to(DEV)

        lr = lr_at(step, max_steps, CFG["lr"], CFG["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=(CFG.get("amp") == "fp16" and DEV == "cuda")):
            out = step_model(x, m)
        # out is a tuple of scalars on one device, or of per-GPU vectors on several; mean()
        # is correct for both and is what keeps the reported numbers comparable
        loss = out[0].mean()
        ce, rl = out[1].mean(), out[2].mean()
        aux = {"F": out[3].mean().item(), "G": out[4].mean().item(),
               "bytes_per_chunk": out[5].mean().item()}
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        scaler.step(opt)
        scaler.update()
        done = step + 1

        if step % CFG["log_every"] == 0 or step == max_steps - 1:
            el = time.time() - t0
            rec = {"step": step, "ce": float(ce.item()), "bpb": float(ce.item()) / math.log(2),
                   "ratio_loss": float(rl.item()), "F": aux["F"], "G": aux["G"],
                   "bytes_per_chunk": aux["bytes_per_chunk"], "lr": lr,
                   "bytes": int(m.sum().item()), "elapsed_s": el,
                   "eta_h": (el / max(done, 1)) * (max_steps - done) / 3600}
            history.append(rec)
            log(f"step {step:6} bpb {rec['bpb']:.4f} ratio {rec['ratio_loss']:.4f} "
                f"F {rec['F']:.3f} G {rec['G']:.3f} bytes/chunk {rec['bytes_per_chunk']:.2f} "
                f"lr {lr:.2e} elapsed {el/60:.1f}m eta {rec['eta_h']:.1f}h")
        if step % CFG["eval_every"] == 0 or step == max_steps - 1:
            ev = evaluate(model, val, SEQ_LEN, BATCH,
                          n_batches=int(CFG.get("eval_batches", 40)))
            evals.append({"step": step, **ev})
            log(f"  eval step {step}: held-out bpb {ev['bpb']:.4f} over {ev['bytes']} bytes")
            # 8.0 bits is uniform over 256 symbols, and an untrained model sits right at it
            # (7.94 in the first rung), so step 0 is exempt: the canary is for a run that has
            # had a chance to learn and is still anti-predicting, which is a broken metric or
            # a diverged run, not a slow start.
            if step > 0 and ev["bpb"] > 8.5:
                save_state(model, opt, scaler, step + 1, history, evals, best, flops,
                           resume_from)
                raise RuntimeError(
                    f"held-out bpb {ev['bpb']:.3f} is far above uniform (8.0): stopping instead "
                    f"of spending more quota on a run that cannot be right")
            if best is None or ev["bpb"] < best["bpb"]:
                best = {"step": step, **ev}
                torch.save({"config": dict(cfg.__dict__), "seed": CFG["seeds"][0],
                            "state": model.state_dict(), "step": step, "eval": ev},
                           os.path.join(OUT, "ckpt-best.pt"))
        if step % CFG.get("save_every", 500) == 0 and step > start_step:
            save_state(model, opt, scaler, step + 1, history, evals, best, flops,
                       resume_from)
        if time.time() - t0 > wall_guard:
            log("wall guard hit")
            save_state(model, opt, scaler, step + 1, history, evals, best, flops,
                       resume_from)
            break

    # what the run actually cost, next to what it was allowed to cost
    realised_chunk = history[-1]["bytes_per_chunk"] if history else None
    if realised_chunk:
        flops["realised_bytes_per_chunk"] = realised_chunk
        flops["realised_training_flops_per_byte"] = training_flops_per_byte(
            cfg, SEQ_LEN, realised_chunk)
        flops["realised_total_flops"] = done * BATCH * SEQ_LEN * \
            flops["realised_training_flops_per_byte"]
        flops["budget_used_fraction"] = flops["realised_total_flops"] / budget

    torch.save({"config": dict(cfg.__dict__), "seed": CFG["seeds"][0],
                "state": model.state_dict(), "step": done},
               os.path.join(OUT, "ckpt-final.pt"))
    save_state(model, opt, scaler, done, history, evals, best, flops, resume_from)
    json.dump({"config": CFG, "steps": done, "stop": "max_steps" if done == max_steps else "wall_guard",
               "history": history, "evals": evals, "best": best, "flops": flops,
               "params": nparams, "elapsed_s": time.time() - t0},
              open(os.path.join(OUT, "eng.json"), "w"), indent=2)
    log("done", done, "steps", f"{(time.time()-t0)/3600:.2f} h",
        "best held-out bpb", best["bpb"] if best else None)


if __name__ == "__main__":
    main()
