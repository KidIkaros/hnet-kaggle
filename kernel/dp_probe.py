# ==== DUAL-GPU PROBE ====
# Kaggle's accelerator is "GPU T4 x2": two 16 GB T4s. Every run so far used cuda:0 only, so
# half the hardware has been idle. Two things need answering before spending more quota:
#
#   1. how many devices are actually visible (the premise);
#   2. what a second GPU buys in step time -- and whether the quota meter charges for it.
#
# The second is answered by the session's own wall-clock against the quota delta, which the
# caller reads outside this kernel.
#
# Measures, at the rung geometry: one GPU, then two via DataParallel, same batch.

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/cfg.json")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

GEOM = dict(d_model=384, n_enc=4, n_main=6, n_dec=4, d_ff=1536, d_state=16)
SEQ, BATCH = 2048, 8


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build():
    cfg = HNetConfig(**GEOM, grad_checkpoint=True)
    torch.manual_seed(0)
    m = HNetLM(cfg).to(DEV)
    init_weights(m)
    global SCAN_IMPL
    SCAN_IMPL = "triton"
    return m, cfg


class LogitsOnly(nn.Module):
    """DataParallel gathers the module's outputs, so the forward must return tensors only.

    Our model returns (logits, aux) where aux carries floats like bytes_per_chunk, and
    gathering a dict of scalars is what killed the first run of this probe.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        logits, _ = self.inner(x)
        return logits


def measure(model, n=6, warm=2):
    x = torch.randint(0, 256, (BATCH, SEQ), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    try:
        for _ in range(warm):
            logits = model(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits = model(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000
    except torch.cuda.OutOfMemoryError:
        return None


def main():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(f"devices visible: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        log(f"  cuda:{i} {p.name} {p.total_memory/1e9:.1f} GB")
    log(f"geometry {GEOM}, seq {SEQ}, batch {BATCH}")
    res = {"devices": n, "names": [], "batch": BATCH, "seq": SEQ}
    for i in range(n):
        res["names"].append(torch.cuda.get_device_name(i))

    m1, cfg = build()
    t1 = measure(LogitsOnly(m1))
    res["one_gpu_ms"] = t1
    log(f"one GPU:  {'OOM' if t1 is None else f'{t1:.0f} ms/step'}")
    del m1
    torch.cuda.empty_cache()

    if n >= 2:
        m2, cfg = build()
        m2 = nn.DataParallel(LogitsOnly(m2))
        t2 = measure(m2)
        res["two_gpu_ms"] = t2
        log(f"two GPUs: {'OOM' if t2 is None else f'{t2:.0f} ms/step'}")
        if t1 and t2:
            res["speedup"] = t1 / t2
            log(f"speedup {t1/t2:.2f}x -- if the quota is billed per session, that is free")
        del m2
        torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(OUT, "dp.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
