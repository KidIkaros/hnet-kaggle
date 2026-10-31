# ==== PROFILER ====
# Where does the step time actually go?
#
# The step is 222 ms for a 0.84 M model, which is ~1e11 FLOPS against the ~1e13 a T4 can do.
# Compiling the scan took the step to 116 ms and fp16 on top bought nothing, so the cost is
# not in the matmuls and not in the scan. This measures each module's forward with CUDA
# events, so the next optimisation targets whatever is actually slow.

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CFG = json.load(open(os.environ.get("CONFIG", "/tmp/cfg.json")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

TOTALS = defaultdict(float)
COUNTS = defaultdict(int)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def timed(module, name):
    """Wrap a module's forward with a synchronised timer."""
    orig = module.forward

    def wrapped(*a, **kw):
        if DEV == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            TOTALS[name] += time.perf_counter() - t0
            COUNTS[name] += 1
            return out
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        TOTALS[name] += time.perf_counter() - t0
        COUNTS[name] += 1
        return out

    module.forward = wrapped


def main():
    log("device", DEV)
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    torch.manual_seed(0)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    x = torch.randint(0, 256, (16, 256), device=DEV)

    for name, mod in [("emb", model.emb), ("router", model.router),
                      ("chunk_layer", model.chunk_layer),
                      ("dechunk_layer", model.dechunk_layer), ("head", model.head),
                      ("norm_f", model.norm_f), ("main_norm", model.main_norm)]:
        timed(mod, name)
    for i, blk in enumerate(model.encoder):
        timed(blk, f"encoder[{i}]")
    for i, blk in enumerate(model.decoder):
        timed(blk, f"decoder[{i}]")
    for i, blk in enumerate(model.main):
        timed(blk, f"main[{i}]")

    N = 20
    with torch.no_grad():
        for _ in range(3):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(N):
            model(x)
        sync()
        fwd = (time.perf_counter() - t0) / N

    TOTALS.clear()
    COUNTS.clear()
    for _ in range(N):
        logits, aux = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
    sync()

    per = {k: TOTALS[k] / COUNTS[k] * 1000 for k in TOTALS}
    res = {"forward_ms": fwd * 1000, "per_module_ms": per}
    log("\nper-module forward time (ms), and share of the forward:")
    tot = sum(per.values())
    for k in sorted(per, key=lambda k: -per[k]):
        log(f"  {k:>16} {per[k]:8.2f} ms   {100*per[k]/max(tot,1e-9):5.1f}%")
    log(f"  {'SUM OF PARTS':>16} {tot:8.2f} ms")
    log(f"  {'MEASURED FORWARD':>16} {fwd*1000:8.2f} ms")
    json.dump(res, open(os.path.join(OUT, "profile.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
