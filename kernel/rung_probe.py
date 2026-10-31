# ==== RUNG PROBE ====
# Two things at once, both needed before the comparable rung can be launched.
#
# 1. FLOPs parity at the scale that matters. The parity gate on small shapes failed by
#    1.8-2.2x, and the reason is not the counter: at d_model 128 the GPU is kernel-launch
#    bound, not FLOP bound (265 GFLOP/s on a T4). The counter can only be expected to predict
#    wall-clock for shapes large enough to saturate the device, which is the scale the rung
#    is at. This measures there.
#
# 2. The rung's geometry. The scan's activations are B x L x d_model x d_state floats, and
#    the chunked scan keeps several rounds of them alive for the backward pass, so the
#    sequence length and the batch size are set by memory, not by preference. This measures
#    the peak memory of candidate geometries, with and without gradient checkpointing, and
#    reports which of them fit in the T4's 14.5 GB.
#
# The realised chunk length is used for the counter's chunk-level terms, because the router
# is untrained here and chunks nowhere near the target ratio.

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
TOLERANCE = 0.15
BUDGET = float(CFG.get("flops_budget", 3e16))

SHAPES = [
    # name, fields, seq_len, batch, checkpoint
    ("E 30M-ish, seq 4096, b4, ckpt",
     dict(d_model=256, n_enc=4, n_main=4, n_dec=4, d_ff=1024, d_state=16), 4096, 4, True),
    ("F 30M-ish, seq 4096, b4, no ckpt",
     dict(d_model=256, n_enc=4, n_main=4, n_dec=4, d_ff=1024, d_state=16), 4096, 4, False),
    ("G bigger, seq 4096, b2, ckpt",
     dict(d_model=256, n_enc=6, n_main=6, n_dec=6, d_ff=1024, d_state=16), 4096, 2, True),
    ("H wide, seq 2048, b4, ckpt",
     dict(d_model=384, n_enc=4, n_main=4, n_dec=4, d_ff=1536, d_state=16), 2048, 4, True),
    ("I seq 2048, b8, ckpt",
     dict(d_model=256, n_enc=4, n_main=4, n_dec=4, d_ff=1024, d_state=16), 2048, 8, True),
]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def measure(fields, seq_len, batch, ckpt, n=6):
    fields = dict(fields)
    fields["grad_checkpoint"] = ckpt
    cfg = HNetConfig(**fields)
    torch.manual_seed(0)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    global SCAN_IMPL
    SCAN_IMPL = "compiled"
    compile_ssm_blocks(model)
    x = torch.randint(0, 256, (batch, seq_len), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        logits, aux = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return aux

    try:
        for _ in range(2):
            aux = step()
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            aux = step()
        sync()
        t = (time.perf_counter() - t0) / n
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    peak = torch.cuda.max_memory_allocated() / 1e6
    nparams = sum(p.numel() for p in model.parameters())
    realised = float(aux["bytes_per_chunk"])
    return {"cfg": cfg, "t": t, "peak": peak, "params": nparams, "chunk": realised,
            "seq_len": seq_len, "batch": batch}


def main():
    log("device", DEV, f"budget {BUDGET:.3e} training FLOPs")
    rows = []
    for name, fields, seq_len, batch, ckpt in SHAPES:
        torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
        r = measure(fields, seq_len, batch, ckpt)
        if r is None:
            log(f"{name}: OUT OF MEMORY")
            rows.append({"name": name, "oom": True})
            torch.cuda.empty_cache() if DEV == "cuda" else None
            continue
        cfg = r["cfg"]
        fpb = training_flops_per_byte(cfg, seq_len, r["chunk"])
        bytes_per_step = batch * seq_len
        charged = fpb * bytes_per_step
        achieved = charged / r["t"]
        steps, per_step = steps_for_budget(cfg, seq_len, batch, r["chunk"], BUDGET)
        rows.append({
            "name": name, "params": r["params"], "seq_len": seq_len, "batch": batch,
            "checkpoint": ckpt, "step_s": r["t"], "peak_memory_mb": r["peak"],
            "realised_bytes_per_chunk": r["chunk"],
            "training_flops_per_byte": fpb,
            "training_flops_per_step": charged,
            "achieved_flops_per_s": achieved,
            "budget_steps": steps,
            "budget_hours": steps * r["t"] / 3600,
        })
        log(f"{name}: {r['t']*1000:7.1f} ms/step, peak {r['peak']:.0f} MB, "
            f"{r['params']/1e6:.2f} M, chunk {r['chunk']:.2f}, "
            f"{achieved/1e9:.0f} GFLOP/s, budget -> {steps} steps = {steps*r['t']/3600:.1f} h")
        torch.cuda.empty_cache() if DEV == "cuda" else None

    ok_rows = [r for r in rows if not r.get("oom")]
    rates = [r["achieved_flops_per_s"] for r in ok_rows]
    spread = max(rates) / min(rates) if rates else 0.0
    log("")
    log(f"achieved rate spread across {len(ok_rows)} shapes: {spread:.3f} "
        f"(tolerance {TOLERANCE})")
    for r in ok_rows:
        log(f"  {r['name']:>38}: {r['achieved_flops_per_s']/1e9:6.0f} GFLOP/s  "
            f"peak {r['peak_memory_mb']:6.0f} MB  {r['budget_hours']:5.1f} h for the budget")
    json.dump({"rows": rows, "spread": spread, "tolerance": TOLERANCE,
               "budget": BUDGET, "pass": spread <= 1 + TOLERANCE},
              open(os.path.join(OUT, "rung_probe.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
