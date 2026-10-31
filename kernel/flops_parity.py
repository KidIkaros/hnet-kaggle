# ==== FLOPs PARITY (GPU) ====
# Ticket 02: the analytic FLOPs-per-byte figure must agree with measured wall-clock within
# the existing parity tolerance (15%, docs/PLAN.md G4), across at least three model shapes.
#
# The existing gate only checked that the counter responds in the right DIRECTION to chunk
# length and layer counts. It never compared the counter with a clock. This does.
#
# The counter charges matmuls, attention and the scan's own arithmetic. It does not charge
# the embedding lookup, the router's nonlinearity, the chunk layer's sort and gather, the
# RMSNorms, the loss, or the optimiser -- so the implied efficiency is well below the T4's
# peak, and that is fine. What has to hold is that the efficiency is the SAME for every
# shape: if the counter mis-charges a term, one shape's efficiency will differ from the
# others. The gate is therefore max(efficiency)/min(efficiency) <= 1.15.

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


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


SHAPES = [
    # name, model fields, seq_len, batch -- deliberately different in every dimension.
    # The scan's activations are B x L x d_model x d_state floats, and the chunked scan keeps
    # several rounds of them alive for the backward pass, so the sequence length is what
    # decides whether a shape fits in the T4's memory at all.
    ("A sweep shape, byte-heavy",
     dict(d_model=128, n_enc=2, n_main=2, n_dec=2, d_ff=512, d_state=16), 256, 16),
    ("B larger model",
     dict(d_model=256, n_enc=4, n_main=4, n_dec=4, d_ff=1024, d_state=16), 256, 8),
    ("C chunk-heavy, many main layers",
     dict(d_model=128, n_enc=2, n_main=6, n_dec=2, d_ff=512, d_state=16), 512, 8),
    ("D wide state",
     dict(d_model=192, n_enc=3, n_main=3, n_dec=3, d_ff=768, d_state=32), 256, 8),
]


def measure(fields, seq_len, batch, chunk_len=6.0, n=10):
    cfg = HNetConfig(**fields)
    torch.manual_seed(0)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    global SCAN_IMPL
    SCAN_IMPL = "compiled"
    compile_ssm_blocks(model)
    x = torch.randint(0, 256, (batch, seq_len), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def fwd_bwd():
        logits, aux = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()

    def full():
        fwd_bwd()
        opt.step()

    for _ in range(3):
        fwd_bwd()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fwd_bwd()
    sync()
    t_fb = (time.perf_counter() - t0) / n

    for _ in range(3):
        full()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        full()
    sync()
    t_full = (time.perf_counter() - t0) / n

    nparams = sum(p.numel() for p in model.parameters())
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEV == "cuda" else 0.0
    # The realised chunk length, not the target. The counter's chunk-level terms are
    # parameterised by it, and an untrained router chunks nowhere near the target ratio, so
    # charging the target would make the counter's chunk-level terms wrong by that factor.
    with torch.no_grad():
        _, aux = model(x)
    realised_chunk = float(aux["bytes_per_chunk"])
    return cfg, t_fb, t_full, nparams, peak, realised_chunk


def main():
    log("device", DEV)
    rows = []
    for name, fields, seq_len, batch in SHAPES:
        torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
        cfg, t_fb, t_full, nparams, peak, realised_chunk = measure(fields, seq_len, batch)
        bytes_per_step = batch * seq_len
        fpb = training_flops_per_byte(cfg, seq_len, realised_chunk)
        # the counter's training FLOPs per byte, against the FLOPs per byte the clock saw
        measured_fpb = bytes_per_step / t_fb                      # bytes per second
        efficiency = fpb / measured_fpb                            # analytic / achieved
        rows.append({
            "name": name, "params": nparams, "seq_len": seq_len, "batch": batch,
            "fwd_bwd_s": t_fb, "full_step_s": t_full,
            "optimiser_overhead": (t_full - t_fb) / t_fb,
            "analytic_training_flops_per_byte": fpb,
            "measured_bytes_per_s": measured_fpb,
            # the FLOPs per second the counter says the step is doing, from the clock's side
            "efficiency": efficiency,
            "achieved_flops_per_s": fpb * bytes_per_step / t_fb,
            "peak_memory_mb": peak,
            "realised_bytes_per_chunk": realised_chunk,
        })
        log(f"{name}: {t_fb*1000:7.1f} ms fwd+bwd, {t_full*1000:7.1f} ms full step, "
            f"efficiency {efficiency:.4f}, {nparams/1e6:.2f} M params, peak {peak:.0f} MB, "
            f"chunk {realised_chunk:.2f}")
        torch.cuda.empty_cache() if DEV == "cuda" else None

    effs = [r["efficiency"] for r in rows]
    spread = max(effs) / min(effs)
    log("")
    log(f"efficiency spread across {len(rows)} shapes: {spread:.4f}  (tolerance {TOLERANCE})")
    for r in rows:
        log(f"  {r['name']:>32}: {r['efficiency']:.4f}  "
            f"({r['achieved_flops_per_s']/1e9:.1f} GFLOP/s achieved)")
    ok = spread <= 1.0 + TOLERANCE
    log(f"PARITY {'PASS' if ok else 'FAIL'}")
    json.dump({"rows": rows, "spread": spread, "tolerance": TOLERANCE, "pass": ok},
              open(os.path.join(OUT, "parity.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
