# ==== SPEED GATE (GPU) ====
# The scan optimisation has to be gated on the device it will run on: inductor generates
# different kernels for CUDA than for CPU, so bit-exactness measured on CPU does not carry.
# This kernel runs, on the T4:
#   1. the bit-exactness gate (reference vs optimised scan, production shapes)
#   2. a full-model step-time benchmark: reference vs optimised
#   3. the fp16 autocast loss match, which is a PRECISION change and is gated separately
#      from the algorithm change -- the paper prescribes amp fp16 with fp32 master weights
#      on a T4, where bf16 does not exist.

import json
import math
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


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def set_scan_impl(name):
    """The kernel inlines the model, so the scan's global is right here."""
    global SCAN_IMPL
    SCAN_IMPL = name
    log(f"scan implementation -> {name}")


def make(B=16, L=256, E=128, N=16, seed=0, device=DEV):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, L, E, N, generator=g) * 0.5 + 0.5
    b = torch.randn(B, L, E, N, generator=g)
    return a.to(device), b.to(device)


def timeit(fn, a, b, chunk, n=30):
    fn(a, b, chunk)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(a, b, chunk)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    log("device", DEV, "torch", torch.__version__)
    res = {}
    a, b = make()
    chunk = 64

    ref = chunked_scan(a, b, chunk)
    fast = chunked_scan_fast(a, b, chunk)
    same_fast = torch.equal(ref, fast)
    log(f"reference vs fast: bit-exact {same_fast}  max abs diff "
        f"{(ref - fast).abs().max().item():.3e}")
    res["fast_bit_exact"] = bool(same_fast)

    comp = chunked_scan_compiled(a, b, chunk)
    same_comp = torch.equal(ref, comp)
    log(f"reference vs compiled: bit-exact {same_comp}  max abs diff "
        f"{(ref - comp).abs().max().item():.3e}")
    res["compiled_bit_exact"] = bool(same_comp)

    t_ref = timeit(chunked_scan, a, b, chunk)
    t_fast = timeit(chunked_scan_fast, a, b, chunk)
    t_comp = timeit(chunked_scan_compiled, a, b, chunk)
    log(f"scan forward: reference {t_ref*1000:.2f} ms  fast {t_fast*1000:.2f} ms "
        f"({t_ref/t_fast:.2f}x)  compiled {t_comp*1000:.2f} ms ({t_ref/t_comp:.2f}x)")
    res["scan_ms"] = {"reference": t_ref*1000, "fast": t_fast*1000, "compiled": t_comp*1000}

    # full-model step time, which is what the rung actually pays
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    cfg = HNetConfig(**fields)
    torch.manual_seed(0)
    model = HNetLM(cfg).to(DEV)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (16, 256), device=DEV)

    def step_time(n=15):
        for _ in range(3):
            logits, aux = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, aux = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    set_scan_impl("reference")
    t_model_ref = step_time()
    set_scan_impl("fast")
    t_model_fast = step_time()
    set_scan_impl("compiled")
    t_model_comp = step_time()
    log(f"full step: reference {t_model_ref*1000:.1f} ms  fast {t_model_fast*1000:.1f} ms "
        f"({t_model_ref/t_model_fast:.2f}x)  compiled {t_model_comp*1000:.1f} ms "
        f"({t_model_ref/t_model_comp:.2f}x)")
    res["step_ms"] = {"reference": t_model_ref*1000, "fast": t_model_fast*1000,
                      "compiled": t_model_comp*1000}

    # model-level equality in fp32: same weights, same input, reference vs the optimised scan
    torch.manual_seed(0)
    mm = HNetLM(cfg).to(DEV)
    init_weights(mm)
    xx = torch.randint(0, 256, (16, 256), device=DEV)
    set_scan_impl("reference")
    with torch.no_grad():
        l_ref, _ = mm(xx)
    set_scan_impl("fast")
    with torch.no_grad():
        l_fast, _ = mm(xx)
    model_same = torch.equal(l_ref, l_fast)
    log(f"model logits, reference vs fast scan (fp32): identical {model_same}  max abs diff "
        f"{(l_ref - l_fast).abs().max().item():.3e}")
    res["model_bit_exact_fp32"] = bool(model_same)
    set_scan_impl("compiled")

    # the combined configuration: fp16 autocast with fp32 master weights, over the compiled scan
    torch.manual_seed(0)
    mc = HNetLM(cfg).to(DEV)
    init_weights(mc)
    oc = torch.optim.AdamW(mc.parameters(), lr=1e-3)
    sc = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))

    def step_time_amp(n=15):
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                logits, aux = mc(xx)
                loss = F.cross_entropy(logits.reshape(-1, 256), xx.reshape(-1))
            oc.zero_grad(set_to_none=True)
            sc.scale(loss).backward()
            sc.step(oc)
            sc.update()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                logits, aux = mc(xx)
                loss = F.cross_entropy(logits.reshape(-1, 256), xx.reshape(-1))
            oc.zero_grad(set_to_none=True)
            sc.scale(loss).backward()
            sc.step(oc)
            sc.update()
        if DEV == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    t_amp = step_time_amp()
    log(f"full step, fp16 + compiled scan: {t_amp*1000:.1f} ms  "
        f"({t_model_ref/t_amp:.2f}x vs the fp32 reference)")
    res["step_ms_fp16_compiled"] = t_amp * 1000
    res["combined_speedup"] = float(t_model_ref / t_amp)

    # fp16 autocast with fp32 master weights: a PRECISION change, gated on a loss match
    set_scan_impl("compiled")
    torch.manual_seed(0)
    m32 = HNetLM(cfg).to(DEV)
    init_weights(m32)
    o32 = torch.optim.AdamW(m32.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    xb = torch.randint(0, 256, (16, 256), device=DEV)
    l32, l16 = [], []
    for step in range(60):
        logits, aux = m32(xb)
        loss32 = F.cross_entropy(logits.reshape(-1, 256), xb.reshape(-1))
        o32.zero_grad(set_to_none=True)
        loss32.backward()
        o32.step()
        l32.append(loss32.item())
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            logits, aux = m32(xb)
            loss16 = F.cross_entropy(logits.reshape(-1, 256), xb.reshape(-1))
        scaler.scale(loss16).backward()
        scaler.step(o32)
        scaler.update()
        l16.append(loss16.item())
    d = abs(np.mean(l32[-10:]) - np.mean(l16[-10:]))
    log(f"fp32 loss {np.mean(l32[-10:]):.4f}  fp16-autocast loss {np.mean(l16[-10:]):.4f}  "
        f"diff {d:.4f}")
    res["fp16_loss"] = {"fp32": float(np.mean(l32[-10:])), "fp16": float(np.mean(l16[-10:])),
                        "abs_diff": float(d)}
    res["fp16_accept"] = bool(d < 0.02)

    if DEV == "cuda":
        res["mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
        log(f"peak memory {res['mem_peak_mb']:.0f} MB")

    json.dump(res, open(os.path.join(OUT, "speed_gate.json"), "w"), indent=2)
    log("done", json.dumps({k: v for k, v in res.items() if k != "scan_ms"}))


if __name__ == "__main__":
    main()
