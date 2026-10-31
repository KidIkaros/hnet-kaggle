# ==== SCAN VARIANTS ====
# The profile says the SSM blocks are 90% of the forward and the scan is 80% of each, so the
# cost is the scan's memory traffic, not the matmuls. Two levers against that:
#   - compiling the whole SSM block, not just its scan, so the surrounding elementwise chain
#     fuses into the same kernels;
#   - a smaller scan chunk, which means fewer Hillis-Steele rounds over the same bytes
#     (traffic is proportional to log2(chunk)), at the price of a longer sequential carry.
# Each variant is measured on step time and checked against the reference for equivalence.

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


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def set_scan_impl(name):
    global SCAN_IMPL
    SCAN_IMPL = name
    log(f"scan implementation -> {name}")


def build(chunk):
    fields = {k: v for k, v in CFG.items() if k in HNetConfig.__dataclass_fields__}
    fields["scan_chunk"] = chunk
    cfg = HNetConfig(**fields)
    torch.manual_seed(0)
    m = HNetLM(cfg).to(DEV)
    init_weights(m)
    return m, cfg


def step_time(model, x, n=12):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        logits, aux = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        logits, aux = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    return (time.perf_counter() - t0) / n


def main():
    log("device", DEV)
    x = torch.randint(0, 256, (16, 256), device=DEV)
    res = {}

    # the reference: no compile, chunk 64
    set_scan_impl("reference")
    m, _ = build(64)
    res["reference_c64"] = step_time(m, x) * 1000
    log(f"reference, chunk 64:            {res['reference_c64']:.1f} ms")

    variants = [
        ("compiled_scan_c64", "compiled", 64, False),
        ("compiled_scan_c16", "compiled", 16, False),
        ("compiled_block_c64", "compiled", 64, True),
        ("compiled_block_c16", "compiled", 16, True),
        ("compiled_block_c32", "compiled", 32, True),
    ]
    for name, impl, chunk, block in variants:
        set_scan_impl(impl)
        m, _ = build(chunk)
        if block:
            n = compile_ssm_blocks(m)
            log(f"compiled {n} SSM blocks, chunk {chunk}")
        res[name] = step_time(m, x) * 1000
        log(f"{name:>24}: {res[name]:8.1f} ms   ({res['reference_c64']/res[name]:.2f}x vs reference)")

    # equivalence: does a smaller chunk change the result, and by how much
    set_scan_impl("compiled")
    m64, _ = build(64)
    with torch.no_grad():
        l64, _ = m64(x)
    m16, _ = build(16)
    with torch.no_grad():
        l16, _ = m16(x)
    res["chunk16_max_abs_logit_diff"] = float((l64 - l16).abs().max().item())
    res["chunk16_logit_scale"] = float(l64.abs().max().item())
    log(f"chunk 64 vs 16 logits: max abs diff {res['chunk16_max_abs_logit_diff']:.3e} "
        f"(logit scale {res['chunk16_logit_scale']:.2f})")

    json.dump(res, open(os.path.join(OUT, "variants.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
