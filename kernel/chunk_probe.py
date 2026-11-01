# ==== SCAN CHUNK PROBE ====
# One question, two phases, in a single GPU session so the session startup is paid once.
#
# The scan is 87.5% of the training step. Its cost could scale with the number of chunk
# carries (L/chunk sequential loops) or with the number of rounds (log2(chunk) passes over the
# whole sequence). Those pull in opposite directions, so the best chunk is an empirical
# question, not a matter of taste:
#
#     chunk   carries   rounds   launches ~ 6 * carries * rounds   traffic ~ rounds * L
#     16      128       4        3072                              4x
#     64      32        6        1152                              6x
#     256     8         8        384                               8x
#     2048    1         11       66                                11x
#
# Phase 1 measures every chunk without compilation, which is fast and shows the algorithmic
# scaling. Phase 2 compiles only the winner, because compiling eight blocks per config would
# cost more than the whole experiment.
#
# Fail-fast: if a chunk is already 2x better than the baseline, the remaining smaller chunks
# are skipped, since the ordering is monotone in that direction.

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
SEQ, BATCH = 2048, 2
CHUNKS = [64, 256, 512, 2048]
BUDGET = 1.5e16


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build(scan_chunk, compile_blocks=False):
    cfg = HNetConfig(**GEOM, scan_chunk=scan_chunk, grad_checkpoint=True)
    torch.manual_seed(0)
    m = HNetLM(cfg).to(DEV)
    init_weights(m)
    global SCAN_IMPL
    if compile_blocks:
        SCAN_IMPL = "compiled"
        compile_ssm_blocks(m)
    else:
        SCAN_IMPL = "reference"
    return m, cfg


def measure(m, n=3, warm=1):
    x = torch.randint(0, 256, (BATCH, SEQ), device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warm):
            logits, aux = m(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, aux = m(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    peak = torch.cuda.max_memory_allocated() / 1e9
    return ms, peak, float(aux["bytes_per_chunk"])


def main():
    log("device", DEV, "torch", torch.__version__)
    log(f"geometry {GEOM}, seq {SEQ}, batch {BATCH}")
    rows = []

    # phase 1: no compilation, all chunks
    base_ms = None
    for c in CHUNKS:
        m, cfg = build(c, compile_blocks=False)
        out = measure(m)
        del m
        torch.cuda.empty_cache()
        if out is None:
            log(f"chunk {c:>5}: OOM")
            continue
        ms, peak, chunk = out
        if base_ms is None:
            base_ms = ms
        rows.append({"chunk": c, "ms": ms, "peak_gb": peak, "realised": chunk,
                     "compiled": False})
        log(f"chunk {c:>5}: {ms:8.0f} ms/step, peak {peak:5.2f} GB, "
            f"realised {chunk:.2f} B/chunk, {base_ms/ms:.2f}x vs chunk {CHUNKS[0]}")
        if ms < base_ms / 2.0 and c >= 512:
            log(f"  already {base_ms/ms:.1f}x better than the baseline; skipping larger chunks")
            break

    if not rows:
        json.dump({"rows": [], "error": "all configs OOMed"}, open(os.path.join(OUT, "chunk.json"), "w"))
        return

    best = min(rows, key=lambda r: r["ms"])
    log(f"\nbest without compilation: chunk {best['chunk']} at {best['ms']:.0f} ms/step")

    # phase 2: compile the winner only
    m, cfg = build(best["chunk"], compile_blocks=True)
    out = measure(m, n=4, warm=2)
    del m
    torch.cuda.empty_cache()
    if out is not None:
        ms, peak, chunk = out
        rows.append({"chunk": best["chunk"], "ms": ms, "peak_gb": peak, "realised": chunk,
                     "compiled": True})
        log(f"chunk {best['chunk']} compiled: {ms:8.0f} ms/step, peak {peak:5.2f} GB, "
            f"{best['ms']/ms:.2f}x vs its uncompiled form")

    # what it means for the run
    final = min([r for r in rows], key=lambda r: r["ms"])
    fpb = training_flops_per_byte(cfg, SEQ, final["realised"])
    steps = BUDGET / (fpb * BATCH * SEQ)
    hours = steps * final["ms"] / 1000 / 3600
    achieved = fpb * BATCH * SEQ / (final["ms"] / 1000)
    log(f"\nbest overall: chunk {final['chunk']}, compiled={final['compiled']}, "
        f"{final['ms']:.0f} ms/step")
    log(f"  {achieved/1e9:.0f} GFLOP/s = {100*achieved/65e12:.2f}% MFU")
    log(f"  {BUDGET:.1e} FLOPs -> {steps:.0f} steps -> {hours:.1f} h")
    json.dump({"rows": rows, "best": final, "steps": steps, "hours": hours,
               "achieved_flops": achieved, "mfu": achieved / 65e12},
              open(os.path.join(OUT, "chunk.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
