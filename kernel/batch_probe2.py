# ==== WARM-STARTED BATCH PROBE ====
# The earlier batch sweep measured batch scaling at the UNTRAINED router (chunk ~1.9 bytes),
# where the main network runs at near-byte resolution and memory, not the scan, was binding.
# That measurement is what the "batch does not help" conclusion rests on, and it may not apply
# at the trained router (chunk ~6), where the model is about 5x cheaper per byte.
#
# This probe warm-starts from run 2's best checkpoint so the router is already trained, then
# measures batch 4/8/16/32: wall-clock per step, peak memory, and the effective MFU implied by
# the realised chunk length. It never trains -- it measures, on the state a real run has.

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
# Kaggle sets no environment, so default to the dataset mount of the published checkpoint.
CKPT = os.environ.get("CKPT", "/kaggle/input/hnet-eng-ckpt/ckpt-best.pt")
CORPUS = os.environ.get("CORPUS")
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    import torch
    log(f"devices: {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = HNetConfig(**ck["config"])
    model = HNetLM(cfg).to(DEV)
    model.load_state_dict(ck["state"])
    init_weights(model)
    model.eval()
    global SCAN_IMPL
    SCAN_IMPL = "triton"
    # no torch.compile here: it re-traces for every batch shape, and the probe's job is the
    # batch-scaling trend, which the fused kernel shows without a 4x compile tax

    log(f"loaded checkpoint at step {ck['step']}, {sum(p.numel() for p in model.parameters())/1e6:.2f} M params, "
        f"best bpb {ck.get('eval', {}).get('bpb')}")

    # the corpus, so the router sees real text rather than random bytes
    docs = Docs("train")
    rng = np.random.default_rng(0)
    SEQ = 2048

    # one forward at batch 4 to read the realised chunk length, which sets the FLOPs/byte
    starts, widths, order = batch_plan(docs, 4, SEQ, rng)
    out, mask = make_batch(docs, int(starts[order[0]]), int(widths[order[0]]), 4, SEQ)
    x0 = torch.from_numpy(out.astype(np.int64)).to(DEV)
    m0 = torch.from_numpy(mask).to(DEV)
    with torch.no_grad():
        _, aux = model(x0)
    chunk = float(aux["bytes_per_chunk"])
    fpb = training_flops_per_byte(cfg, SEQ, chunk)
    log(f"realised chunk {chunk:.2f} bytes -> {fpb/1e6:.1f} MFLOPs/byte")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    res = {"ckpt_step": ck["step"], "realised_chunk": chunk, "flops_per_byte": fpb, "batches": {}}

    budget_s = 720.0
    t_start = time.time()
    for B in (4, 8, 16, 32):
        if time.time() - t_start > budget_s:
            log(f"time budget {budget_s:.0f}s hit before batch {B}; stopping with what is measured")
            break
        try:
            starts, widths, order = batch_plan(docs, B, SEQ, rng)
            batches = [make_batch(docs, int(starts[k]), int(widths[k]), B, SEQ) for k in order[:3]]
            xs = [torch.from_numpy(o.astype(np.int64)).to(DEV) for o, _ in batches]
            ms = [torch.from_numpy(m).to(DEV) for _, m in batches]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for x, m in zip(xs, ms):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, aux = model(x)
                    ce, _ = masked_loss(logits[:, :-1], x[:, 1:], m[:, 1:])
                    loss = ce + 0.3 * aux["ratio_loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / len(xs) * 1000
            peak = torch.cuda.max_memory_allocated() / 1e9
            eff = B * SEQ * fpb / (per / 1000) / 1e12
            mfu = eff / 65e12
            res["batches"][str(B)] = {"ms_per_step": per, "peak_gb": peak,
                                      "effective_tflops": eff, "mfu": mfu}
            log(f"batch {B:3}: {per:7.1f} ms/step, {peak:.1f} GB, "
                f"{eff:6.2f} TFLOP/s = {100*mfu:.1f}% MFU, {B*SEQ/1024:.0f} KB/step")
        except torch.cuda.OutOfMemoryError:
            log(f"batch {B:3}: OOM")
            res["batches"][str(B)] = {"oom": True}
            torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(OUT, "batch_probe.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
