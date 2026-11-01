"""How does throughput scale with batch size? Local sweep, before spending quota.

We run at 0.34% of the T4's peak because the batch is tiny. The question is where the knee
is: if MFU saturates at batch 8, that is the target and no kernel rewrite is needed; if it
keeps climbing to batch 32, then memory (and therefore the scan) is the binding constraint and
the rewrite is justified.

Reports, per batch size: ms/step, peak memory, achieved GFLOP/s, MFU. Skips sizes that do not
fit rather than failing, so the sweep is safe on an 8 GB card.

Usage:
    python tools/batchsweep.py --d-model 384 --seq-len 2048
"""

import argparse
import gc
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
import model as M
from flops import training_flops_per_byte

T4_PEAK = 65e12


def log(*a):
    print(*a, flush=True)


def ckpt_blocks(model, main_too=True):
    def wrap(fn):
        def inner(*a):
            return torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)
        return inner
    for blk in list(model.encoder) + list(model.decoder):
        blk.forward = wrap(blk.forward)
    if main_too:
        for blk in model.main:
            blk.forward = wrap(blk.forward)


def measure(cfg, seq_len, batch, n=10, warm=3):
    torch.manual_seed(0)
    model = M.HNetLM(cfg).cuda()
    M.init_weights(model)
    ckpt_blocks(model)
    M.SCAN_IMPL = "reference"
    x = torch.randint(0, 256, (batch, seq_len), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warm):
            logits, aux = model(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, aux = model(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
    except torch.cuda.OutOfMemoryError:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return None
    peak = torch.cuda.max_memory_allocated() / 1e9
    chunk = float(aux["bytes_per_chunk"])
    fpb = training_flops_per_byte(cfg, seq_len, chunk)
    achieved = fpb * batch * seq_len / (ms / 1000)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ms, peak, achieved, achieved / T4_PEAK, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-enc", type=int, default=4)
    ap.add_argument("--n-main", type=int, default=6)
    ap.add_argument("--n-dec", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--batches", type=str, default="1,2,4,8,16")
    args = ap.parse_args()

    cfg = M.HNetConfig(d_model=args.d_model, n_enc=args.n_enc, n_main=args.n_main,
                       n_dec=args.n_dec, d_ff=args.d_ff, d_state=args.d_state,
                       grad_checkpoint=False)
    name = torch.cuda.get_device_name(0)
    log(f"device {name}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log(f"geometry d{args.d_model} enc{args.n_enc}/main{args.n_main}/dec{args.n_dec} "
        f"ff{args.d_ff} N{args.d_state}, seq {args.seq_len}")
    log(f"{'batch':>6} {'ms/step':>9} {'peak GB':>8} {'GFLOP/s':>9} {'MFU':>7} {'chunk':>6} {'MB/step':>8}")
    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        out = measure(cfg, args.seq_len, b)
        if out is None:
            log(f"{b:>6} {'OOM':>9}")
            continue
        ms, peak, achieved, mfu, chunk = out
        log(f"{b:>6} {ms:9.0f} {peak:8.2f} {achieved/1e9:9.0f} {mfu*100:6.2f}% "
            f"{chunk:6.2f} {b*args.seq_len/1e6:8.2f}")
        rows.append({"batch": b, "ms": ms, "peak_gb": peak, "gflops": achieved,
                     "mfu": mfu, "chunk": chunk})

    # what the numbers mean for a fixed budget
    if rows:
        budget = 1.5e16
        log(f"\nfor a {budget:.1e} FLOPs budget:")
        for r in rows:
            fpb = training_flops_per_byte(cfg, args.seq_len, r["chunk"])
            steps = budget / (fpb * r["batch"] * args.seq_len)
            log(f"  batch {r['batch']:>2}: {steps:7.0f} steps x {r['ms']/1000:6.2f} s "
                f"= {steps*r['ms']/1000/3600:5.1f} h, {steps*r['batch']*args.seq_len/1e6:7.1f} MB seen")


if __name__ == "__main__":
    main()
