"""Where does the step time actually go? Timing decomposition, local, no quota spent.

We know the SSM blocks dominate the forward (90% at d128) and the scan dominates each block.
This measures it at the rung geometry and separates the pieces, so the fix targets the part
that is actually expensive rather than the part we assume is.

Variants (each mathematically wrong on purpose -- this is a timing experiment, not a model):
  full        the real step
  no_scan     the scan replaced by an identity, same tensor shapes
  no_main     the chunk-level transformer skipped
  no_ssm      encoder and decoder blocks skipped
  no_router   boundaries fixed to every 6th byte instead of computed

Usage: python tools/timebreak.py --d-model 384 --seq-len 2048 --batch 4
"""

import argparse
import gc
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
import model as M


def log(*a):
    print(*a, flush=True)


def build(d_model, seq_len, n_enc, n_main, n_dec, d_ff, d_state, ckpt=True):
    cfg = M.HNetConfig(d_model=d_model, n_enc=n_enc, n_main=n_main, n_dec=n_dec, d_ff=d_ff,
                       d_state=d_state)
    torch.manual_seed(0)
    m = M.HNetLM(cfg).cuda()
    M.init_weights(m)
    if ckpt:
        def wrap(fn):
            def inner(*a):
                return torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)
            return inner
        for blk in list(m.encoder) + list(m.decoder):
            blk.forward = wrap(blk.forward)
    return m, cfg


def timed(m, seq_len, batch, n=8, warm=3):
    x = torch.randint(0, 256, (batch, seq_len), device="cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
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
        return (time.perf_counter() - t0) / n * 1000, float(aux["bytes_per_chunk"])
    except torch.cuda.OutOfMemoryError:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-enc", type=int, default=4)
    ap.add_argument("--n-main", type=int, default=6)
    ap.add_argument("--n-dec", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--d-state", type=int, default=16)
    args = ap.parse_args()

    log(f"device {torch.cuda.get_device_name(0)}")
    log(f"d{args.d_model} enc{args.n_enc}/main{args.n_main}/dec{args.n_dec} "
        f"seq {args.seq_len} batch {args.batch}")

    orig_scan = M.scan
    orig_router_forward = M.Router.forward
    rows = {}

    def run(label, patch=None, unpatch=None):
        gc.collect(); torch.cuda.empty_cache()
        m, cfg = build(args.d_model, args.seq_len, args.n_enc, args.n_main, args.n_dec,
                       args.d_ff, args.d_state)
        if patch:
            patch()
        ms, chunk = timed(m, args.seq_len, args.batch)
        if unpatch:
            unpatch()
        rows[label] = ms
        log(f"  {label:>12}: {'OOM' if ms is None else f'{ms:8.0f} ms'}   chunk {chunk}")
        del m
        gc.collect(); torch.cuda.empty_cache()

    run("full")

    # the scan, replaced by an identity of the same shape
    run("no_scan", patch=lambda: setattr(M, "scan", lambda a, b, c: b),
        unpatch=lambda: setattr(M, "scan", orig_scan))

    # the chunk-level transformer
    def no_main():
        for blk in ():
            pass
    gc.collect(); torch.cuda.empty_cache()
    m, cfg = build(args.d_model, args.seq_len, args.n_enc, args.n_main, args.n_dec,
                   args.d_ff, args.d_state)
    m.main = torch.nn.ModuleList([])
    ms, chunk = timed(m, args.seq_len, args.batch)
    rows["no_main"] = ms
    log(f"  {'no_main':>12}: {'OOM' if ms is None else f'{ms:8.0f} ms'}")
    del m
    gc.collect(); torch.cuda.empty_cache()

    # the outer SSM blocks
    m, cfg = build(args.d_model, args.seq_len, args.n_enc, args.n_main, args.n_dec,
                   args.d_ff, args.d_state)
    m.encoder = torch.nn.ModuleList([])
    m.decoder = torch.nn.ModuleList([])
    ms, chunk = timed(m, args.seq_len, args.batch)
    rows["no_ssm"] = ms
    log(f"  {'no_ssm':>12}: {'OOM' if ms is None else f'{ms:8.0f} ms'}")
    del m

    full = rows.get("full")
    if full:
        log("\nshare of the step:")
        for k in ("no_scan", "no_main", "no_ssm"):
            v = rows.get(k)
            if v:
                log(f"  removing {k:>8} saves {100*(1 - v/full):5.1f}% "
                    f"({full - v:6.0f} ms of {full:.0f})")


if __name__ == "__main__":
    main()
