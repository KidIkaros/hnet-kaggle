"""Where does the memory actually go? Local dry run, no Kaggle quota spent.

The rung probe measured 10.1 GB peak at d256/seq4096/batch4 with the SSM blocks
checkpointed, and we assumed the scan was responsible. This tests that assumption directly by
varying which parts are checkpointed, and by varying how compressed the router is.

`max_memory_allocated()` reports bytes allocated by the caching allocator, which is a property
of the graph, not of the device -- so a number measured on an 8 GB card is the number a T4
would try to allocate. Runs that OOM here would fit on a T4, so the variants are ordered
cheapest-memory-first and each is skipped if it cannot run.

Usage:
    python tools/memprofile.py --d-model 384 --seq-len 2048 --batch 4
"""

import argparse
import gc
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
import model as M


def log(*a):
    print(*a, flush=True)


def build(d_model, seq_len, batch, n_enc, n_main, n_dec, d_ff, d_state, ckpt_ssm, ckpt_main):
    cfg = M.HNetConfig(d_model=d_model, n_enc=n_enc, n_main=n_main, n_dec=n_dec, d_ff=d_ff,
                       d_state=d_state, grad_checkpoint=False)
    torch.manual_seed(0)
    model = M.HNetLM(cfg).cuda()
    M.init_weights(model)

    # checkpoint selectively, so the two contributions can be separated
    if ckpt_ssm:
        for blk in list(model.encoder) + list(model.decoder):
            blk.forward = _ckpt(blk.forward)
    if ckpt_main:
        for blk in model.main:
            blk.forward = _ckpt2(blk.forward)
    return model, cfg


def _ckpt(fn):
    def wrapped(*a):
        return torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)
    return wrapped


def _ckpt2(fn):
    def wrapped(x, mask):
        return torch.utils.checkpoint.checkpoint(fn, x, mask, use_reentrant=False)
    return wrapped


def run(model, seq_len, batch, steps=3):
    """Peak memory for a forward+backward+step at this geometry, or None if it OOMs."""
    x = torch.randint(0, 256, (batch, seq_len), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(steps):
            logits, aux = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    peak = torch.cuda.max_memory_allocated() / 1e6
    return peak, float(aux["bytes_per_chunk"])


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

    log(f"device {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB")
    log(f"geometry d{args.d_model} enc{args.n_enc}/main{args.n_main}/dec{args.n_dec} "
        f"ff{args.d_ff} N{args.d_state}, seq {args.seq_len}, batch {args.batch}")

    variants = [
        ("nothing checkpointed", False, False),
        ("SSM only (current)", True, False),
        ("main net only", False, True),
        ("both", True, True),
    ]
    results = []
    for name, ck_ssm, ck_main in variants:
        gc.collect()
        torch.cuda.empty_cache()
        model, cfg = build(args.d_model, args.seq_len, args.batch, args.n_enc, args.n_main,
                           args.n_dec, args.d_ff, args.d_state, ck_ssm, ck_main)
        nparams = sum(p.numel() for p in model.parameters())
        out = run(model, args.seq_len, args.batch)
        if out is None:
            log(f"  {name:>22}: OOM on this card")
            results.append((name, None, None))
        else:
            peak, chunk = out
            log(f"  {name:>22}: peak {peak:7.0f} MB   realised chunk {chunk:.2f}   "
                f"({nparams/1e6:.2f} M params)")
            results.append((name, peak, chunk))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # what the model itself costs, for reference
    best = min((r for r in results if r[1]), key=lambda r: r[1], default=None)
    if best:
        log(f"\nsmallest peak: {best[0]} at {best[1]:.0f} MB")


if __name__ == "__main__":
    main()
