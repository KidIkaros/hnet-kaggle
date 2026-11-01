"""Does the scan's cost scale with launches (log2 chunk) or with traffic (chunk count)?

The scan is 87.5% of the step. It launches ~6 kernels per round, with log2(scan_chunk) rounds
per chunk and L/scan_chunk chunks. A small chunk means many chunks (more sequential carries,
same rounds); a large chunk means more rounds over the whole sequence but far fewer launches.

    chunk    chunks  rounds  launches ~ L/chunk * 6 * rounds
    16       128     4       3072
    64       32      6       1152
    256      8       8       384
    2048     1       11      66

If launches dominate, time should fall steeply as the chunk grows. If traffic dominates, it
should rise, since each round touches the whole sequence.

Usage: python tools/chunksweep.py --d-model 384 --seq-len 2048 --batch 2
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


def log(*a):
    print(*a, flush=True)


def measure(cfg, seq_len, batch, n=8, warm=3):
    torch.manual_seed(0)
    m = M.HNetLM(cfg).cuda()
    M.init_weights(m)

    def wrap(fn):
        def inner(*a):
            return torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)
        return inner
    for blk in list(m.encoder) + list(m.decoder):
        blk.forward = wrap(blk.forward)

    x = torch.randint(0, 256, (batch, seq_len), device="cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    try:
        for _ in range(warm):
            logits, aux = m(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            logits, aux = m(x)
            F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
    except torch.cuda.OutOfMemoryError:
        del m; gc.collect(); torch.cuda.empty_cache()
        return None
    del m; gc.collect(); torch.cuda.empty_cache()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-enc", type=int, default=4)
    ap.add_argument("--n-main", type=int, default=6)
    ap.add_argument("--n-dec", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--chunks", type=str, default="16,64,256,1024,2048")
    args = ap.parse_args()

    log(f"device {torch.cuda.get_device_name(0)}")
    log(f"d{args.d_model} seq {args.seq_len} batch {args.batch}")
    log(f"{'chunk':>6} {'chunks':>7} {'rounds':>7} {'launches':>9} {'ms/step':>9} {'vs best':>8}")
    rows = []
    for c in [int(x) for x in args.chunks.split(",")]:
        cfg = M.HNetConfig(d_model=args.d_model, n_enc=args.n_enc, n_main=args.n_main,
                           n_dec=args.n_dec, d_ff=args.d_ff, d_state=args.d_state,
                           scan_chunk=c)
        ms = measure(cfg, args.seq_len, args.batch)
        nch = max(1, args.seq_len // c)
        rounds = max(1, c.bit_length() - 1)
        launches = nch * 6 * rounds
        if ms is None:
            log(f"{c:>6} {nch:>7} {rounds:>7} {launches:>9} {'OOM':>9}")
            continue
        rows.append((c, ms))
        log(f"{c:>6} {nch:>7} {rounds:>7} {launches:>9} {ms:9.0f}")
    if rows:
        best = min(rows, key=lambda r: r[1])
        log(f"\nbest chunk {best[0]} at {best[1]:.0f} ms/step")
        base = next((ms for c, ms in rows if c == 64), None)
        if base:
            log(f"chunk 64 (what we run) is {base/best[1]:.2f}x the best")


if __name__ == "__main__":
    main()
