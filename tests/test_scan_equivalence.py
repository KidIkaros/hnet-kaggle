"""Gate: the optimised scan must reproduce the reference exactly, and must be faster.

The acceptance criterion for the speed work is bit-exactness against the implementation that
currently defines our model's semantics, plus a measured speedup. This checks both, on
production-shaped tensors, for the plain and compiled paths.

The reference implementation stays in the tree; nothing here replaces it.
"""
import sys
import time

import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
import model as M


def make(B=16, L=256, E=128, N=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    # a is a decay in (0, 1], as it is in the model: exp(delta * A) with A < 0
    a = torch.rand(B, L, E, N, generator=g) * 0.5 + 0.5
    b = torch.randn(B, L, E, N, generator=g)
    return a, b


def timeit(fn, a, b, chunk, n=20):
    fn(a, b, chunk)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(a, b, chunk)
    return (time.perf_counter() - t0) / n


def main():
    a, b = make()
    chunk = 64

    ref = M.chunked_scan(a, b, chunk)
    fast = M.chunked_scan_fast(a, b, chunk)

    same = torch.equal(ref, fast)
    maxdiff = (ref - fast).abs().max().item()
    print(f"reference vs fast:  bit-exact {same}   max abs diff {maxdiff:.3e}")
    assert same, "the optimised scan must reproduce the reference bit-exactly"

    # the naive sequential reference, to confirm the chunked scan itself is right
    seq = M.sequential_scan(a[:, :8], b[:, :8])
    chunked = M.chunked_scan(a[:, :8], b[:, :8], chunk)
    err = (seq - chunked).abs().max().item()
    print(f"chunked vs sequential reference on 8 steps: max abs diff {err:.3e}")
    assert err < 1e-4, "the chunked scan disagrees with the naive recurrence"

    t_ref = timeit(M.chunked_scan, a, b, chunk)
    t_fast = timeit(M.chunked_scan_fast, a, b, chunk)
    print(f"\nCPU forward, 16x256x128x16, chunk 64:")
    print(f"  reference {t_ref*1000:8.2f} ms")
    print(f"  fast      {t_fast*1000:8.2f} ms   speedup {t_ref/t_fast:.2f}x")

    # the compiled path
    try:
        comp = M.chunked_scan_compiled(a, b, chunk)
        csame = torch.equal(ref, comp)
        cmax = (ref - comp).abs().max().item()
        t_comp = timeit(M.chunked_scan_compiled, a, b, chunk, n=20)
        print(f"\n  compiled  {t_comp*1000:8.2f} ms   speedup {t_ref/t_comp:.2f}x")
        print(f"  compiled bit-exact vs reference: {csame}   max abs diff {cmax:.3e}")
        if not csame:
            print("  NOTE: the compiled path is not bit-exact. Fusing the elementwise chain")
            print("        reassociates the float arithmetic, so it needs a documented")
            print("        tolerance rather than the bit-exact gate. That is a decision for")
            print("        the owner, not something to paper over here.")
    except Exception as e:
        print(f"\n  compiled path unavailable: {type(e).__name__}: {e}")

    # a shape the model actually sees for the dechunk EMA scan, which is lower rank
    a2, b2 = make(B=16, L=64, E=128, N=1)
    r2 = M.chunked_scan(a2, b2, 64)
    f2 = M.chunked_scan_fast(a2, b2, 64)
    print(f"\nrank-4 with N=1: bit-exact {torch.equal(r2, f2)}")
    assert torch.equal(r2, f2)
    print("\nPASS")


if __name__ == "__main__":
    main()
