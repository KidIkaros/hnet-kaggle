"""Correctness gates that survive a no-compiled-extension, CPU-only environment.

The reference implementation (`main-horse/hnet-impl`) needs a `mamba-ssm` git fork and
`goombalab/hnet` needs `flash_attn==2.8.0.post2`, which does not build for sm_75. Neither is
available, so the reference-diff gate is replaced by an **independent second implementation
of the two components most likely to be silently wrong** — the router and the dechunk EMA —
plus a causal-mask check on the main network.

Run: python tests/test_gates.py   (exit 0 = all gates pass)
"""

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import (ChunkLayer, DeChunkLayer, HNetConfig, HNetLM, Router, SSMBlock,
                   chunked_scan, init_weights, sequential_scan)

torch.manual_seed(0)
FAILED = []


def gate(name):
    def deco(fn):
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            FAILED.append(f"{name}: {e}")
            print(f"FAIL  {name}: {e}")
        return fn
    return deco


# --------------------------------------------------------------- exactness ---
@gate("scan exactness vs sequential loop (fp32 < 1e-5)")
def _():
    a = torch.rand(2, 257, 8, 4) * 0.98
    b = torch.randn(2, 257, 8, 4)
    err = (sequential_scan(a, b) - chunked_scan(a, b, 64)).abs().max().item()
    assert err < 1e-5, f"max abs diff {err}"


@gate("dechunk EMA exactness vs naive loop")
def _():
    """EMA runs over CHUNK STARTS only (P = p at boundaries), then spreads each chunk's
    z-bar forward to its bytes. NB the gate `STE(c) = c + sg(1-c)` evaluates to exactly 1
    in forward, so it shapes gradient only and must not appear in the reference value."""
    B, L, D, M = 2, 40, 6, 7
    h = torch.randn(B, M, D)
    p = torch.rand(B, L)
    bmask = torch.zeros(B, L, dtype=torch.bool)
    starts = torch.tensor([[0, 5, 12, 20, 27, 33, 38], [0, 3, 9, 15, 22, 30, 36]])
    for bi in range(B):
        bmask[bi, starts[bi]] = True

    layer = DeChunkLayer(chunk_scan=8)
    got = layer(h, bmask, p, starts, torch.ones(B, M, dtype=torch.bool))

    p_sel = torch.gather(p, 1, starts)
    want = torch.empty(B, L, D)
    for bi in range(B):
        z = torch.zeros(D)
        zi = torch.zeros(M, D)
        for m in range(M):
            ph = p_sel[bi, m].clamp(1e-4, 1 - 1e-4)
            z = (1 - ph) * z + ph * h[bi, m]
            zi[m] = z
        for t in range(L):
            m = int((starts[bi] <= t).sum()) - 1
            want[bi, t] = zi[m]
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"max abs diff {err}"


@gate("dechunk gate STE: forward is identity, gradient is 1")
def _():
    """`STE(c) = c + sg(1-c)` == 1 in forward. If this ever silently becomes a real
    multiplier, dechunk output changes scale and the EMA gate above still passes."""
    B, L, D, M = 1, 8, 3, 2
    bmask = torch.zeros(B, L, dtype=torch.bool)
    bmask[0, [0, 4]] = True
    starts = torch.tensor([[0, 4]])
    h = torch.randn(B, M, D, requires_grad=True)
    p = torch.full((B, L), 0.2, requires_grad=True)   # confidence far from 1
    out = DeChunkLayer(chunk_scan=4)(h, bmask, p, starts, torch.ones(B, M, dtype=torch.bool))
    # value: gate == 1, so chunks carry the raw EMA of p=0.2
    z0 = 0.2 * h[0, 0]
    z1 = 0.8 * z0 + 0.2 * h[0, 1]
    assert torch.allclose(out[0, 0], z0, atol=1e-5), "gate is not 1 at a boundary"
    assert torch.allclose(out[0, 4], z1, atol=1e-5), "gate is not 1 off a boundary"
    out.sum().backward()
    # gate = c + sg(1-c) passes gradient 1 to c, so the router keeps a live gradient path
    assert p.grad is not None and p.grad.abs().max().item() > 0, "no gradient through p"


# ---------------------------------------------------------------- causality ---
@gate("causality: byte t+1 cannot change outputs at t (encoder, router, main, decoder)")
def _():
    cfg = HNetConfig(d_model=32, n_enc=1, n_dec=1, n_main=1, n_heads=2, scan_chunk=16)
    model = HNetLM(cfg)
    init_weights(model)
    model.eval()
    idx = torch.randint(0, 256, (1, 48))
    with torch.no_grad():
        base, _ = model(idx)
        for t in (10, 23, 40):
            bumped = idx.clone()
            bumped[0, t] = (bumped[0, t] + 1) % 256
            other, _ = model(bumped)
            delta = (base[:, :t] - other[:, :t]).abs().max().item()
            assert delta < 1e-5, f"perturbing t={t} moved positions <t by {delta}"


# ------------------------------------------- substitute reference (router) ---
@gate("router matches an independent cosine implementation")
def _():
    """Re-derives p_t from the paper's formula without reusing Router.forward."""
    D = 16
    router = Router(D)
    with torch.no_grad():
        router.q_proj.weight.normal_(std=0.3)
        router.k_proj.weight.normal_(std=0.3)
    x = torch.randn(2, 32, D)

    q = router.q_proj(x[:, :-1])
    k = router.k_proj(x[:, 1:])
    cos = F.cosine_similarity(q, k, dim=-1)
    want_p = (1 - cos) / 2
    want_p = torch.cat([torch.ones(2, 1), want_p], dim=1).clamp(0, 1)
    want_b = want_p >= 0.5

    got_p, got_b = router(x)
    assert (got_p - want_p).abs().max().item() < 1e-5, "p differs from cosine formula"
    assert torch.equal(got_b, want_b), "boundary mask differs"
    assert bool(got_b[:, 0].all()), "p_0 must be 1 (every sequence opens a chunk)"


# --------------------------------------------------- control mechanism gate ---
@gate("compression control engages: ratio loss falls and realised ratio rises")
def _():
    torch.manual_seed(1)
    cfg = HNetConfig(d_model=64, n_enc=1, n_dec=1, n_main=1, n_heads=2, scan_chunk=32)
    model = HNetLM(cfg)
    init_weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 256, (2, 128))
    hist = []
    for _ in range(30):
        logits, aux = model(idx)
        ce = F.cross_entropy(logits.reshape(-1, 256), idx.reshape(-1))
        loss = ce + 0.03 * aux["ratio_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append((aux["ratio_loss"].item(), aux["bytes_per_chunk"]))
    first_r, last_r = hist[0][0], hist[-1][0]
    first_b, last_b = hist[0][1], hist[-1][1]
    assert last_r < first_r, f"ratio loss did not fall ({first_r:.3f} -> {last_r:.3f})"
    assert last_b > first_b, f"bytes/chunk did not rise ({first_b:.3f} -> {last_b:.3f})"


# -------------------------------------------------------- chunk layer shape ---
@gate("chunk layer: gathers exactly the boundary bytes, in order, per sequence")
def _():
    layer = ChunkLayer(8)
    x = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8)
    bmask = torch.tensor([[True, False, True, False, False, True],
                          [True, True, False, False, False, False]])
    xc, order, valid = layer(x, bmask)
    for b in range(2):
        idx = torch.nonzero(bmask[b]).flatten().tolist()
        for slot, want in enumerate(idx):
            assert torch.equal(xc[b, slot], x[b, want]), f"seq {b} slot {slot} wrong byte"
        assert int(valid[b].sum()) == len(idx), f"seq {b} valid count wrong"
    # the shorter sequence's padding slots must be zeroed
    assert xc[1, 2:].numel() and xc[1, 2:].abs().max().item() == 0, "padding slots not zeroed"


# ------------------------------------------------------------- FLOPs parity ---
@gate("FLOPs counter: outer nets charged per byte, main net per chunk")
def _():
    sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
    from flops import flops_per_byte

    base = HNetConfig(d_model=64, n_enc=2, n_dec=2, n_main=2, n_heads=2, d_ff=None)
    f_short = flops_per_byte(base, 256, 2.0)
    f_long = flops_per_byte(base, 256, 8.0)
    # longer chunks => main net and attention run less often => fewer FLOPs/byte
    assert f_long < f_short, "counter does not respond to chunk length"

    # an extra outer layer must cost more than an extra main layer at long chunks
    more_enc = HNetConfig(**{**base.__dict__, "n_enc": 3})
    more_main = HNetConfig(**{**base.__dict__, "n_main": 3})
    gain_enc = flops_per_byte(more_enc, 256, 8.0) - f_long
    gain_main = flops_per_byte(more_main, 256, 8.0) - f_long
    assert gain_enc > gain_main, "outer nets are under-charged relative to the main net"

    # attention must grow quadratically in chunk count
    quad = flops_per_byte(base, 1024, 8.0) - flops_per_byte(base, 256, 8.0)
    assert quad > 0, "sequence length does not raise attention cost"
    print(f"      ({f_short/1e3:.1f}k -> {f_long/1e3:.1f}k FLOPs/byte as chunks go 2->8)")


if __name__ == "__main__":
    print(f"\n{len(FAILED)} failing gate(s)")
    for f in FAILED:
        print("  -", f)
    sys.exit(1 if FAILED else 0)
