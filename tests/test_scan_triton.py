"""Equivalence tests for the fused scan. Written before the kernel exists.

The fused scan must reproduce the reference recurrence exactly enough that a training run is
the same run:

    h_t = a_t * h_{t-1} + b_t,   a_t = exp(delta_t * A),   b_t = delta_t * Bc_t * x_t
    y_t = sum_n h_{t,n} * Cc_{t,n} + D * x_t

The reference path is `model.scan` on explicitly materialised `a` and `b`, with autograd
providing the gradients. The fused path computes the same thing without ever materialising
(a, b) as (B, L, E, N) tensors, and supplies its own backward. So there are two things to
check, and the second is the one that breaks quietly:

  1. forward agreement with the reference;
  2. gradient agreement with autograd's answer for the same function, for every input.

Run: python tests/test_scan_triton.py
"""

import sys

import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
import model as M

FAILED = []


def case(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def reference(x, delta, A, Bc, Cc, Dskip, chunk=64):
    """The current model's own arithmetic, with autograd tracking every input."""
    a = torch.exp(delta.unsqueeze(-1) * A)                      # (B,L,E,N)
    b = delta.unsqueeze(-1) * Bc.unsqueeze(2) * x.unsqueeze(-1)  # (B,L,E,N)
    h = M.chunked_scan(a, b, chunk)
    return (h * Cc.unsqueeze(2)).sum(-1) + Dskip * x


def main():
    if not torch.cuda.is_available():
        print("no CUDA device; skipping")
        return 0
    import scan_triton

    torch.manual_seed(0)
    B, L, E, N = 2, 256, 64, 16
    dev = "cuda"

    def make():
        x = torch.randn(B, L, E, device=dev, requires_grad=True)
        delta = (torch.rand(B, L, E, device=dev) * 0.5 + 0.01).requires_grad_()
        A = (-torch.rand(E, N, device=dev) - 0.1).requires_grad_()
        Bc = torch.randn(B, L, N, device=dev, requires_grad=True)
        Cc = torch.randn(B, L, N, device=dev, requires_grad=True)
        D = torch.randn(E, device=dev, requires_grad=True)
        return x, delta, A, Bc, Cc, D

    # ---- forward
    args_ref = make()
    args_new = tuple(a.detach().clone().requires_grad_() for a in args_ref)
    y_ref = reference(*args_ref)
    y_new = scan_triton.ssm_scan(*args_new)
    err = (y_ref - y_new).abs().max().item()
    rel = err / y_ref.abs().max().item()
    case("forward matches the reference", rel < 2e-4, f"max abs {err:.3e}, rel {rel:.2e}")

    # ---- backward: gradients must match autograd through the reference
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_new.backward(g)
    names = ["x", "delta", "A", "Bc", "Cc", "D"]
    for nm, r, n in zip(names, args_ref, args_new):
        if r.grad is None or n.grad is None:
            case(f"grad {nm} present", False, f"{r.grad is None=} {n.grad is None=}")
            continue
        e = (r.grad - n.grad).abs().max().item()
        scale = r.grad.abs().max().item() + 1e-9
        case(f"grad {nm} matches autograd", e / scale < 5e-3,
             f"max abs {e:.3e}, rel {e/scale:.2e}")

    # ---- long sequence, the shape the rung actually uses
    B2, L2 = 1, 2048
    x = torch.randn(B2, L2, E, device=dev, requires_grad=True)
    delta = (torch.rand(B2, L2, E, device=dev) * 0.5 + 0.01).requires_grad_()
    A = (-torch.rand(E, N, device=dev) - 0.1).requires_grad_()
    Bc = torch.randn(B2, L2, N, device=dev, requires_grad=True)
    Cc = torch.randn(B2, L2, N, device=dev, requires_grad=True)
    D = torch.randn(E, device=dev, requires_grad=True)
    with torch.no_grad():
        y_ref2 = reference(x, delta, A, Bc, Cc, D)
        y_new2 = scan_triton.ssm_scan(x, delta, A, Bc, Cc, D)
    rel2 = (y_ref2 - y_new2).abs().max().item() / y_ref2.abs().max().item()
    case("forward matches at L=2048", rel2 < 2e-4, f"rel {rel2:.2e}")

    print(f"\n{len(FAILED)} failing test(s)")
    for f in FAILED:
        print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
