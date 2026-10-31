"""Is the router learning at all, and is the compression healthy?

Two free diagnostics on checkpoints we already have:

1. **Router drift.** The router is identity-initialised and excluded from reinit, so at step 0
   the boundary probability is a pure function of the *raw byte embeddings*: p_t depends on
   cos(x_t, x_{t-1}) with no learned transform. If W_q and W_k barely move from identity, the
   chunk boundaries are decided by which byte embedding follows which -- which is exactly the
   byte-phase behaviour we measured in three modalities, and it would mean the router is
   contributing almost nothing.

2. **Compression health.** The ratio loss should collapse, spike above its initial value, then
   settle to the target within a few hundred steps. Runs that never enter that regime have bad
   loss. F is the selected fraction, G the mean boundary probability, and the healthy target is
   F = G = 1/N.
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")

RUNS = [
    ("0.84M utf-8  seed0", "/home/ikaaros/.hermes/cache/scratch/replres/ckpt-s0-n4.pt",
     "/home/ikaaros/.hermes/cache/scratch/replres/seed-s0-n4.json"),
    ("2.43M utf-8  seed0", "/home/ikaaros/.hermes/cache/scratch/q1scale/ckpt-s0.pt",
     "/home/ikaaros/.hermes/cache/scratch/q1scale/seed-s0.json"),
]


def drift(state, key):
    w = state[f"router.{key}.weight"].float()
    d = w.shape[0]
    eye = torch.eye(d)
    return (torch.linalg.matrix_norm(w - eye) / torch.linalg.matrix_norm(eye)).item()


def main():
    for name, ck, js in RUNS:
        if not os.path.exists(ck):
            print(f"{name}: missing {ck}")
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        st = blob["state"]
        N = blob["config"]["target_ratio"]
        dq, dk = drift(st, "q_proj"), drift(st, "k_proj")
        # how far is the router from "identity"? and are q and k drifting apart from each other?
        q = st["router.q_proj.weight"].float()
        k = st["router.k_proj.weight"].float()
        d = q.shape[0]
        qk = (torch.linalg.matrix_norm(q - k) / torch.linalg.matrix_norm(q)).item()
        print(f"\n=== {name}   (target ratio N={N}, so healthy F=G=1/{N}={1/N:.3f})")
        print(f"  ||W_q - I|| / ||I|| : {dq:.4f}")
        print(f"  ||W_k - I|| / ||I|| : {dk:.4f}")
        print(f"  ||W_q - W_k|| / ||W_q|| : {qk:.4f}   (0 = q and k are the same map)")
        if os.path.exists(js):
            d2 = json.load(open(js))
            hist = d2.get("history", [])
            if hist:
                print("  step   ce     bpb    F      G      L_ratio  bpc")
                for h in hist[:: max(1, len(hist) // 12)]:
                    print(f"  {h['step']:>5} {h['ce']:.4f} {h.get('bpb',0):.4f} "
                          f"{h.get('F',float('nan')):.3f} {h.get('G',float('nan')):.3f} "
                          f"{h.get('ratio_loss',float('nan')):.3f}  {h.get('bytes_per_chunk',0):.2f}")
                last = hist[-1]
                print(f"  final: F={last.get('F'):.3f} G={last.get('G'):.3f} "
                      f"ratio_loss={last.get('ratio_loss'):.3f} "
                      f"bpc={last.get('bytes_per_chunk'):.2f} "
                      f"(target bpc={N:.1f})")


if __name__ == "__main__":
    main()
