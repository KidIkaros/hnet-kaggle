"""Gate: the alignment probe must be index-aligned and react correctly.

bmask[t] flags byte t as a chunk start, so the probe must compare byte t.
An off-by-one reads as a large negative lift on CJK and ~0 on ASCII, which
looks like a finding rather than a bug -- this catches it.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/src")
from model import HNetConfig, HNetLM, init_weights

torch.manual_seed(0)

# mixed-script batch: CJK (3 bytes/char) then ASCII
cjk = "字体边界应当与字符起点对齐。" * 8
asc = "boundaries should land on character starts. " * 2
raw = (cjk + asc).encode("utf-8")
L = len(raw)
batch = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))[None, :]

cfg = HNetConfig(d_model=64, n_enc=1, n_dec=1, n_main=1, scan_chunk=64)
model = HNetLM(cfg)
init_weights(model)
logits, aux = model(batch[:, :-1])
bm = aux["bmask"].numpy()[0]
lead = ((batch[:, :-1].numpy() & 0xC0) != 0x80)[0]

assert bm.shape == lead.shape, f"length mismatch {bm.shape} vs {lead.shape}"

lead_frac = lead.mean()
assert 0.2 < lead_frac < 0.6, f"test batch is not mixed-script: lead rate {lead_frac}"


def lift(mask):
    if not mask.any():
        return None
    return lead[mask].mean() - lead.mean()


l_oracle = lift(lead)

print(f"lead rate {lead_frac:.3f}")
print(f"all-boundary lift {lift(np.ones_like(lead)):+.3f} (must be ~0)")
print(f"lead-only lift    {l_oracle:+.3f} (must be >0.3)")
print(f"non-lead lift     {lift(~lead):+.3f} (must be <0)")

# The bug this replaces: bmask compared against lead[t+1], i.e. shifted the other
# way. On a mostly-multibyte batch that shift alone dragged the number negative.
shifted = np.concatenate([lead[1:], [True]])
print(f"lead-only lift under the old off-by-one: "
      f"{lead[shifted].mean() - lead.mean():+.3f}")

assert abs(lift(np.ones_like(lead))) < 1e-6
assert lift(~lead) < 0
assert l_oracle > 0.3
assert lead[shifted].mean() - lead.mean() < l_oracle, "off-by-one should under-read"
print("probe OK")
