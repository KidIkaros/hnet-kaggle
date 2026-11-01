#!/usr/bin/env bash
# Fact sheet. Run this BEFORE any plan that spends quota.
#
# The rule it exists to enforce, from /grilling: finding facts is the agent's job, never the
# user's, and nothing may be left silently assumed. Every plan tonight rested on facts that
# were assumed rather than checked -- one GPU instead of two, quota billed per GPU instead of
# per session, a batch-limited bottleneck that was not batch-limited, an evaluation that was
# off by one. Each was a one-command lookup.
#
#   tools/facts.sh
#
# Costs nothing: no GPU work, no quota. Prints what is true, and marks anything it cannot
# verify from here rather than guessing.
set -uo pipefail
cd "$(dirname "$0")/.."
KAGGLE=/home/ikaaros/.venvs/kg/bin/kaggle
GPU_PY=/home/ikaaros/.venvs/hnetgpu/bin/python

echo "== devices (local machine) =="
if [ -x "$GPU_PY" ]; then
  "$GPU_PY" - <<'PY' 2>/dev/null || echo "  could not query"
import torch
n = torch.cuda.device_count()
print(f"  local devices: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"    cuda:{i} {p.name} {p.total_memory/1e9:.1f} GB")
print(f"  torch {torch.__version__}, cuda {torch.version.cuda}")
try:
    import triton
    print(f"  triton {triton.__version__}")
except Exception as e:
    print(f"  triton unavailable ({type(e).__name__})")
PY
else
  echo "  no CUDA interpreter at $GPU_PY"
fi

echo
echo "== devices (Kaggle, last measured) =="
DP=/home/ikaaros/.hermes/cache/scratch/dpprobe2/dp.json
if [ -f "$DP" ]; then
  python3 - "$DP" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  devices visible in a Kaggle session: {d['devices']} ({', '.join(d['names'])})")
if "speedup" in d:
    print(f"  one GPU {d['one_gpu_ms']:.0f} ms/step, two {d['two_gpu_ms']:.0f} ms/step "
          f"-> {d['speedup']:.2f}x")
PY
else
  echo "  UNVERIFIED -- run kernel_dp and pull dp.json"
fi

echo
echo "== quota =="
"$KAGGLE" quota 2>/dev/null | sed -n '1,3p' | sed 's/^/  /'

echo
echo "== corpus =="
C=/home/ikaaros/.hermes/cache/scratch/eng_corpus
if [ -f "$C/meta.json" ]; then
  python3 - "$C" <<'PY'
import json, os, sys
c = sys.argv[1]
m = json.load(open(os.path.join(c, "meta.json")))
tr = os.path.join(c, "train", "train.bin")
va = os.path.join(c, "val", "val.bin")
print(f"  train {os.path.getsize(tr)/1e9:.2f} GB, val {os.path.getsize(va)/1e6:.1f} MB")
print(f"  {m['train_docs']} train docs, {m['val_docs']} val docs, "
      f"{m['duplicate_documents_dropped']} duplicates dropped before the split")
print(f"  source {m['source_repo']} ({m['licence']}), split seed {m['split_seed']}")
PY
else
  echo "  corpus not found at $C"
fi

echo
echo "== last completed run =="
LAST=/home/ikaaros/.hermes/cache/scratch/engrung_out/eng.json
if [ -f "$LAST" ]; then
  python3 - "$LAST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
f = d["flops"]
print(f"  {d['steps']} steps, {d['params']/1e6:.2f} M params, {d['elapsed_s']/3600:.2f} h")
print(f"  nominal budget {f['budget_flops']:.2e}, realised {f['realised_total_flops']:.2e} "
      f"({f['budget_used_fraction']:.2f}x)")
print(f"  realised chunk {f['realised_bytes_per_chunk']:.2f} bytes/chunk")
PY
else
  echo "  no completed run found"
fi

echo
echo "== local GPU in use? =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader \
    | sed 's/^/  /'
fi

echo
echo "Anything above marked UNVERIFIED is an assumption, not a fact."
