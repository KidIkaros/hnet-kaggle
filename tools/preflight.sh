#!/usr/bin/env bash
# Preflight gate: run QC locally, and push to Kaggle only if every check passes.
#
# The point is that a push cannot happen by accident. Today's wasted GPU hours came from
# launching runs whose configuration was wrong in ways a local check would have caught in two
# minutes, so the push is now downstream of the check.
#
#   tools/preflight.sh configs/eng-rung.json kernel_eng
#
# Environment:
#   QC_PYTHON     interpreter with CUDA torch for the performance stage
#                 (default: /home/ikaaros/.venvs/hnetgpu/bin/python)
#   QUOTA_HOURS   GPU hours left on Kaggle (default 23)
#   SKIP_PERF=1   run correctness and static checks only (no GPU needed)
set -euo pipefail

CONFIG=${1:?usage: preflight.sh CONFIG KERNEL_DIR}
KERNEL_DIR=${2:?usage: preflight.sh CONFIG KERNEL_DIR}
QUOTA_HOURS=${QUOTA_HOURS:-23}
cd "$(dirname "$0")/.."

QC_PYTHON=${QC_PYTHON:-/home/ikaaros/.venvs/hnetgpu/bin/python}
if [ ! -x "$QC_PYTHON" ]; then
  QC_PYTHON=/home/ikaaros/.venvs/hnet/bin/python
  echo "warning: no CUDA interpreter at the default path, falling back to $QC_PYTHON"
  echo "         the performance stage will be skipped, not passed"
fi

# The performance stage runs on the owner's desktop GPU, so it is opt-in. Correctness and
# static checks are cheap and always run; a measurement of one configuration is allowed with
# PERF=local, and anything comparing configurations belongs on Kaggle as a short job.
if [ "${PERF:-0}" = "local" ]; then
  EXTRA="--perf-steps ${PERF_STEPS:-3}"
else
  EXTRA="--no-perf"
  echo "note: skipping the GPU performance stage (set PERF=local to include one short measurement)"
fi

echo "=== QC ($CONFIG) ==="
"$QC_PYTHON" tools/qc.py --config "$CONFIG" --quota-hours "$QUOTA_HOURS" $EXTRA

# A tripwire against launching a long run by accident. Everything below this line costs
# quota, and the failures that cost the most were runs nobody meant to start this way.
STEPS=$(python3 - "$CONFIG" <<'PY'
import json, sys
sys.path.insert(0, "src")
from flops import steps_for_budget
from model import HNetConfig
c = json.load(open(sys.argv[1]))
f = {k: v for k, v in c.items() if k in HNetConfig.__dataclass_fields__}
cfg = HNetConfig(**f)
budget = float(c.get("flops_budget") or 1e16)
print(steps_for_budget(cfg, int(c["seq_len"]), int(c["batch"]), cfg.target_ratio, budget)[0])
PY
)
echo "derived steps: $STEPS"
if [ "$STEPS" -gt 2000 ] && [ "${FULL:-0}" != "1" ]; then
  echo
  echo "refusing to push a ${STEPS}-step run without confirmation."
  echo "This is a long run that will spend real quota. Re-run with FULL=1 to launch it:"
  echo "    FULL=1 tools/preflight.sh $CONFIG $KERNEL_DIR"
  exit 1
fi

echo "=== build ($KERNEL_DIR) ==="
python3 "$KERNEL_DIR/build.py" "$CONFIG"
python3 -m py_compile "$KERNEL_DIR/main.py"

echo "=== push ==="
/home/ikaaros/.venvs/kg/bin/kaggle kernels push -p "$KERNEL_DIR" 2>&1 | tail -1
echo "pushed; check with: kaggle kernels status <slug>"
