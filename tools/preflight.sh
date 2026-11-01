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

echo "=== build ($KERNEL_DIR) ==="
python3 "$KERNEL_DIR/build.py" "$CONFIG"
python3 -m py_compile "$KERNEL_DIR/main.py"

echo "=== push ==="
/home/ikaaros/.venvs/kg/bin/kaggle kernels push -p "$KERNEL_DIR" 2>&1 | tail -1
echo "pushed; check with: kaggle kernels status <slug>"
