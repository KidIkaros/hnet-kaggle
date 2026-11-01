#!/usr/bin/env bash
# Watch a Kaggle kernel run until it stops.
#
# Kaggle serves a script kernel's logs only after it finishes, so progress cannot be read
# mid-run. What can be watched is the thing that actually matters: whether the run is still
# alive, how much quota it has consumed, and how long it has been going. That is enough to
# catch the failure that would otherwise cost hours -- a run that died early looks exactly
# like a healthy one if nobody is looking.
#
#   tools/watch_run.sh mauricew/hnet-english-rung [interval_seconds]
#
# Writes:
#   .scratch/watch/<slug>.log      one timestamped line per poll
#   .scratch/watch/<slug>.status   the latest line, for a quick read
# Exits 0 on COMPLETE, 1 on ERROR, so the notification says which.
set -uo pipefail

SLUG=${1:?usage: watch_run.sh OWNER/SLUG [interval]}
INTERVAL=${2:-600}
KAGGLE=/home/ikaaros/.venvs/kg/bin/kaggle
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/.scratch/watch"
NAME=$(basename "$SLUG")
LOG="$DIR/$NAME.log"
STATUS="$DIR/$NAME.status"
mkdir -p "$DIR"

start=$(date +%s)
echo "$(date -Is) watching $SLUG every ${INTERVAL}s" >> "$LOG"

while true; do
  now=$(date +%s)
  elapsed=$(( now - start ))
  raw=$("$KAGGLE" kernels status "$SLUG" 2>&1 | tail -1)
  quota=$("$KAGGLE" quota 2>/dev/null | awk '/^GPU/ {print $3}' | head -1)
  if echo "$raw" | grep -q "COMPLETE"; then
    line="$(date -Is) COMPLETE after $((elapsed/3600))h$(( (elapsed%3600)/60 ))m, GPU left ${quota:-?}"
    echo "$line" | tee -a "$LOG" > "$STATUS"
    exit 0
  elif echo "$raw" | grep -q "ERROR"; then
    line="$(date -Is) ERROR after $((elapsed/3600))h$(( (elapsed%3600)/60 ))m, GPU left ${quota:-?}"
    echo "$line" | tee -a "$LOG" > "$STATUS"
    exit 1
  elif echo "$raw" | grep -q "RUNNING"; then
    line="$(date -Is) running, ${elapsed}s elapsed, GPU left ${quota:-?}"
  else
    line="$(date -Is) unexpected status: ${raw:0:120}"
  fi
  echo "$line" | tee -a "$LOG" > "$STATUS"
  sleep "$INTERVAL"
done
