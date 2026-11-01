#!/usr/bin/env bash
# Poll a Kaggle kernel until it stops running, appending status and quota each time.
#
# Kaggle's log endpoint returns nothing while a kernel is running, so status is the only live
# signal available from here; the logs and outputs arrive when it finishes. This polls that
# signal, so a completion or an error is noticed within one interval instead of whenever
# someone happens to look.
#
#   tools/poll_run.sh mauricew/hnet-eng-3e16 [interval_seconds] [logfile]
set -uo pipefail
SLUG=${1:?usage: poll_run.sh owner/kernel-slug [interval] [logfile]}
INTERVAL=${2:-300}
LOG=${3:-/tmp/poll_${SLUG##*/}.log}
KAGGLE=/home/ikaaros/.venvs/kg/bin/kaggle

mkdir -p "$(dirname "$LOG")"
echo "polling $SLUG every ${INTERVAL}s -> $LOG"
while true; do
  ST=$("$KAGGLE" kernels status "$SLUG" 2>&1 | tail -1)
  Q=$("$KAGGLE" quota 2>&1 | sed -n '3p' | tr -s ' ')
  printf '%s | %s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ST" "$Q" >> "$LOG"
  printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "${ST#*has status }"
  case "$ST" in
    *RUNNING*|*QUEUED*|*PENDING*|*"KernelWorkerStatus.RUNNING"*) sleep "$INTERVAL" ;;
    *) break ;;
  esac
done
echo "FINAL: $ST"
