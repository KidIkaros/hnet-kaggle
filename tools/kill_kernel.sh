#!/bin/bash
# Kill a kernel's live session. Deleting the kernel is the ONLY reliable CLI kill:
# pushing a new version queues behind (or beside) the running session, and
# `kernels status` reports the latest version only -- a CPU no-op can read COMPLETE
# while the GPU session underneath is still burning quota.
# Postmortem 2026-11-01: a "killed" hung GPU probe billed 2.47 h past its kill push.
#
# usage: tools/kill_kernel.sh <owner>/<slug> [more slugs...]
set -e
KG=/home/ikaaros/.venvs/kg/bin/kaggle
for slug in "$@"; do
  echo "== $slug: status before kill =="
  "$KG" kernels status "$slug" 2>&1 | tail -1 || true
  "$KG" kernels delete -y "$slug"
  echo "== $slug deleted (session terminated; kernel and its outputs are gone) =="
done
echo "== verify the meter stopped: run 'kaggle quota' now and again in ~3 min =="
