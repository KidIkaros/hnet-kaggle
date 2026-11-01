# ==== PARALLEL LAUNCHER ====
# Runs one independent training job per GPU, in one session.
#
# Why this instead of DataParallel: the dual-GPU smoke showed torch.compile inside
# DataParallel does not work here -- a parameter stays on cuda:0 while its activation is on
# cuda:1, and gradient checkpointing then trips a SavedTensorHooks assertion. The supported
# alternative (DDP) needs a multi-process rewrite of the driver.
#
# Running independent jobs is better for this experiment anyway. Quota is billed per session,
# not per GPU, so two seeds finishing in 5.9 h cost the same as one seed finishing in 5.9 h:
# the second seed is free, and it is the seed noise the write-up needs.
#
# Kaggle runs a script kernel as a single file copied to /kaggle/src/script.py; siblings are
# not there. So the driver source travels inside this file, base64-encoded (build.py fills in
# DRIVER_B64), and is written to a temporary path before the workers start.

import base64
import json
import os
import subprocess
import sys
import tempfile
import time

DRIVER_B64 = None  # replaced at build time

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
BASE = json.load(open(os.environ.get("CONFIG", "/tmp/eng.json")))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    if not DRIVER_B64:
        raise SystemExit("DRIVER_B64 was not filled in by build.py; this launcher cannot run")
    import torch

    n = torch.cuda.device_count() if torch.cuda.is_available() else 1
    seeds = list(BASE.get("seeds", [0])) or [0]
    while len(seeds) < n:
        seeds.append(seeds[-1] + 1)
    seeds = seeds[:n]
    log(f"{n} device(s) visible; one job per device, seeds {seeds}")

    tmp = tempfile.mkdtemp()
    driver = os.path.join(tmp, "eng_driver.py")
    open(driver, "w").write(base64.b64decode(DRIVER_B64).decode())

    procs = []
    for i in range(n):
        d = os.path.join(OUT, f"s{seeds[i]}")
        os.makedirs(d, exist_ok=True)
        cfg = dict(BASE, seeds=[seeds[i]])
        cfg_path = os.path.join(d, "config.json")
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), CONFIG=cfg_path, OUT_DIR=d)
        p = subprocess.Popen([sys.executable, driver], env=env, cwd=tmp)
        procs.append((seeds[i], p))
        log(f"seed {seeds[i]} -> cuda:{i}, out {d}, pid {p.pid}")

    t0 = time.time()
    for seed, p in procs:
        rc = p.wait()
        log(f"seed {seed} exited with {rc} after {(time.time()-t0)/3600:.2f} h")

    # a job that failed leaves no eng.json; say so here rather than leaving it to the reader
    missing = [s for s, _ in procs if not os.path.exists(os.path.join(OUT, f"s{s}", "eng.json"))]
    if missing:
        log(f"WARNING: no results for seeds {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
