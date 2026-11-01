import json

open("/tmp/eng.json", "w").write("{\"d_model\": 384, \"n_enc\": 4, \"n_main\": 6, \"n_dec\": 4, \"d_ff\": 1536, \"d_state\": 16, \"d_conv\": 4, \"n_heads\": 6, \"scan_chunk\": 64, \"target_ratio\": 6.0, \"alpha\": 0.3, \"seq_len\": 2048, \"batch\": 4, \"lr\": 0.001, \"warmup\": 2000, \"wd\": 0.1, \"clip\": 1.0, \"seeds\": [0, 1], \"lr_mult_outer\": 2.0, \"lr_mult_main\": 1.5, \"compile_blocks\": true, \"grad_checkpoint\": true, \"resume\": true, \"save_every\": 1000, \"max_steps\": 200000, \"eval_every\": 2000, \"log_every\": 200, \"wall_guard_s\": 25200, \"flops_budget\": 3e+16, \"note\": \"step 2: 3e16 nominal, both seeds in parallel, one per T4\", \"scan_impl\": \"triton\", \"data_parallel\": false, \"amp\": \"fp16\", \"eval_batches\": 40}")
# ==== PARALLEL LAUNCHER ====
# Runs one independent training job per GPU, in one session.
#
# Why this instead of DataParallel: the dual-GPU smoke showed that torch.compile inside
# DataParallel does not work here -- a parameter stays on cuda:0 while its activation is on
# cuda:1, and gradient checkpointing then trips a SavedTensorHooks assertion. The supported
# alternative (DDP) needs a multi-process rewrite of the driver.
#
# Running independent jobs is better for this experiment anyway. Quota is billed per session,
# not per GPU, so two seeds finishing in 5.9 h cost the same as one seed finishing in 5.9 h:
# the second seed is free, and it is the seed noise the write-up needs.
#
# Each job gets its own config, output directory and GPU. If one dies the other keeps going --
# the session ends when the slower one does, so there is nothing to gain by killing it.

import json
import os
import subprocess
import sys
import time

# resolve the driver next to this file, not next to the working directory: the launcher is
# run from the repo root in a dry run and from /kaggle/working in a session, and only one of
# those happens to have eng_driver.py in the cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
BASE = json.load(open(os.environ.get("CONFIG", "/tmp/eng.json")))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 1
    seeds = list(BASE.get("seeds", [0])) or [0]
    while len(seeds) < n:
        seeds.append(seeds[-1] + 1)
    seeds = seeds[:n]
    log(f"{n} device(s) visible; one job per device, seeds {seeds}")

    procs = []
    for i in range(n):
        d = os.path.join(OUT, f"s{seeds[i]}")
        os.makedirs(d, exist_ok=True)
        cfg = dict(BASE, seeds=[seeds[i]])
        cfg_path = os.path.join(d, "config.json")
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), CONFIG=cfg_path, OUT_DIR=d)
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "eng_driver.py")],
                             env=env, cwd=HERE)
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
