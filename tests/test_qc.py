"""Tests for the QC gate, at its public seam: the CLI's exit code and its record.

Expected values come from independent sources, never from recomputing what the gate does:

  - the main network's multiplier is 1.5 because the official implementation's
    `apply_optimization_params` calls `param._optim.update(...)`, so the second application
    overwrites the first. A composed reading would give 3.0, and that is the bug these tests
    exist to catch.
  - the bytes-per-parameter floor of 10 is a judgement, but it is a *fixed* one, so a config
    below it must fail regardless of what the gate's own arithmetic says.

Run: python tests/test_qc.py
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = "/home/ikaaros/Coding/kaggle/hnet"
QC = os.path.join(ROOT, "tools", "qc.py")
PY = sys.executable

FAILED = []


def case(name, config, expect_pass, must_mention=None):
    """Run the gate on a config and assert on its exit code and record."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        json.dump(config, open(path, "w"))
        r = subprocess.run([PY, QC, "--config", path, "--no-perf"],
                           capture_output=True, text=True, timeout=600, cwd=ROOT)
        ok = (r.returncode == 0) == expect_pass
        detail = f"exit {r.returncode}, expected {'pass' if expect_pass else 'fail'}"
        if ok and must_mention:
            ok = must_mention.lower() in r.stdout.lower()
            detail += f"; looked for {must_mention!r}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
        if not ok:
            FAILED.append(name)
            print("    tail:", r.stdout.strip().splitlines()[-3:])
        return r


def base(**over):
    cfg = json.load(open(os.path.join(ROOT, "configs", "eng-rung.json")))
    cfg.update(over)
    return cfg


def main():
    print("QC gate tests")

    # 1. the exact bug we shipped: the composed multiplier must fail the spec stage
    case("a composed main multiplier fails", base(lr_mult_main=3.0), False,
         must_mention="main learning-rate multiplier")

    # 2. a budget so small the model is far under-trained must fail the static stage
    case("a budget buying under 10 bytes/param fails", base(flops_budget=1e14), False,
         must_mention="bytes/param")

    # 3. the corrected config must pass the static, correctness and spec stages
    case("the corrected rung config passes", base(), True)

    # 4. the seam the spec check relies on, asserted against the official literals
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import model as M
    probe = M.HNetLM(M.HNetConfig(d_model=64, n_enc=1, n_main=1, n_dec=1))
    groups = {g["name"]: g["lr_mult"] for g in M.param_groups(probe, 2.0, 1.5)}
    ok = groups.get("outer") == 2.0 and groups.get("main") == 1.5
    print(f"  [{'PASS' if ok else 'FAIL'}] param_groups(2.0, 1.5) -> outer 2.0, main 1.5 "
          f"(got {groups})")
    if not ok:
        FAILED.append("param_groups semantics")

    print(f"\n{len(FAILED)} failing test(s)")
    for f in FAILED:
        print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
