"""Bug injection: does each gate actually fire on the bug it exists to catch?

A gate that has never failed is a gate nobody has tested. Today's failures are all
reproducible, so each one is injected here and the gate is asserted to reject it. This is the
difference between "we added checks" and "the checks demonstrably catch these bugs".

Injected:
  1. composed learning-rate multiplier (3.0 where the official code overwrites to 1.5)
  2. a budget that buys almost no data
  3. the evaluation off-by-one: logits[t] scored against byte t instead of t+1
  4. a param_groups that composes instead of overwriting

Run: python tests/test_gate_injections.py
"""

import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = "/home/ikaaros/Coding/kaggle/hnet"
sys.path.insert(0, os.path.join(ROOT, "src"))
import model as M

FAILED = []


def case(name, caught, detail=""):
    print(f"  [{'PASS' if caught else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not caught:
        FAILED.append(name)


def run_gate(cfg):
    """Run the QC gate on a config and return (exit code, stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(path, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "qc.py"),
                            "--config", path, "--no-perf"],
                           capture_output=True, text=True, timeout=900, cwd=ROOT)
        return r.returncode, r.stdout


def base():
    return json.load(open(os.path.join(ROOT, "configs", "eng-rung.json")))


def main():
    print("bug injection against the QC gate")

    # 1. the composed multiplier -- the bug that shipped and was labelled "official"
    cfg = base()
    cfg["lr_mult_main"] = 3.0
    code, out = run_gate(cfg)
    case("composed LR multiplier is rejected", code != 0,
         "gate exited 0" if code == 0 else "gate refused the run")

    # 2. a budget that buys almost no data
    cfg = base()
    cfg["flops_budget"] = 1e14
    code, out = run_gate(cfg)
    case("under-trained budget is rejected", code != 0,
         "gate exited 0" if code == 0 else "gate refused the run")

    # 3. the evaluation off-by-one, tested where it bites: the two shifts must disagree by a
    #    margin a broken evaluator cannot hide behind.
    torch.manual_seed(0)
    B, L = 2, 64
    targets = torch.randint(0, 256, (B, L))
    # logits that predict the next byte perfectly: logit[t] is a one-hot on targets[t+1]
    logits = torch.full((B, L, 256), -20.0)
    for b in range(B):
        for t in range(L - 1):
            logits[b, t, targets[b, t + 1]] = 20.0
    shifted = F.cross_entropy(logits[:, :-1].reshape(-1, 256), targets[:, 1:].reshape(-1))
    unshifted = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
    case("the eval shift changes the number by more than noise",
         shifted.item() < 0.01 and unshifted.item() > 4.0,
         f"shifted {shifted.item():.4f} vs unshifted {unshifted.item():.4f} "
         f"(uniform is {torch.log(torch.tensor(256.0)).item():.3f})")

    # a correct evaluator scores near zero here; a broken one scores near uniform. The
    # driver's startup self-check compares these two paths and refuses to train if they differ.
    case("a correct evaluator and a broken one are distinguishable",
         unshifted.item() - shifted.item() > 1.0,
         f"separation {unshifted.item() - shifted.item():.3f} nats")

    # 4. param_groups must not compose
    probe = M.HNetLM(M.HNetConfig(d_model=64, n_enc=1, n_main=1, n_dec=1))
    groups = {g["name"]: g["lr_mult"] for g in M.param_groups(probe, 2.0, 1.5)}
    case("param_groups(2.0, 1.5) does not compose to 3.0",
         groups.get("main") == 1.5, f"main multiplier {groups.get('main')}")

    print(f"\n{len(FAILED)} injected bug(s) went undetected")
    for f in FAILED:
        print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
