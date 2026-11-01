"""Early stopping: does each guard fire on the thing it exists to catch?

The question that prompted this file was "do we have any early stopping if something goes
awry?". The answer had a hole in it: the canary tested `bpb > 8.5`, and in Python `nan > 8.5`
is False, so a run that diverged to NaN would have trained for hours and handed back a
checkpoint of nothing. A guard that cannot fire on the worst case is not a guard.

Run: python tests/test_guards.py
"""

import math
import os
import sys

ROOT = "/home/ikaaros/Coding/kaggle/hnet"
os.environ.setdefault("CONFIG", os.path.join(ROOT, "configs", "eng-rung.json"))
os.environ.setdefault("OUT_DIR", "/tmp/guardtest")
os.environ.setdefault("CORPUS", "/home/ikaaros/.hermes/cache/scratch/eng_corpus")
sys.path.insert(0, os.path.join(ROOT, "kernel_eng"))

from main import guard_reason  # noqa: E402

FAILED = []


def case(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def main():
    print("early-stopping guards")
    nan, inf = float("nan"), float("inf")

    case("NaN stops the run", guard_reason(100, nan, 1.4) is not None,
         "this is the case the old `bpb > 8.5` check missed")
    case("NaN would have passed a naive threshold", not (nan > 8.5),
         "nan > 8.5 is False — hence the isfinite test")
    case("+Inf stops the run", guard_reason(100, inf, 1.4) is not None)
    case("-Inf stops the run", guard_reason(100, -inf, 1.4) is not None,
         "a negative loss is not a good result, it is a broken one")
    case("worse than uniform stops the run", guard_reason(100, 8.6, 1.4) is not None)
    case("step 0 is exempt", guard_reason(0, 7.94, None) is None,
         "an untrained model sits at 7.94; the canary is for a run that had a chance to learn")
    case("a healthy run keeps going", guard_reason(5000, 1.42, 1.40) is None)
    case("a small regression keeps going", guard_reason(5000, 1.55, 1.40) is None,
         "1.55 is worse than 1.40 but well under the 50% line")
    case("a 50% regression stops the run", guard_reason(5000, 2.20, 1.40) is not None)
    case("the regression guard needs a best to compare against",
         guard_reason(100, 5.0, None) is None,
         "no best yet, and 5.0 is under uniform, so there is nothing to judge it by")

    print(f"\n{len(FAILED)} guard(s) did not fire")
    for f in FAILED:
        print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
