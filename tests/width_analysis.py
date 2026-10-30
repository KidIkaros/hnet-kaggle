"""Width-manipulation analysis: the pre-registered test from ticket 17.

Predicted: eng ~0 (structural), rus positive, cmn negative, eng_comb not aligned,
rus_comb not aligned and worse than rus. Falsifier: eng_comb clearly positive
(CI excludes 0 and mean > +0.10) kills H1.
"""
import glob
import json
import re
import statistics as st
from math import comb

BPC = {"eng": 1.00, "eng_comb": 2.69, "rus": 1.80, "rus_comb": 3.52, "cmn": 2.67}
ARMS = ["eng", "eng_comb", "rus", "rus_comb", "cmn"]
PRED = {"eng": "~0 (structural)", "eng_comb": "not aligned", "rus": "positive (+0.178)",
        "rus_comb": "not aligned", "cmn": "negative (-0.147)"}
REPL = {"rus": 0.178, "cmn": -0.147, "eng": 0.0}

files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/widthres/seed-s*.json"),
               key=lambda f: int(re.search(r"seed-s(\d+)", f).group(1)))
per = {a: [] for a in ARMS}
for f in files:
    d = json.load(open(f))
    ev = d["evals"][-3:]
    for a in ARMS:
        v = [x["per_language"][a]["align_lift"] for x in ev
             if x["per_language"][a]["align_lift"] is not None]
        if v:
            per[a].append(st.mean(v))

print(f"{'arm':>9} {'B/char':>6} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  {'verdict':<12} prediction")
res = {}
for a in ARMS:
    v = per[a]
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    hw = 1.96 * sd / (len(v) ** 0.5) if len(v) > 1 else 0.0
    lo, hi = m - hw, m + hw
    if lo > 0 or hi < 0:
        verdict = "aligned" if lo > 0 else "not aligned"
    else:
        verdict = "not aligned" if m <= 0.05 else "unresolved"
    res[a] = {"n": len(v), "mean": m, "sd": sd, "ci": [lo, hi], "verdict": verdict,
              "values": v, "bpc": BPC[a]}
    print(f"{a:>9} {BPC[a]:>6.2f} {len(v):>3} {m:>+8.3f} {sd:>7.3f} "
          f"[{lo:>+7.3f}, {hi:>+7.3f}]  {verdict:<12} {PRED[a]}")

print("\nharness controls (must reproduce the replication):")
for a in ("rus", "cmn", "eng"):
    d = res[a]["mean"] - REPL[a]
    ok = abs(d) < 0.15
    print(f"  {a}: here {res[a]['mean']:+.3f} vs replication {REPL[a]:+.3f} "
          f"(diff {d:+.3f}) {'OK' if ok else 'HARNESS MISMATCH'}")

print("\npre-registered checks")
ec = res["eng_comb"]
falsified = ec["ci"][0] > 0 and ec["mean"] > 0.10
checks = [
    ("rus is aligned", res["rus"]["verdict"] == "aligned"),
    ("cmn is not aligned", res["cmn"]["verdict"] == "not aligned"),
    ("eng_comb is not aligned", ec["verdict"] == "not aligned"),
    ("rus_comb is not aligned", res["rus_comb"]["verdict"] == "not aligned"),
    ("rus_comb <= rus", res["rus_comb"]["mean"] <= res["rus"]["mean"]),
    ("eng ~0 (structural)", abs(res["eng"]["mean"]) < 0.05),
]
for name, ok in checks:
    print(f"  {name}: {'yes' if ok else 'NO'}")
print(f"\nfalsifier (eng_comb clearly positive) triggered: {falsified}")
if falsified:
    print("  -> H1 IS DEAD. Width does not cause the effect.")
else:
    print("  -> H1 survives; width predicts alignment loss as pre-registered.")
json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/widthres/summary.json", "w"), indent=2)
