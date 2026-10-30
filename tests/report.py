"""Build the final result table from the replication artifacts.

Emits the per-language mean/SD/CI, the H1 ordering test, and a markdown block that can
be pasted into the write-up. Reads every seed json it finds, so partial results are
usable.
"""
import glob
import json
import re
import statistics as st

BPC = {"eng": 1.00, "deu": 1.02, "rus": 1.80, "arb": 1.78, "hin": 2.53,
       "cmn": 2.67, "jpn": 2.68}
ORDER = ["arb", "rus", "hin", "cmn", "jpn"]
LANGS = ["eng", "deu", "rus", "cmn", "jpn", "arb"]

files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/replres/seed-s*.json"),
               key=lambda f: int(re.search(r"seed-s(\d+)", f).group(1)))
per = {l: [] for l in LANGS}
steps = []
for f in files:
    d = json.load(open(f))
    if not d.get("evals"):
        continue
    steps.append((d["seed"], d["steps"], d["stop"]))
    ev = d["evals"][-3:]
    for l in LANGS:
        vals = [x["per_language"][l]["align_lift"] for x in ev
                if x["per_language"][l]["align_lift"] is not None]
        if vals:
            per[l].append(st.mean(vals))

print("seeds:", steps, "\n")
print(f"{'lang':>5} {'B/char':>6} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  verdict")
res = {}
for l in LANGS:
    v = per[l]
    if not v:
        continue
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    hw = 1.96 * sd / (len(v) ** 0.5) if len(v) > 1 else 0.0
    lo, hi = m - hw, m + hw
    if lo > 0 or hi < 0:
        verdict = "aligned" if lo > 0 else "not aligned"
    else:
        verdict = "not aligned" if m <= 0.05 else "unresolved"
    res[l] = {"n": len(v), "mean": m, "sd": sd, "ci": [lo, hi], "verdict": verdict,
              "bpc": BPC[l], "values": v}

md = ["| script | bytes/char | n seeds | mean lift | seed SD | 95% CI | verdict |",
      "|---|---|---|---|---|---|---|"]
for l in LANGS:
    if l not in res:
        continue
    r = res[l]
    md.append(f"| {l} | {r['bpc']:.2f} | {r['n']} | {r['mean']:+.3f} | {r['sd']:.3f} | "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['verdict']} |")
print("\n".join(md))

def overlap(a, b):
    return not (res[a]["ci"][1] < res[b]["ci"][0] or res[b]["ci"][1] < res[a]["ci"][0])


print("\nH1 grouping test (predicted arb ~ rus  >  cmn ~ jpn; hin not in this mixture)")
checks = []
if "arb" in res and "rus" in res:
    checks.append(("arb ~ rus (CIs overlap)", overlap("arb", "rus")))
if "cmn" in res and "jpn" in res:
    checks.append(("cmn ~ jpn (CIs overlap)", overlap("cmn", "jpn")))
if "rus" in res and "cmn" in res:
    checks.append(("rus CI separated from cmn CI",
                   res["rus"]["ci"][0] > res["cmn"]["ci"][1]))
if "arb" in res and "cmn" in res:
    checks.append(("arb CI separated from cmn CI",
                   res["arb"]["ci"][0] > res["cmn"]["ci"][1]))
for name, ok in checks:
    print(f"  {name}: {'yes' if ok else 'NO'}")
kill = res["rus"]["ci"][0] > res["cmn"]["ci"][1] if "rus" in res and "cmn" in res else False
print(f"\nexplicit kill criterion (rus and cmn overlap): {'MET -> H1 dead' if not kill else 'not met'}")
print(f"H1: {'CONFIRMED on the languages measured' if all(c[1] for c in checks) else 'PARTIAL/FAILED'}")
from math import comb


def sign_test(vals, positive=True):
    """One-sided binomial sign test: P(k or more successes | p=0.5)."""
    n = len(vals)
    k = sum(1 for v in vals if v > 0) if positive else sum(1 for v in vals if v < 0)
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return k, n, tail / 2 ** n


print("\nsign test across seeds (one-sided, binomial)")
for l in LANGS:
    if l not in res:
        continue
    v = res[l]["values"]
    kp, n, p_pos = sign_test(v, True)
    kn, _, p_neg = sign_test(v, False)
    verdict = ("all positive" if kp == n else
               "all negative" if kn == n else
               f"{kp}/{n} positive")
    print(f"  {l}: {verdict}; P(all+ | p=.5) = {p_pos:.4f}, P(all- | p=.5) = {p_neg:.4f}")

print("hin: not in this replication's mixture -- H1's position for hin rests on the earlier,"
      "\n     mixture-sensitive runs (+0.229, -0.037, +0.081) and is UNREPLICATED.")
json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/replres/summary.json", "w"), indent=2)
open("/home/ikaaros/.hermes/cache/scratch/replres/table.md", "w").write("\n".join(md))
