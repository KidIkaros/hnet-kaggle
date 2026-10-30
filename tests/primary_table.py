"""Primary table: the headline 6-language mixture, seeds 0-9 (10 seeds).

This is the pre-registered configuration (ticket 15): fixed mixture, N=4, 5000 steps,
evaluation on held-out val.bin with the last-3-eval mean per seed. Decision rules are
the ones fixed before the run.
"""
import glob
import json
import re
import statistics as st

LANGS = ["eng", "deu", "rus", "cmn", "jpn", "arb"]
BPC = {"eng": 1.00, "deu": 1.02, "rus": 1.80, "cmn": 2.67, "jpn": 2.68, "arb": 1.78}
DIRS = ["/home/ikaaros/.hermes/cache/scratch/replres",
        "/home/ikaaros/.hermes/cache/scratch/replres2"]

files = []
for d in DIRS:
    files += sorted(glob.glob(d + "/seed-s*.json"),
                    key=lambda f: int(re.search(r"seed-s(\d+)", f).group(1)))
files = sorted(files, key=lambda f: int(re.search(r"seed-s(\d+)", f).group(1)))

per = {l: [] for l in LANGS}
seeds = []
for f in files:
    d = json.load(open(f))
    if not d.get("evals"):
        continue
    seeds.append(d["seed"])
    ev = d["evals"][-3:]
    for l in LANGS:
        v = [x["per_language"][l]["align_lift"] for x in ev
             if x["per_language"][l]["align_lift"] is not None]
        if v:
            per[l].append(st.mean(v))

print(f"seeds: {sorted(seeds)}\n")
print(f"{'script':>6} {'B/char':>6} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  {'neg':>6}  verdict")
res = {}
for l in LANGS:
    v = per[l]
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    hw = 1.96 * sd / (len(v) ** 0.5) if len(v) > 1 else 0.0
    lo, hi = m - hw, m + hw
    if lo > 0:
        verdict = "aligned"
    elif hi < 0:
        verdict = "not aligned"
    elif m > 0.05:
        verdict = "unresolved"
    else:
        verdict = "not aligned"
    neg = sum(1 for x in v if x < 0)
    res[l] = {"n": len(v), "mean": m, "sd": sd, "ci": [lo, hi], "verdict": verdict,
              "negative_seeds": neg, "values": v, "bpc": BPC[l]}
    print(f"{l:>6} {BPC[l]:>6.2f} {len(v):>3} {m:>+8.3f} {sd:>7.3f} "
          f"[{lo:>+7.3f}, {hi:>+7.3f}]  {neg:>2}/{len(v)}  {verdict}")

print("\nH1 grouping (predicted arb ~ rus > cmn ~ jpn):")
def overlap(a, b):
    return not (res[a]["ci"][1] < res[b]["ci"][0] or res[b]["ci"][1] < res[a]["ci"][0])
for a, b in (("arb", "rus"), ("cmn", "jpn")):
    print(f"  {a} ~ {b} (CIs overlap): {overlap(a, b)}")
print(f"  rus CI separated from cmn CI: {res['rus']['ci'][0] > res['cmn']['ci'][1]}")
json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/primary10.json", "w"), indent=2)
