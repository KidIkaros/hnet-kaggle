"""Replication analysis: per-language mean, seed SD, 95% CI, and the ordering test.

Decision rules are pre-registered in ticket 15:
  aligned     -- CI excludes 0
  not aligned -- CI includes 0 and mean <= +0.05
  unresolved  -- otherwise

Kill criterion for H1 (width): the per-language means must order
arb ~ rus > hin > cmn ~ jpn. If rus and cmn overlap, H1 is dead.
"""
import glob
import json
import re
import statistics as st

BPC = {"eng": 1.00, "deu": 1.02, "rus": 1.80, "arb": 1.78, "hin": 2.53,
       "cmn": 2.67, "jpn": 2.68}
ORDER = ["arb", "rus", "hin", "cmn", "jpn"]

files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/replres/seed-s*.json"),
               key=lambda f: int(re.search(r"seed-s(\d+)", f).group(1)))
if not files:
    files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/replres/seed-s0-n4.json"))
langs = ["eng", "deu", "rus", "cmn", "jpn", "arb"]

per = {l: [] for l in langs}
seeds_used = []
for f in files:
    d = json.load(open(f))
    if not d["evals"]:
        continue
    seeds_used.append(d["seed"])
    ev = d["evals"][-3:]
    for l in langs:
        vals = [x["per_language"][l]["align_lift"] for x in ev
                if x["per_language"][l]["align_lift"] is not None]
        if vals:
            per[l].append(st.mean(vals))

print(f"seeds used: {seeds_used}\n")
print(f"{'lang':>5} {'B/char':>6} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  verdict")
res = {}
for l in langs:
    v = per[l]
    if not v:
        continue
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    hw = 1.96 * sd / (len(v) ** 0.5) if len(v) > 1 else 0.0
    lo, hi = m - hw, m + hw
    if not (lo > 0 or hi < 0):
        verdict = "not aligned" if m <= 0.05 else "unresolved"
    else:
        verdict = "aligned" if lo > 0 else "not aligned"
    res[l] = {"n": len(v), "mean": m, "sd": sd, "ci": [lo, hi], "verdict": verdict,
              "bpc": BPC[l]}
    print(f"{l:>5} {BPC[l]:>6.2f} {len(v):>3} {m:>+8.3f} {sd:>7.3f} "
          f"[{lo:>+7.3f}, {hi:>+7.3f}]  {verdict}")

print("\nH1 ordering test (predicted arb ~ rus > hin > cmn ~ jpn)")
got = [l for l in ORDER if l in res]
ok = True
for a, b in zip(got, got[1:]):
    ma, mb = res[a]["mean"], res[b]["mean"]
    good = ma > mb
    ok &= good
    print(f"  {a} ({ma:+.3f}) {'>' if good else 'NOT >'} {b} ({mb:+.3f})")
sep = res["rus"]["ci"][0] > res["cmn"]["ci"][1]
print(f"  rus CI and cmn CI separated: {sep}")
print(f"\nH1 verdict: {'CONFIRMED' if ok and sep else 'NOT CONFIRMED (kill criterion met)'}")
json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/replres/summary.json", "w"), indent=2)
