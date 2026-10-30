"""Thai cell, merged over both kernels: seeds 0-2 (n=3) plus seeds 3-4 (n=5 total).

The pre-registered control rule: rus and cmn must reproduce their earlier values
within 0.15, or the run is a harness failure rather than a result.
"""
import glob
import json
import re
import statistics as st

ARMS = ["tha", "eng_comb", "cmn", "rus"]
BPC = {"tha": 2.82, "eng_comb": 2.69, "cmn": 2.67, "rus": 1.80}
EARLIER = {"rus": 0.178, "cmn": -0.205, "eng_comb": 0.094}

files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/thares/seed-s*.json")
               + glob.glob("/home/ikaaros/.hermes/cache/scratch/thares2/seed-s*.json"),
               key=lambda f: (0 if "thares/" in f else 1,
                              int(re.search(r"seed-s(\d+)", f).group(1))))
per = {a: [] for a in ARMS}
seeds = []
for f in files:
    d = json.load(open(f))
    if not d.get("evals"):
        continue
    seeds.append(d["seed"])
    ev = d["evals"][-3:]
    for a in ARMS:
        v = [x["per_language"][a]["align_lift"] for x in ev
             if x["per_language"][a]["align_lift"] is not None]
        if v:
            per[a].append(st.mean(v))

print(f"seeds: {sorted(seeds)}\n")
print(f"{'arm':>9} {'B/char':>6} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  verdict")
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
          f"[{lo:>+7.3f}, {hi:>+7.3f}]  {verdict}")
    print(f"{'':>9} per-seed: {[round(x, 3) for x in v]}")

print("\nharness controls (must reproduce the earlier run within 0.15):")
ok_all = True
for a, ref in EARLIER.items():
    d = res[a]["mean"] - ref
    ok = abs(d) < 0.15
    ok_all &= ok
    print(f"  {a}: here {res[a]['mean']:+.3f} vs earlier {ref:+.3f} "
          f"(diff {d:+.3f}) {'OK' if ok else 'OUT OF TOLERANCE'}")

tha = res["tha"]
print(f"\nThai: {tha['verdict']} (mean {tha['mean']:+.3f}, CI "
      f"[{tha['ci'][0]:+.3f}, {tha['ci'][1]:+.3f}], "
      f"{sum(1 for x in tha['values'] if x < 0)}/{tha['n']} seeds negative)")
print(f"harness verdict: {'PASS' if ok_all else 'FAIL -- report as harness failure first'}")
json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/thares/merged.json", "w"), indent=2)
