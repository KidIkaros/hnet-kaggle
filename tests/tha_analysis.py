"""Thai-cell analysis: the pre-registered test from ticket 18."""
import glob
import json
import re
import statistics as st

ARMS = ["tha", "eng_comb", "cmn", "rus"]
BPC = {"tha": 2.82, "eng_comb": 2.69, "cmn": 2.67, "rus": 1.80}
SPACE = {"tha": 0.050, "eng_comb": 0.155, "cmn": 0.036, "rus": 0.132}
ASCII = {"tha": 0.032, "eng_comb": 0.372, "cmn": 0.062, "rus": 0.113}

files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/thares/seed-s*.json"),
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

print(f"{'arm':>9} {'B/char':>6} {'space':>6} {'ascii':>6} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  verdict")
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
              "values": v}
    print(f"{a:>9} {BPC[a]:>6.2f} {SPACE[a]:>6.3f} {ASCII[a]:>6.3f} {len(v):>3} {m:>+8.3f} "
          f"{sd:>7.3f} [{lo:>+7.3f}, {hi:>+7.3f}]  {verdict}")
    print(f"{'':>9} per-seed: {[round(x, 3) for x in v]}")

tha = res["tha"]
aligned = tha["verdict"] == "aligned"
print(f"\nThai verdict: {tha['verdict']} (mean {tha['mean']:+.3f}, CI "
      f"[{tha['ci'][0]:+.3f}, {tha['ci'][1]:+.3f}])")
print("pre-registered consequence:")
if aligned:
    print("  -> the effect is specific to LOGOGRAPHIC scripts: the chunker learns character")
    print("     structure wherever characters are alphabetic, at any width, spaced or not.")
else:
    print("  -> every surviving explanation (width / spaces / ASCII landmarks) predicts Thai")
    print("     fails, and it does: the honest answer is a TWO-FACTOR description with no")
    print("     mechanism identified.")
json.dump(res, open("/home/ikaaros/.hermes/cache/scratch/thares/summary.json", "w"), indent=2)
