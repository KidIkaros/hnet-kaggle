"""Delimiter-landmark test: does adding spaces to Chinese make it align?

Chinese, Japanese and Thai have almost no whitespace; the aligned scripts all have it. The
removal test showed spaces are not *necessary* (deleting them from Russian and Hindi left
alignment intact). This is the other direction: are they *sufficient*?
"""
import glob
import json
import statistics as st

ARMS = ["cmn", "cmn_sp3", "cmn_sp6", "rus", "arb"]
files = sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/spacedres/seed-s*.json")
               + glob.glob("/home/ikaaros/.hermes/cache/scratch/spout/seed-s*.json"))
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
             if a in x["per_language"] and x["per_language"][a].get("align_lift") is not None]
        if v:
            per[a].append(st.mean(v))

print(f"seeds: {sorted(set(seeds))}\n")
print(f"{'arm':>9} {'n':>3} {'mean':>8} {'per-seed':>26}")
for a in ARMS:
    v = per[a]
    if not v:
        continue
    print(f"{a:>9} {len(v):>3} {st.mean(v):>+8.3f}   {[round(x,3) for x in v]}")

print("\npre-registered reading:")
for base, sp in (("cmn", "cmn_sp3"), ("cmn", "cmn_sp6")):
    if base in per and sp in per and per[base] and per[sp]:
        d = st.mean(per[sp]) - st.mean(per[base])
        print(f"  {base} {st.mean(per[base]):+.3f} -> {sp} {st.mean(per[sp]):+.3f} "
              f"(change {d:+.3f}) {'ALIGNED' if st.mean(per[sp]) > 0.15 else 'still not aligned'}")
