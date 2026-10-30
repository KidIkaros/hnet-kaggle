"""Pooled analysis: every measurement of every arm, across all mixtures.

The per-mixture numbers vary a lot (and so does their variance), so the pooled
figure is the honest summary: sign consistency across mixtures, not a single
mixture's magnitude.
"""
import glob
import json
import statistics as st

ARMS = ["eng", "eng_comb", "rus", "rus_comb", "cmn", "jpn", "arb", "tha", "hin"]
DIRS = ["/home/ikaaros/.hermes/cache/scratch/replres",
        "/home/ikaaros/.hermes/cache/scratch/replres2",
        "/home/ikaaros/.hermes/cache/scratch/widthres",
        "/home/ikaaros/.hermes/cache/scratch/thares",
        "/home/ikaaros/.hermes/cache/scratch/thares2"]


def lifts(ev, arm):
    out = []
    for x in ev:
        pl = x["per_language"].get(arm)
        if pl and pl["align_lift"] is not None:
            out.append(pl["align_lift"])
    return out


pool = {a: [] for a in ARMS}
per_mix = {}
for d in DIRS:
    tag = "replres" if "replres" in d else d.rsplit("/", 1)[-1]
    for f in sorted(glob.glob(d + "/seed-s*.json")):
        b = json.load(open(f))
        if not b.get("evals"):
            continue
        ev = b["evals"][-3:]
        for a in ARMS:
            v = lifts(ev, a)
            if v:
                m = st.mean(v)
                pool[a].append(m)
                per_mix.setdefault((a, tag), []).append(m)

print(f"{'arm':>9} {'n':>3} {'mean':>8} {'sd':>7} {'95% CI':>18}  {'neg seeds':>10}  verdict")
out = {}
for a in ARMS:
    v = pool[a]
    if not v:
        continue
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    hw = 1.96 * sd / (len(v) ** 0.5) if len(v) > 1 else 0.0
    neg = sum(1 for x in v if x < 0)
    if m - hw > 0:
        verdict = "aligned"
    elif m + hw < 0:
        verdict = "not aligned"
    elif m > 0.05:
        verdict = "unresolved"
    else:
        verdict = "not aligned"
    out[a] = {"n": len(v), "mean": m, "sd": sd, "ci": [m - hw, m + hw],
              "negative_seeds": neg, "verdict": verdict}
    print(f"{a:>9} {len(v):>3} {m:>+8.3f} {sd:>7.3f} [{m-hw:>+7.3f}, {m+hw:>+7.3f}]  "
          f"{neg:>4}/{len(v)}  {verdict}")

print("\nper-mixture means (the spread is the point):")
for (a, tag), v in sorted(per_mix.items()):
    if a in ("rus", "cmn", "tha", "eng_comb"):
        print(f"  {a:>9} [{tag:>8}] {st.mean(v):+.3f}  n={len(v)}")
json.dump(out, open("/home/ikaaros/.hermes/cache/scratch/pooled.json", "w"), indent=2)
