"""Per-N alignment summary from the sweep artifacts (last 3 evals averaged)."""
import json
import glob

LANGS = ["eng", "deu", "rus", "cmn", "jpn", "arb"]
rows = []
for f in sorted(glob.glob("/home/ikaaros/.hermes/cache/scratch/swres/seed-s0-n*.json")):
    d = json.load(open(f))
    ev = d["evals"][-3:]
    rec = {"N": d["ratio"], "steps": d["steps"], "stop": d["stop"]}
    for l in LANGS:
        lifts = [x["per_language"][l]["align_lift"] for x in ev
                 if x["per_language"][l]["align_lift"] is not None]
        bpcs = [x["per_language"][l]["bytes_per_chunk"] for x in ev]
        rec[l] = (sum(lifts) / len(lifts) if lifts else None,
                  sum(bpcs) / len(bpcs))
    rows.append(rec)

print("N   " + "  ".join(f"{l:>16}" for l in LANGS))
for r in rows:
    cells = []
    for l in LANGS:
        lift, bpc = r[l]
        cells.append(f"{lift:+.3f} @{bpc:4.2f}B")
    print(f"{r['N']:>3g} " + "  ".join(f"{c:>16}" for c in cells))

print("\ncharacter width per script: rus/arb 2B, cmn 3B, jpn mixed")
print("chunks per character:")
for r in rows:
    line = []
    for l, w in (("rus", 2), ("arb", 2), ("cmn", 3), ("jpn", 2.6)):
        _, bpc = r[l]
        line.append(f"{l} {bpc/w:.2f}")
    print(f"  N={r['N']:>3g}: " + ", ".join(line))
json.dump(rows, open("/home/ikaaros/.hermes/cache/scratch/swres/summary.json", "w"), indent=2)
