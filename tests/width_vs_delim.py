"""Width vs word-delimitation: every language, its byte width, space rate, and lift.

Combines the 6-language sweep (swres) with the 7-language Hindi run (hinres), reading
each language from the run where it was present, at N=4 and N=6.
"""
import glob
import json
import os

import numpy as np

CORPUS = "/home/ikaaros/.hermes/cache/scratch/kout4/corpus"
LANGS = ["rus", "arb", "hin", "cmn", "jpn", "deu", "eng"]


def shape(lang):
    d = open(os.path.join(CORPUS, lang, "train.bin"), "rb").read()
    t = d.decode("utf-8")
    return len(d) / len(t), t.count(" ") / len(t)


def lift(lang, n):
    """Last-3-eval mean lift and realised bytes/chunk, preferring the Hindi run."""
    for d in ("/home/ikaaros/.hermes/cache/scratch/hinres",
              "/home/ikaaros/.hermes/cache/scratch/swres"):
        f = os.path.join(d, f"seed-s0-n{n:g}.json")
        if not os.path.exists(f):
            continue
        blob = json.load(open(f))
        if lang not in blob["evals"][0]["per_language"]:
            continue
        ev = blob["evals"][-3:]
        vals = [x["per_language"][lang]["align_lift"] for x in ev
                if x["per_language"][lang]["align_lift"] is not None]
        bpcs = [x["per_language"][lang]["bytes_per_chunk"] for x in ev]
        return (np.mean(vals) if vals else float("nan")), np.mean(bpcs)
    return float("nan"), float("nan")


print(f"{'lang':>5} {'B/char':>7} {'space':>6} | {'N=4 lift':>9} {'@B':>5} | "
      f"{'N=6 lift':>9} {'@B':>5}")
rows = []
for l in LANGS:
    bpc_char, space = shape(l)
    l4, b4 = lift(l, 4)
    l6, b6 = lift(l, 6)
    rows.append({"lang": l, "bytes_per_char": bpc_char, "space_rate": space,
                 "n4": [l4, b4], "n6": [l6, b6]})
    print(f"{l:>5} {bpc_char:>7.2f} {space:>6.3f} | {l4:>+9.3f} {b4:>5.2f} | "
          f"{l6:>+9.3f} {b6:>5.2f}")

pos = [r for r in rows if r["n4"][0] > 0.1]
neg = [r for r in rows if r["n4"][0] < -0.1]
print("\npositive at N=4:", [(r["lang"], round(r["bytes_per_char"], 2),
                             round(r["space_rate"], 3)) for r in pos])
print("negative at N=4:", [(r["lang"], round(r["bytes_per_char"], 2),
                            round(r["space_rate"], 3)) for r in neg])
json.dump(rows, open("/home/ikaaros/.hermes/cache/scratch/width_delim.json", "w"), indent=2)
