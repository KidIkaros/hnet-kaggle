"""Preview the replication from the live kernel log, per seed and per language.

The kernel logs "  s<seed> eval <lang> bpb .. bpc .. lift +X" every 250 steps; the
per-seed value we care about is the mean of the last three such lines per language.
"""
import re
import statistics as st
from collections import defaultdict

LOG = "/home/ikaaros/.hermes/cache/scratch/rl.txt"
LANGS = ["eng", "deu", "rus", "cmn", "jpn", "arb"]

rows = defaultdict(lambda: defaultdict(list))
for line in open(LOG, errors="ignore"):
    m = re.search(r"s(\d+) eval (\w+) bpb [0-9.]+ bpc [0-9.]+ lift ([-+0-9.]+|n/a)", line)
    if m:
        seed, lang, lift = int(m.group(1)), m.group(2), m.group(3)
        if lift != "n/a":
            rows[seed][lang].append(float(lift))

print(f"{'seed':>5} " + " ".join(f"{l:>7}" for l in LANGS))
means = defaultdict(list)
for seed in sorted(rows):
    cells = []
    for l in LANGS:
        v = rows[seed][l][-3:]
        if v:
            m = st.mean(v)
            means[l].append(m)
            cells.append(f"{m:+7.3f}")
        else:
            cells.append(f"{'--':>7}")
    print(f"{seed:>5} " + " ".join(cells))

print("\nso far (across completed seeds)")
for l in LANGS:
    v = means[l]
    if not v:
        continue
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"  {l}: n={len(v)} mean {m:+.3f} sd {sd:.3f} "
          f"values {[round(x, 3) for x in v]}")
