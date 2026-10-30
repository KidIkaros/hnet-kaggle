"""Encoding-invariance analysis: UTF-8 vs UTF-16LE boundary placement.

The UTF-16 model's u16_lift is the same for every UTF-8 checkpoint it is compared
against (one UTF-16 model, many UTF-8 models), so it is averaged once per seed and
the UTF-8 side is averaged over the ten checkpoints.
"""
import json
import statistics as st
from collections import defaultdict

data = json.load(open("/home/ikaaros/.hermes/cache/scratch/invres/invariance.json"))
LANGS = ["eng", "rus", "cmn", "jpn", "arb"]

u8 = defaultdict(list)
u16 = defaultdict(list)
agree = defaultdict(list)
f1 = defaultdict(list)
u8b = defaultdict(list)
u16b = defaultdict(list)
for row in data:
    for lang in LANGS:
        r = row[lang]
        if r["u8_lift"] is not None:
            u8[lang].append(r["u8_lift"])
        if r["u16_lift"] is not None:
            u16[lang].append(r["u16_lift"])
        if r["agreement"] is not None:
            agree[lang].append(r["agreement"])
            f1[lang].append(r["char_f1"])
        if r["u8_boundaries"] is not None:
            u8b[lang].append(r["u8_boundaries"])
        if r["u16_boundaries"] is not None:
            u16b[lang].append(r["u16_boundaries"])

print(f"rows (u16 seed x utf-8 checkpoint): {len(data)}")
print(f"{'script':>6} {'u8_lift':>18} {'u16_lift':>18} {'agreement':>18} {'char_f1':>16}")
out = {}
for lang in LANGS:
    m8 = st.mean(u8[lang]) if u8[lang] else None
    m16 = st.mean(u16[lang]) if u16[lang] else None
    ma = st.mean(agree[lang]) if agree[lang] else None
    mf = st.mean(f1[lang]) if f1[lang] else None
    out[lang] = {"u8_lift": m8, "u16_lift": m16, "agreement": ma, "char_f1": mf,
                 "u8_boundaries": st.mean(u8b[lang]) if u8b[lang] else None,
                 "u16_boundaries": st.mean(u16b[lang]) if u16b[lang] else None,
                 "n": len(u8[lang])}
    fmt = lambda v: f"{v:>+8.3f}" if v is not None else "     n/a"
    print(f"{lang:>6} {fmt(m8):>18} {fmt(m16):>18} "
          f"{fmt(ma):>18} {fmt(mf):>16}")

print("\nper UTF-16 seed (u16_lift is per seed; u8 is the paired UTF-8 model)")
by_seed = defaultdict(lambda: defaultdict(list))
for row in data:
    for lang in LANGS:
        r = row[lang]
        if r["u16_lift"] is not None:
            by_seed[row["u16_seed"]][lang].append(r["u16_lift"])
for seed in sorted(by_seed):
    print(f"  u16 seed {seed}: " + "  ".join(
        f"{l} {st.mean(by_seed[seed][l]):+.3f}" for l in LANGS if by_seed[seed][l]))

print("\nreading:")
for lang in LANGS:
    o = out[lang]
    if o["u8_lift"] is None or o["u16_lift"] is None:
        continue
    same = (o["u8_lift"] > 0) == (o["u16_lift"] > 0)
    print(f"  {lang}: u8 {o['u8_lift']:+.3f} -> u16 {o['u16_lift']:+.3f} "
          f"({'same sign' if same else 'SIGN FLIPS'}), "
          f"agreement {o['agreement']:.3f}, f1 {o['char_f1']:.3f}")
json.dump(out, open("/home/ikaaros/.hermes/cache/scratch/invres/summary.json", "w"), indent=2)
