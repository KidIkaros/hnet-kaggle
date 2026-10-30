"""Matched-N=4 UTF-16 invariance: 10 UTF-8 checkpoints x 2 UTF-16 seeds."""
import json
import statistics as st

d = json.load(open("/home/ikaaros/.hermes/cache/scratch/u16n4res/invariance.json"))
langs = ["eng", "rus", "cmn", "jpn", "arb"]
print(f"{len(d)} pairings = {len({x['u8_ckpt'] for x in d})} UTF-8 checkpoints x "
      f"{len({x['u16_seed'] for x in d})} UTF-16 seeds\n")
print(f"{'lang':>5} {'u8 lift':>16} {'u16 lift':>16} {'agree':>7} {'f1':>6} {'sign flips':>11}")
for l in langs:
    u8 = [x[l]["u8_lift"] for x in d]
    u16 = [x[l]["u16_lift"] for x in d]
    ag = [x[l]["agreement"] for x in d]
    f1 = [x[l]["char_f1"] for x in d]
    flips = sum(1 for a, b in zip(u8, u16) if (a > 0) != (b > 0))
    print(f"{l:>5} {st.mean(u8):>+9.3f} +-{st.stdev(u8):.3f} "
          f"{st.mean(u16):>+9.3f} +-{st.stdev(u16):.3f} "
          f"{st.mean(ag):>7.3f} {st.mean(f1):>6.3f} {flips:>7}/{len(u8)}")
print("\nA sign flip means the same script aligned to characters in one encoding and")
print("anti-aligned in the other. Chance agreement on 12 windows with ~50 boundaries each")
print("is around 0.25-0.35, so agreement values in that band mean the boundary sets are")
print("essentially unrelated across encodings.")
