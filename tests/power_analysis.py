"""Power analysis for the seed replication, from the numbers already measured.

Across-run SDs are inflated by mixture variation; the same-mixture seed pair (Q1 seeds
0 and 1, N=6) is the honest estimate of seed noise. Both are reported, because the
gap between them is itself a finding.
"""
import statistics as st

# same mixture (Q1, 7-lang), two seeds, N=6, full-val lifts
same_mix = {"rus": [0.269, 0.233], "arb": [0.382, 0.312],
            "cmn": [-0.162, -0.127], "jpn": [-0.234, -0.003],
            "deu": [0.009, 0.011], "eng": [-0.003, 0.003]}

# across runs at N=4 (mixtures differ) -- upper bound on noise
across_n4 = {"rus": [0.301, 0.270, 0.162, 0.211],
             "cmn": [-0.206, -0.022, -0.225],
             "jpn": [-0.280, -0.023, -0.096],
             "hin": [0.229, -0.037], "arb": [0.233, 0.282]}

BPC = {"eng": 1.00, "deu": 1.02, "rus": 1.80, "arb": 1.78, "hin": 2.53,
       "cmn": 2.67, "jpn": 2.68}


def power(sd, n, effect):
    """n seeds, 95% CI half-width at sd; can it resolve |effect|?"""
    hw = 1.96 * sd / (n ** 0.5)
    return hw, abs(effect) > hw


print("language  B/char   seed-SD(n=2)  mean   CI half-width @4 seeds  resolves?")
for l, v in same_mix.items():
    sd = st.stdev(v) if len(v) > 1 else 0.0
    mean = st.mean(v)
    hw, ok = power(sd, 4, mean)
    print(f"{l:>8}  {BPC.get(l, 0):>5.2f}   {sd:>10.3f}  {mean:>+6.3f}   "
          f"{hw:>12.3f}          {'YES' if ok else 'no'}")

print("\nacross-run SD at N=4 (mixtures differ) and the seeds needed for half-width 0.05")
for l, v in across_n4.items():
    sd = st.stdev(v) if len(v) > 1 else 0.0
    n_needed = (1.96 * sd / 0.05) ** 2
    print(f"{l:>8}  B/char {BPC.get(l, 0):>4.2f}  sd {sd:>6.3f}  "
          f"mean {st.mean(v):+.3f}  seeds for hw=0.05: {n_needed:>6.1f}")

print("\nwidth hypothesis: alignment vs bytes/char, pooled over all measured runs")
pts = []
for l, v in across_n4.items():
    pts.append((BPC[l], st.mean(v)))
pts.sort()
for b, a in pts:
    print(f"  {b:>4.2f} B/char -> {a:+.3f}")
