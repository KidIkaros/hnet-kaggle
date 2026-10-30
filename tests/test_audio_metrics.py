"""Metric audit: can onset_lift detect a chunker that really does land on onsets?

Two metric bugs have already cost this project real runs (an off-by-one in the alignment probe,
and a continuation-byte numbering error in the offset analysis). So before the audio arm spends
four CPU-hours, the onset metric is pointed at a synthetic oracle: boundaries placed exactly on
onsets must score a large positive lift, and boundaries placed at random must score ~0.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ikaaros/Coding/kaggle/hnet/kernel_audio")
DATA = "/home/ikaaros/.hermes/cache/scratch/aud/packed"
META = json.load(open(os.path.join(DATA, "meta.json")))
SR, N = META["sample_rate"], META["n_samples"]
_onsets = np.fromfile(os.path.join(DATA, "onsets.u32"), dtype="<u4").astype(np.int64)
ONSETS = np.sort(_onsets[_onsets < N])
BPS = {k: v["bytes_per_sample"] for k, v in META["encodings"].items()}


def onset_lift(bmask, base_byte, arm, tol_ms=8):
    """Copy of the kernel implementation, kept in sync by hand."""
    bps = BPS[arm]
    tol = int(tol_ms * SR / 1000)          # tolerance in SAMPLES, like ONSETS
    pos = (np.flatnonzero(bmask) + base_byte) // bps   # byte offset -> sample index
    if len(pos) == 0:
        return None
    idx = np.searchsorted(ONSETS, pos)
    lo = np.clip(idx - 1, 0, len(ONSETS) - 1)
    hi = np.clip(idx, 0, len(ONSETS) - 1)
    d = np.minimum(np.abs(pos - ONSETS[lo]), np.abs(ONSETS[hi] - pos))
    hit = float(np.mean(d <= tol))
    allpos = (np.arange(base_byte, base_byte + len(bmask), bps)) // bps
    i2 = np.searchsorted(ONSETS, allpos)
    l2 = np.clip(i2 - 1, 0, len(ONSETS) - 1)
    h2 = np.clip(i2, 0, len(ONSETS) - 1)
    d2 = np.minimum(np.abs(allpos - ONSETS[l2]), np.abs(ONSETS[h2] - allpos))
    base = float(np.mean(d2 <= tol))
    return {"hit": hit, "base": base, "lift": hit - base}


def main():
    arm, seq = "pcm16le", 32768      # 2 s of audio: a 64 ms window holds too few onsets
    bps = BPS[arm]
    win_start = int(N * 0.5) * bps        # N is samples; byte offsets scale by bps
    win_start -= win_start % bps
    n_windows = 12

    # oracle: every onset inside the window, snapped to the sample grid
    tot_or, tot_rand = [], []
    for w in range(n_windows):
        b = win_start + w * seq
        b -= b % bps
        lo_s, hi_s = b // bps, (b + seq) // bps     # window bounds in SAMPLES
        inside = ONSETS[(ONSETS >= lo_s) & (ONSETS < hi_s)]
        bmask = np.zeros(seq, bool)
        for o in inside:
            j = int(o - lo_s) * bps          # sample offset -> byte offset inside the window
            if 0 <= j < seq:
                bmask[j] = True
        if bmask.sum() >= 3:
            r = onset_lift(bmask, b, arm)
            if r:
                tot_or.append(r["lift"])
        # random boundaries at the same rate
        rng = np.random.default_rng(w)
        n = max(3, int(bmask.sum()))
        rnd = np.zeros(seq, bool)
        rnd[rng.choice(np.arange(0, seq, bps), size=n, replace=False)] = True
        r = onset_lift(rnd, b, arm)
        if r:
            tot_rand.append(r["lift"])

    print(f"oracle (boundaries on onsets): lift {np.mean(tot_or):+.3f} "
          f"over {len(tot_or)} windows")
    print(f"random (same rate):            lift {np.mean(tot_rand):+.3f} "
          f"over {len(tot_rand)} windows")
    print(f"base rate of on-onset positions: {onset_lift(np.ones(seq, bool), win_start, arm)['base']:.3f}")
    assert np.mean(tot_or) > 0.4, "oracle must score a large positive lift"
    assert abs(np.mean(tot_rand)) < 0.12, "random boundaries must score ~0"
    print("PASS: the metric separates a real on-onset chunker from a random one")

    # second check: the LE/BE jaccard machinery, on identical synthetic masks
    le = np.zeros(seq, bool); le[::17] = True
    be = le.copy()
    inter, union = float((le & be).sum()), float((le | be).sum())
    print(f"identical masks jaccard {inter/union:.3f} (must be 1.000)")
    assert abs(inter / union - 1.0) < 1e-9
    be2 = np.zeros(seq, bool); be2[3::17] = True
    print(f"shifted masks  jaccard {float((le & be2).sum())/float((le | be2).sum()):.3f} (must be 0)")


if __name__ == "__main__":
    main()
