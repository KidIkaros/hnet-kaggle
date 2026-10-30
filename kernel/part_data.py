# ===========================================================================
#  dev loop: pack -> train 200 steps -> health curve -> artifacts
# ===========================================================================
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
CORPUS = os.path.join(OUT, "corpus")
SMOKE_CFG = dict(d_model=128, n_enc=2, n_dec=2, n_main=2, d_state=16, scan_chunk=64,
                 seq_len=256, batch=4, lr=1e-3, wd=0.1, warmup=20, clip=1.0,
                 target_ratio=6.0, alpha=0.03, max_steps=200, log_every=10)
_ov = os.environ.get("CFG_OVERRIDE")
if _ov:
    SMOKE_CFG.update(json.loads(_ov))
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 4.0 * 3600))
PACK_MB = float(os.environ.get("PACK_MB", 8))
LANGUAGES = os.environ.get("LANGUAGES", "deu cmn arb").split()

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- 1. data ---
def step_data():
    os.makedirs(OUT, exist_ok=True)
    if glob.glob(os.path.join(CORPUS, "*", "train.bin")):
        log("corpus already packed, reusing")
    else:
        log(f"packing {PACK_MB} MB/lang for {LANGUAGES}")
        for lang in LANGUAGES:
            pack_lang(lang, CORPUS, int(PACK_MB * 1e6), int(0.5 * 1e6))

    # validation: the corpus choice is 17 days old, so read it back and check it
    report = {}
    for lang in LANGUAGES:
        meta = json.load(open(os.path.join(CORPUS, lang, "meta.json")))
        raw = open(os.path.join(CORPUS, lang, "train.bin"), "rb").read()
        text = raw.decode("utf-8")            # raises if the packing split a char
        sample = text[:4000].replace("\n", " ")
        report[lang] = {
            "bytes": len(raw),
            "docs": meta["docs"],
            "lang_score_median": meta["language_score_median"],
            "html_tags": sample.count("<"),
            "truncated_midword": bool(text) and not text.rstrip()[-1:].isalnum(),
            "sample": sample[:180],
        }
        log(f"  {lang}: {len(raw)} bytes {meta['docs']} docs "
            f"lang_score={meta['language_score_median']:.3f} :: {report[lang]['sample'][:90]}")
    with open(os.path.join(OUT, "corpus_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return [os.path.join(CORPUS, l, "train.bin") for l in LANGUAGES]


class Stream:
    """Round-robin byte windows over per-language streams."""

    def __init__(self, rng, batch, seq_len, paths):
        self.rng, self.batch, self.seq_len, self.paths = rng, batch, seq_len, paths
        self.data = [np.memmap(p, dtype=np.uint8, mode="r") for p in paths]
        self.lens = [len(d) for d in self.data]

    def batch_bytes(self):
        out = np.empty((self.batch, self.seq_len + 1), dtype=np.int64)
        for b in range(self.batch):
            i = int(self.rng.integers(len(self.data)))
            off = int(self.rng.integers(max(1, self.lens[i] - self.seq_len - 1)))
            out[b] = np.frombuffer(self.data[i][off:off + self.seq_len + 1], dtype=np.uint8)
        return out


# ------------------------------------------------------------- 2. model ----
