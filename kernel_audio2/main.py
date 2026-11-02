"""Pack LibriSpeech train-clean-100 into the English-corpus byte format (CPU, internet).

Phase 0 of docs/AUDIO_PHASE.md: 100 h of speech -> raw 16 kHz s16le bytes, split into
train/val by document, in the exact format the English scaling harness reads:

    train.bin / val.bin        raw byte stream (s16le mono PCM)
    train_docs.u32             LE u32 start offsets (bytes) of each document
    train_lens.u32             LE u32 lengths (bytes) of each document
    val_docs.u32 / val_lens.u32
    onsets.u64                 LE u64 global sample positions of energy-rise onsets
    meta.json

One LibriSpeech utterance = one document. Val split by md5(utterance id) % 50 == 0 (~2%).
16 kHz per AUDIO_PHASE.md (onset metric needs the temporal resolution; the 8 kHz
comparability arm can be derived later in another free CPU pass).

Session management rules (postmortem 2026-11-01):
  - hard watchdog: a daemon thread os._exit()s at DEADLINE_S even if the main thread is
    stuck inside a C call (signal.alarm cannot fire there);
  - progress line every clip batch, flushed, so a hung run is visible in the log.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

import numpy as np

OUT = "/kaggle/working"
SR = 16000
TAR_URLS = [
    "https://us.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
]
DEADLINE_S = 4 * 3600
T0 = time.time()


def _watchdog():
    time.sleep(DEADLINE_S)
    print(f"WATCHDOG: hard exit after {DEADLINE_S}s", flush=True)
    os._exit(90)


threading.Thread(target=_watchdog, daemon=True).start()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def fetch_tar(dest):
    if os.path.exists(dest):
        log("tar already present:", dest)
        return
    for url in TAR_URLS:
        try:
            log("downloading", url)
            t = time.time()
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
            log(f"downloaded in {(time.time() - t) / 60:.1f} min, "
                f"{os.path.getsize(dest) / 1e9:.2f} GB")
            return
        except Exception as e:  # try next mirror
            log("  failed:", repr(e))
    raise RuntimeError("all mirrors failed")


def ffmpeg_bin():
    p = shutil.which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg  # pip fallback binary
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_to_pcm16k(blob, ff):
    p = subprocess.run([ff, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=blob, capture_output=True,
                       timeout=600)
    if p.returncode != 0:
        return None
    return np.frombuffer(p.stdout, dtype="<i2")


def detect_onsets(pcm, hop_ms=10, win_ms=25, k=2.5):
    hop, win = int(SR * hop_ms / 1000), int(SR * win_ms / 1000)
    if len(pcm) < win + hop:
        return []
    nf = 1 + (len(pcm) - win) // hop
    idx = np.arange(nf)[:, None] * hop + np.arange(win)[None, :]
    frames = pcm[idx].astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
    db = 20 * np.log10(rms + 1e-9)
    d = np.diff(db, prepend=db[0])
    thr = d.mean() + k * d.std()
    onsets, last = [], -10 ** 9
    for i in np.flatnonzero(d > thr):
        pos = int(i * hop)
        if pos - last > int(0.05 * SR):
            onsets.append(pos)
            last = pos
    return onsets


def main():
    os.makedirs(OUT, exist_ok=True)
    ff = ffmpeg_bin()
    log("ffmpeg:", ff)

    fetch_tar(os.path.join(OUT, "train-clean-100.tar.gz"))
    # keep the tar OUT of the kernel output (6.3 GB of the 20 GB budget)
    work = "/tmp/ls"
    os.makedirs(work, exist_ok=True)
    tar_path = os.path.join(OUT, "train-clean-100.tar.gz")
    os.rename(tar_path, os.path.join(work, "train-clean-100.tar.gz"))

    splits = {}
    for name in ("train", "val"):
        splits[name] = {"f": open(os.path.join(OUT, f"{name}.bin"), "wb"),
                        "docs": [], "lens": []}
    onsets = []
    n_docs = n_utt = 0
    total = 0

    with tarfile.open(os.path.join(work, "train-clean-100.tar.gz")) as t:
        # stream members in tar order; flacs and their transcripts are grouped per chapter
        for m in t:
            if time.time() - T0 > DEADLINE_S - 120:
                log("approaching deadline, stopping ingest at", n_utt, "utterances")
                break
            if not m.name.lower().endswith(".flac"):
                continue
            uid = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            pcm = decode_to_pcm16k(t.extractfile(m).read(), ff)
            if pcm is None or len(pcm) < SR // 2:
                log("  skip", m.name)
                continue
            name = "val" if hashlib.md5(uid.encode()).hexdigest()[-1] in "0" else "train"
            # md5%10==0 -> 10%; tighten to ~10% of speakers' utterances for a usable val
            s = splits[name]
            pos = total
            b = pcm.tobytes()
            s["f"].write(b)
            s["docs"].append(pos)
            s["lens"].append(len(b))
            total += len(b)
            n_utt += 1
            for o in detect_onsets(pcm):
                onsets.append(pos // 2 + o)  # global sample index
            if n_utt % 2000 == 0:
                log(f"{n_utt} utterances, {total / 1e9:.2f} GB, "
                    f"train {len(splits['train']['docs'])} / val {len(splits['val']['docs'])}")
    for s in splits.values():
        s["f"].close()

    n_samples = total // 2
    for name in ("train", "val"):
        np.array(splits[name]["docs"], dtype="<u4").tofile(
            os.path.join(OUT, f"{name}_docs.u32"))
        np.array(splits[name]["lens"], dtype="<u4").tofile(
            os.path.join(OUT, f"{name}_lens.u32"))
    np.array(onsets, dtype="<u8").tofile(os.path.join(OUT, "onsets.u64"))

    # ---- gates -----------------------------------------------------------------
    log("gates:")
    t_docs, t_lens = splits["train"]["docs"], splits["train"]["lens"]
    v_docs, v_lens = splits["val"]["docs"], splits["val"]["lens"]
    assert sum(t_lens) + sum(v_lens) == total, "doc lengths must tile the stream"
    assert t_docs == sorted(t_docs) and v_docs == sorted(v_docs), "offsets must ascend"
    # no gaps or overlaps within each split's own span
    for docs, lens in ((t_docs, t_lens), (v_docs, v_lens)):
        for a, la, b in zip(docs, lens, docs[1:]):
            assert a + la <= b, (a, la, b)
    on = np.fromfile(os.path.join(OUT, "onsets.u64"), dtype="<u8")
    assert on.max() < n_samples
    tr_h = sum(t_lens) / 2 / SR / 3600
    va_h = sum(v_lens) / 2 / SR / 3600
    log(f"  stream tiles exactly, offsets ascend, no overlap: OK")
    log(f"  train {len(t_docs)} docs / {tr_h:.1f} h, val {len(v_docs)} docs / {va_h:.1f} h")
    log(f"  onsets {len(on)}, max {on.max()} < {n_samples}: OK")

    meta = {"sample_rate": SR, "bytes_per_sample": 2, "source": "LibriSpeech train-clean-100",
            "licence": "CC-BY-4.0", "n_samples": n_samples,
            "train_docs": len(t_docs), "val_docs": len(v_docs),
            "train_bytes": sum(t_lens), "val_bytes": sum(v_lens),
            "onsets": len(on), "onset_rule": "energy-rise 10ms hop 25ms win 2.5sd",
            "format": "train.bin + *_docs.u32 (LE u32 byte offsets) + *_lens.u32; onsets.u64 "
                      "= global sample index",
            "val_split_rule": "md5(utt_id)%10==0", "packer": "kernel_audio2/main.py"}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    log("meta.json written")
    for f in sorted(os.listdir(OUT)):
        log(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) / 1e6:.1f} MB")
    log("DONE")


if __name__ == "__main__":
    sys.exit(main())
