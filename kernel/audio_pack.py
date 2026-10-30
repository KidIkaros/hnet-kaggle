"""Pack speech into several byte encodings of the *same samples*.

The audio arm asks the question the text arm could not answer cleanly: is learned chunking
following the signal, or the layout of its serialization? The sharpest available test is
PCM-LE versus PCM-BE -- identical samples, identical byte count, every byte pair swapped. A
chunker that follows the signal puts boundaries at the same byte offsets in both; a chunker
that follows layout cannot.

Everything here is one source signal, decoded once, then re-emitted:

    pcm16le.bin   2 B/sample, little-endian          (the reference)
    pcm16be.bin   2 B/sample, big-endian             (same samples, byte order flipped)
    mulaw.bin     1 B/sample, G.711 mu-law           (lossy 8-bit companding)
    f32le.bin     4 B/sample, normalized float32

plus onsets.u32 (word onsets from the corpus transcripts, in samples) as a semantic reference,
and meta.json carrying the byte->sample mapping for each encoding.

Sample rate is 8 kHz: at 16 kHz a 256-byte window is 8 ms, which is barely one pitch period.
At 8 kHz, 256 bytes of mu-law is 32 ms and int16 needs 512 bytes for the same 32 ms, so the
per-encoding sequence length is recorded in meta.json rather than assumed.
"""
import io
import json
import os
import subprocess
import tarfile
import wave

import numpy as np

SRC = "/home/ikaaros/.hermes/cache/scratch/aud"
OUT = os.path.join(SRC, "packed")
SR = 8000
TARGET_S = 45 * 60


def mu_law_encode(x):
    """G.711 mu-law, standard formulation, input int16."""
    mu = 255.0
    xf = x.astype(np.float64) / 32768.0
    sign = np.sign(xf)
    mag = np.abs(xf)
    y = sign * np.log1p(mu * mag) / np.log1p(mu)
    # 8-bit code, sign bit included, so -128..127
    return np.clip(np.round(y * 127.0), -128, 127).astype(np.int8)


def mu_law_decode(c):
    mu = 255.0
    y = c.astype(np.float64) / 127.0
    x = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def detect_onsets(pcm, sr=SR, hop_ms=10, win_ms=25, k=2.5):
    """Energy-rise onsets: the acoustic reference for "did a boundary land on an event".

    Corpus transcripts were tried first and rejected -- the shards carry one segment per clip
    in several languages, which is too sparse to support a statistic. Energy rises are computed
    from the signal itself, so they are available for every clip and cannot be missing.
    """
    hop, win = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
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
        pos = i * hop
        if pos - last > int(0.05 * sr):
            onsets.append(int(pos))
            last = pos
    return onsets


def decode_to_pcm(blob, ext):
    """Decode any container to 8 kHz mono s16le via ffmpeg."""
    ff = "/home/ikaaros/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
    p = subprocess.run([ff, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"],
                       input=blob, capture_output=True, timeout=600)
    if p.returncode != 0:
        return None
    return np.frombuffer(p.stdout, dtype="<i2")


def main():
    os.makedirs(OUT, exist_ok=True)
    tars = sorted(f for f in os.listdir(SRC) if f.endswith(".tar"))
    seen, uniq = set(), []
    for f in tars:
        h = os.path.getsize(os.path.join(SRC, f))
        if h in seen:
            print(f"  dedupe {f} (same size as an earlier tar)")
            continue
        seen.add(h)
        uniq.append(f)
    tars = uniq
    print(f"{len(tars)} tars: {tars}")
    chunks, docs, onsets = [], [], []
    total = 0
    per_clip = int(TARGET_S * SR / max(1, len(tars)))
    for tf in tars:
        with tarfile.open(os.path.join(SRC, tf)) as t:
            members = t.getmembers()
            audio = [m for m in members if m.name.lower().endswith(
                (".webm", ".mp3", ".flac", ".wav", ".ogg", ".m4a"))]
            meta = {m.name.rsplit(".", 1)[0]: m for m in members if m.name.endswith(".json")}
            for m in audio:
                pcm = decode_to_pcm(t.extractfile(m).read(), m.name)
                if pcm is None or len(pcm) < SR // 2:
                    print(f"  skip {tf}/{m.name}")
                    continue
                info = {}
                key = m.name.rsplit(".", 1)[0]
                if key in meta:
                    try:
                        info = json.loads(t.extractfile(meta[key]).read().decode("utf-8"))
                    except Exception:
                        info = {}
                # every clip contributes an equal share, so the corpus is not one long speaker
                pcm = pcm[:per_clip]
                start = total
                chunks.append(pcm)
                total += len(pcm)
                docs.append({"tar": tf, "file": m.name, "start_sample": start,
                             "n_samples": int(len(pcm)), "lang": tf.split(".")[0],
                             "dur_s": round(len(pcm) / SR, 3)})
                for o in detect_onsets(pcm):
                    onsets.append(start + o)
                if total / SR >= TARGET_S:
                    break
        print(f"  {tf}: {total/SR:.0f} s cumulative, {len(docs)} docs")

    pcm = np.concatenate(chunks)[:TARGET_S * SR]
    n = len(pcm)
    # the trim can cut the last docs; keep only what survives, clipped to the stream
    kept = []
    for d in docs:
        if d["start_sample"] >= n:
            continue
        d["n_samples"] = min(d["n_samples"], n - d["start_sample"])
        d["dur_s"] = round(d["n_samples"] / SR, 3)
        kept.append(d)
    docs = kept
    print(f"\ntotal {n} samples = {n/SR/60:.1f} min at {SR} Hz, {len(docs)} docs, "
          f"{len(onsets)} word onsets")

    be = pcm.astype(">i2")
    mu = mu_law_encode(pcm)
    f32 = (pcm.astype(np.float32) / 32768.0)

    open(os.path.join(OUT, "pcm16le.bin"), "wb").write(pcm.astype("<i2").tobytes())
    open(os.path.join(OUT, "pcm16be.bin"), "wb").write(be.tobytes())
    open(os.path.join(OUT, "mulaw.bin"), "wb").write(mu.tobytes())
    open(os.path.join(OUT, "f32le.bin"), "wb").write(f32.astype("<f4").tobytes())
    np.array([o for o in onsets if o < n], dtype="<u4").tofile(os.path.join(OUT, "onsets.u32"))

    meta = {"sample_rate": SR, "n_samples": n, "docs": docs,
            "onset_source": "energy-rise, 10 ms hop, 25 ms window, 2.5 sd over mean delta",
            "encodings": {
                "pcm16le": {"bytes_per_sample": 2, "seq_len_for_32ms": 512},
                "pcm16be": {"bytes_per_sample": 2, "seq_len_for_32ms": 512},
                "mulaw":   {"bytes_per_sample": 1, "seq_len_for_32ms": 256},
                "f32le":   {"bytes_per_sample": 4, "seq_len_for_32ms": 1024}},
            "onsets": int(sum(1 for o in onsets if o < n))}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

    # ---- gates -----------------------------------------------------------------
    print("\ngates:")
    a = np.fromfile(os.path.join(OUT, "pcm16le.bin"), dtype="<i2")
    b = np.fromfile(os.path.join(OUT, "pcm16be.bin"), dtype=">i2")
    c = np.fromfile(os.path.join(OUT, "mulaw.bin"), dtype=np.int8)
    d = np.fromfile(os.path.join(OUT, "f32le.bin"), dtype="<f4")
    assert len(a) == len(b) == len(c) == len(d) == n, (len(a), len(b), len(c), len(d), n)
    assert np.array_equal(a, b), "LE and BE must decode to identical samples"
    raw_le = np.fromfile(os.path.join(OUT, "pcm16le.bin"), dtype=np.uint8)
    raw_be = np.fromfile(os.path.join(OUT, "pcm16be.bin"), dtype=np.uint8)
    assert np.array_equal(raw_le.reshape(-1, 2)[:, ::-1].ravel(), raw_be), "BE must be a byte swap"
    assert not np.array_equal(raw_le, raw_be), "byte swap must actually change the bytes"
    assert np.array_equal(np.round(d.astype(np.float64) * 32768).clip(-32768, 32767).astype(np.int16), a), "f32 round-trip"
    err = np.abs(mu_law_decode(c).astype(np.int32) - a.astype(np.int32))
    print(f"  LE == BE decoded:            OK ({n} samples)")
    print(f"  BE is byte-swap of LE:       OK ({100*np.mean(raw_le != raw_be):.1f}% of bytes differ)")
    print(f"  f32 round-trip exact:        OK")
    print(f"  mu-law round-trip:           mean |err| {err.mean():.1f} LSB, "
          f"p99 {np.percentile(err, 99):.0f} LSB, max {err.max()} LSB")
    assert err.mean() < 200, "mu-law round-trip error too large for an 8 kHz speech signal"
    on = np.fromfile(os.path.join(OUT, "onsets.u32"), dtype="<u4")
    print(f"  onsets within range:         OK ({len(on)} onsets, max {on.max()} < {n})")
    assert on.max() < n
    sizes = {k: os.path.getsize(os.path.join(OUT, k)) for k in
             ("pcm16le.bin", "pcm16be.bin", "mulaw.bin", "f32le.bin")}
    print(f"  sizes: {sizes}")
    print(f"  total {sum(sizes.values())/1e6:.1f} MB")


if __name__ == "__main__":
    main()
