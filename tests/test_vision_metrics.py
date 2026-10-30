"""Metric audit for the vision arm.

Three units bugs have already been caught this way, all of them in metrics that returned
plausible zeros. The vision alignment maps a byte offset to (channel, row, column) and then
indexes a 2-D mask; an off-by-one there would silently report a null.

Oracle: boundaries placed exactly on edge pixels must show a large positive lift, boundaries
placed at random must show ~0, and the pixel-start share must be 1.0 for a bmask that only
touches channel 0.
"""
import json
import os

import numpy as np

DATA = "/home/ikaaros/.hermes/cache/scratch/vis/packed"
META = json.load(open(os.path.join(DATA, "meta.json")))
SIZE = META["size"]
NIMG = META["n_images"]
BPP = META["bytes_per_pixel"]
ROW = META["row_bytes"]
EDGES = np.fromfile(os.path.join(DATA, "edges.bin"), np.uint8).reshape(NIMG, SIZE, SIZE)
OBJ = np.fromfile(os.path.join(DATA, "objmask.bin"), np.uint8).reshape(NIMG, SIZE, SIZE)


def decompose(arm, pos):
    bpp, row = BPP[arm], ROW[arm]
    y = (pos // row) % SIZE
    x = (pos % row) // bpp
    c = pos % bpp
    return c, y, x


def alignment(arm, bmask, base, img_idx):
    pos = np.flatnonzero(bmask)
    if len(pos) < 8:
        return None
    c, y, x = decompose(arm, pos + base)
    e = EDGES[img_idx].ravel()[y * SIZE + x].astype(bool)
    o = OBJ[img_idx].ravel()[y * SIZE + x].astype(bool)
    allpos = np.arange(base, base + len(bmask))
    c2, y2, x2 = decompose(arm, allpos)
    e2 = EDGES[img_idx].ravel()[y2 * SIZE + x2].astype(bool)
    o2 = OBJ[img_idx].ravel()[y2 * SIZE + x2].astype(bool)
    bpp = BPP[arm]
    return {"edge_lift": float(e.mean() - e2.mean()),
            "obj_lift": float(o.mean() - o2.mean()),
            "pixel_start": float(np.mean(c == 0)) if bpp > 1 else None,
            "row_start": float(np.mean(x == 0))}


def main():
    arm, seq = "rgb", 4096        # ~21 rows: a 256-byte window holds too few edge pixels
    img = 1500
    base = img * META["bytes_per_image"][arm]

    # sanity on the mapping itself
    c, y, x = decompose(arm, np.array([0, 1, 2, 3, 191, 192, 12287]))
    print("decompose(0,1,2,3,191,192,12287) ->")
    print("  c:", c.tolist(), " y:", y.tolist(), " x:", x.tolist())
    # 191 = 63*3 + 2 -> channel 2 of pixel 63; the first version of this literal had 0 there
    assert c.tolist() == [0, 1, 2, 0, 2, 0, 2]
    assert y.tolist() == [0, 0, 0, 0, 0, 1, 63]
    assert x.tolist() == [0, 0, 0, 1, 63, 0, 63]

    # oracle: boundaries on edge pixels (channel 0 only)
    ey, ex = np.nonzero(EDGES[img])
    bmask = np.zeros(seq, bool)
    for yy, xx in zip(ey, ex):
        j = yy * ROW[arm] + xx * BPP[arm]
        if 0 <= j < seq:
            bmask[j] = True
    r = alignment(arm, bmask, base, img)
    print(f"\noracle on edges:      edge_lift {r['edge_lift']:+.3f}  pixel_start {r['pixel_start']:.2f}")

    # oracle: boundaries on object pixels, windowed where the objects actually are
    oy, ox = np.nonzero(OBJ[img])
    y0 = int(oy.min())
    base2 = img * META["bytes_per_image"][arm] + y0 * ROW[arm]
    bm2 = np.zeros(seq, bool)
    for yy, xx in zip(oy, ox):
        j = (yy - y0) * ROW[arm] + xx * BPP[arm]
        if 0 <= j < seq:
            bm2[j] = True
    r2 = alignment(arm, bm2, base2, img)
    print(f"oracle on objects:    obj_lift  {r2['obj_lift']:+.3f}  pixel_start {r2['pixel_start']:.2f}")

    # random, same rate
    rng = np.random.default_rng(0)
    lifts, olifts = [], []
    for t in range(40):
        bm = np.zeros(seq, bool)
        bm[rng.choice(seq, size=max(8, int(bmask.sum())), replace=False)] = True
        rr = alignment(arm, bm, base, img)
        lifts.append(rr["edge_lift"])
        olifts.append(rr["obj_lift"])
    print(f"random, same rate:    edge_lift {np.mean(lifts):+.3f}  obj_lift {np.mean(olifts):+.3f}")

    # a channel-1-only mask must read pixel_start == 0
    bm3 = np.zeros(seq, bool)
    bm3[1::3] = True
    r3 = alignment(arm, bm3, base, img)
    print(f"channel-1 only:       pixel_start {r3['pixel_start']:.2f} (must be 0.00)")

    assert r["edge_lift"] > 0.5, "an on-edge oracle must score a large positive lift"
    assert abs(np.mean(lifts)) < 0.08, "random boundaries must score ~0"
    assert r["pixel_start"] > 0.99 and r3["pixel_start"] < 0.01
    print("\nPASS: the metric separates an on-edge chunker from a random one, and the")
    print("      channel/row/column mapping is correct.")


if __name__ == "__main__":
    main()
