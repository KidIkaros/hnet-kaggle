"""Pack images into several byte encodings of the *same pixels*.

The text arm ended with a byte-phase finding. This asks whether that is about text or about
serialized data in general. Images are the cleanest test because their "characters" are pixels
and their "words" are rows, so a layout hypothesis and a semantics hypothesis predict different
boundaries and can be told apart:

    layout    -> boundaries on pixel starts (offset mod 3), or on row starts (offset mod w*3)
    semantics -> boundaries on edges and on object outlines

Four encodings of the same pixels:

    rgb.bin       row-major RGB          3 B/pixel
    bgr.bin       row-major BGR          3 B/pixel, channels permuted per pixel
    planar.bin    channel-planar RGB     3 B/pixel, planes separated
    gray.bin      luminance              1 B/pixel

`rgb` vs `bgr` is the audio arm's LE/BE trick in image form: identical byte count, identical
pixel grid, only the channel order inside each pixel changes. If chunking follows the image,
boundaries land on the same pixel offsets in both; if it follows byte values, they cannot.

Ground truth stored alongside, one byte per pixel per image:

    edges.bin     Sobel gradient magnitude, thresholded at the 90th percentile
    objmask.bin   1 where a segmentation polygon from the source annotations covers the pixel
"""
import io
import json
import os

import fsspec
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

SHARD = ("https://huggingface.co/datasets/athkot/coco-20-synthetic-50k/resolve/main/"
         "data/train-00000-of-00011.parquet")
OUT = "/home/ikaaros/.hermes/cache/scratch/vis/packed"
SIZE = 64
N_IMAGES = 2000


def sobel_edges(gray):
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90)
    return (mag >= thr).astype(np.uint8)


def object_mask(obj, size):
    """Rasterise the segmentation polygons at the target resolution."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    sx, sy = size / obj["w"], size / obj["h"]
    seg = obj.get("segmentation") or []
    for poly in seg:
        if not poly or len(poly) < 6:
            continue
        pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
        d.polygon(pts, fill=1)
    return np.array(m, dtype=np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = fsspec.filesystem("https")
    pf = pq.ParquetFile(fs.open(SHARD, "rb"))
    rgb, bgr, planar, gray = [], [], [], []
    edges, objmask, docs = [], [], []
    taken = 0
    for rg in range(pf.metadata.num_row_groups):
        if taken >= N_IMAGES:
            break
        rows = pf.read_row_group(rg).to_pylist()
        for r in rows:
            if taken >= N_IMAGES:
                break
            try:
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            except Exception:
                continue
            w0, h0 = im.size
            im = im.resize((SIZE, SIZE), Image.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)              # (SIZE, SIZE, 3)
            g = np.asarray(im.convert("L"), dtype=np.uint8)
            rgb.append(a.tobytes())
            bgr.append(a[:, :, ::-1].tobytes())
            planar.append(np.ascontiguousarray(a.transpose(2, 0, 1)).tobytes())
            gray.append(g.tobytes())
            edges.append(sobel_edges(g).tobytes())
            objs = r.get("objects") or {}
            bboxes = objs.get("bbox") or []
            segs = objs.get("segmentation") or []
            spec = {"w": w0, "h": h0, "segmentation": segs, "n_obj": len(bboxes)}
            objmask.append(object_mask(spec, SIZE).tobytes())
            docs.append({"image_id": r["image_id"], "w": w0, "h": h0,
                         "n_obj": len(bboxes), "n_poly": len(segs)})
            taken += 1
    n = taken
    bpp = SIZE * SIZE * 3
    print(f"{n} images at {SIZE}x{SIZE}, {bpp} B/image RGB")
    for name, arr, sz in (("rgb", rgb, bpp), ("bgr", bgr, bpp), ("planar", planar, bpp),
                          ("gray", gray, SIZE * SIZE), ("edges", edges, SIZE * SIZE),
                          ("objmask", objmask, SIZE * SIZE)):
        blob = b"".join(arr)
        assert len(blob) == n * sz, (name, len(blob), n * sz)
        open(os.path.join(OUT, f"{name}.bin"), "wb").write(blob)
        print(f"  {name:>8}.bin {len(blob)/1e6:8.1f} MB  {sz} B/image")

    meta = {"size": SIZE, "n_images": n, "bytes_per_image":
            {"rgb": bpp, "bgr": bpp, "planar": bpp, "gray": SIZE * SIZE},
            "bytes_per_pixel": {"rgb": 3, "bgr": 3, "planar": 3, "gray": 1},
            "row_bytes": {"rgb": SIZE * 3, "bgr": SIZE * 3, "planar": SIZE * 3, "gray": SIZE},
            "docs": docs}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

    # ---- gates -------------------------------------------------------------
    print("\ngates:")
    a = np.fromfile(os.path.join(OUT, "rgb.bin"), np.uint8).reshape(n, SIZE, SIZE, 3)
    b = np.fromfile(os.path.join(OUT, "bgr.bin"), np.uint8).reshape(n, SIZE, SIZE, 3)
    p = np.fromfile(os.path.join(OUT, "planar.bin"), np.uint8).reshape(n, 3, SIZE, SIZE)
    g = np.fromfile(os.path.join(OUT, "gray.bin"), np.uint8).reshape(n, SIZE, SIZE)
    e = np.fromfile(os.path.join(OUT, "edges.bin"), np.uint8).reshape(n, SIZE, SIZE)
    o = np.fromfile(os.path.join(OUT, "objmask.bin"), np.uint8).reshape(n, SIZE, SIZE)
    assert np.array_equal(b, a[:, :, :, ::-1]), "bgr must be rgb with channels reversed"
    assert not np.array_equal(a, b), "the channel permutation must change the bytes"
    assert np.array_equal(p, a.transpose(0, 3, 1, 2)), "planar must be rgb transposed"
    luma = 0.299 * a[:, :, :, 0] + 0.587 * a[:, :, :, 1] + 0.114 * a[:, :, :, 2]
    assert np.allclose(g, luma, atol=1.5), "gray must be the ITU-R 601-2 luma of rgb"
    print(f"  rgb -> bgr is a channel reversal:      OK ({100*np.mean(a != b):.1f}% of bytes differ)")
    print(f"  rgb -> planar is a transpose:          OK")
    print(f"  gray == 601 luma of rgb:               OK (max |diff| {np.abs(g-luma).max():.0f})")
    print(f"  edge density: {e.mean():.3f}   object coverage: {o.mean():.3f}")
    assert 0.05 < e.mean() < 0.25, "edge density outside the expected 90th-percentile band"
    assert 0.05 < o.mean() < 0.95, "object coverage implausible"
    assert (o.sum(axis=(1, 2)) > 0).mean() > 0.9, "most images must have a non-empty object mask"
    print(f"  images with a non-empty object mask:   OK "
          f"({(o.sum(axis=(1,2))>0).mean()*100:.0f}%)")
    print(f"  total {(sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)))/1e6:.1f} MB")


if __name__ == "__main__":
    main()
