# Vision I: does a byte-level chunker find image structure, or raster phase?

Type: task
Status: open
Blocked by: 21

## Question

The study's central method is now: **does the learned structure track the semantics of the data,
or the layout of its encoding?** In text the answer was layout — a byte phase that happened to
coincide with characters in some scripts. That question is not about text. It transfers to any
modality where the model reads a serialized stream.

Images serialized to bytes are the cleanest next test. A 32x32 RGB image is 3072 bytes; its
"characters" are pixels (3 bytes each) and its "words" are rows (96 bytes). Its semantics are
edges and objects. So:

- **layout structure** = boundaries on pixel starts (offset mod 3), row starts (offset mod 96),
  or a fixed phase within those;
- **semantic structure** = boundaries on high-gradient positions (edges) and object contours.

The byte-level model is already built. This needs no new architecture, only new data and a new
metric.

## Design

- Data: a small image corpus (CIFAR-10 or equivalent, available on Kaggle) serialized to bytes in
  **four encodings of the same images**: row-major RGB, channel-planar RGB, grayscale, and PNG
  bytes. Same pixels, different byte layout.
- Model: the frozen 0.84 M config, 5000 steps, N=4, 2 seeds per encoding.
- Metrics, per encoding: (a) boundary concentration by offset mod 3 and mod 96 (pixel and row
  phase); (b) alignment to edge pixels -- the gradient magnitude of the same image, thresholded,
  as the "semantic" reference; (c) the encoding-invariance comparison: do the four encodings'
  boundary sets agree in *pixel* space?

## Pre-registered prediction

From the text result: **the chunker will lock to a layout phase and will not align to edges**,
and the phase will differ between encodings of the same images. Specifically, row-major RGB
should show concentration at offsets mod 3 (pixel phase), and the four encodings should agree in
pixel space at roughly the same low rate (0.25-0.45) that UTF-8 and UTF-16 agreed in character
space.

**Kill criterion.** If boundaries align to edges above chance in *all four* encodings, then
semantic chunking is reachable in bytes and the text finding is about text rather than about
byte-level chunkers -- which would be the more interesting outcome and would demand a re-read of
the whole study.

## Cost

CPU-feasible (images are small; 5000 steps at 0.84 M took ~3 h per seed on CPU). ~12 h for four
encodings x 2 seeds if run sequentially, or ~6 h with two parallel kernels. No GPU needed.
