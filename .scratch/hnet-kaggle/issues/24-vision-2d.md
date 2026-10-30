# Vision II: a 2D H-Net -- objects, or the raster grid?

Type: task
Status: open
Blocked by: 23

## Question

Cartesia's vision claim is that a multi-dimensional H-Net replaces the backbone of a pixel-space
diffusion model, "uncovers unsupervised semantic concepts directly from raw data", and produces
chunks that "need not be spatially contiguous or homogeneous at the pixel level" -- routinely
grouping disjointed background regions into one chunk, and apparently inferring 3D structure from
flat 2D projections.

That is a strong claim about *semantic* chunking in 2D. The text result here says the 1D chunker
learns a layout phase, not semantics. Is 2D different? The 2D case has no serialization to latch
onto -- there is no "byte phase" in a 2D patch grid -- which is exactly why it might be.

## Design

- A 2D router over patch embeddings (cosine between neighbouring patches in both axes, the direct
  analogue of the 1D rule), 2D scatter/gather for chunking, and a main network that attends over
  the chunk *set* rather than a sequence, since chunks need not be contiguous.
- Data: synthetic images with **known** object masks (shapes on backgrounds, where the ground-truth
  segmentation is exact) plus a real set for qualitative comparison.
- Metrics: (a) chunk-purity against the ground-truth mask (are chunks within one object?);
  (b) spatial contiguity of chunks (Cartesia says they need not be contiguous -- measure how often
  they are not); (c) a random-partition baseline matched for chunk count and size.

## Pre-registered prediction

If 2D chunking is semantic, chunk purity should exceed the matched random partition by a wide
margin and chunks should concentrate on object interiors and boundaries. If 2D chunking is also a
layout artifact, purity will sit near the random baseline and chunk shapes will be
grid-structured.

**Kill criterion.** Chunk purity within 1 sd of the matched random baseline on synthetic data with
unambiguous objects means the 2D chunker is not finding objects either, and the "discovers
semantic concepts" claim does not hold at this scale.

## Cost

The 2D implementation is the real work: router, 2D scatter, set-attention main net, dechunk. ~1
day of engineering, then CPU-feasible at small image sizes. This is the most expensive item on
the roadmap and the most directly comparable to a published claim.
