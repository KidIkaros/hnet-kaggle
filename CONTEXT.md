# H-Net

Building a tokenizer-free hierarchical language model: a model that reads raw bytes, decides its
own segmentation, and is usable as a drop-in replacement for a tokenized model.

## The model

**Chunk**:
A contiguous span of bytes the router emits as one unit for the main network. The H-Net has no
vocabulary and no tokenizer; chunks are not tokens.
_Avoid_: token, wordpiece, subword

**Tokenizer-free**:
The model consumes raw bytes directly and produces raw bytes directly. No fixed vocabulary
exists anywhere in it.
_Avoid_: byte-level (implying only the input is bytes), raw-mode

**Byte phase**:
A fixed alignment between chunk boundaries and the byte grid, chosen by the router from byte
values rather than from content. Our measured finding is that the router settles on a byte
phase, and that the phase changes when the same text is re-encoded.
_Avoid_: chunk alignment, grid phase

**Content-aligned chunking**:
Chunk boundaries that land on characters, words, edges or other meaning-bearing positions. The
opposite of a byte phase; the property a learned tokenizer is supposed to have.
_Avoid_: semantic chunking, aligned chunking

**Compression ratio (N)**:
The target number of bytes per chunk the model is asked to hold. With N = 6 the model aims for
six bytes per chunk.
_Avoid_: compression rate, chunk size, BPIC (a measured quantity, not the target)

**Selected fraction (F)**:
The fraction of byte positions the router actually turns into chunk starts.
_Avoid_: sparsity, selection rate

**Boundary probability (G)**:
The mean router probability over positions. F and G both settle to 1/N in a healthy run.
_Avoid_: routing rate

**Ratio loss**:
The auxiliary loss that penalises deviation of F and G from 1/N. It is what makes the
compression target hold rather than drift.
_Avoid_: compression loss, chunk loss

**Settling**:
The compression dynamics reaching 1/N and staying there, with the loss monotone afterwards. A run
that never settles is not a model, whatever its other statistics look like.
_Avoid_: convergence (overloaded with the language loss)

## Measurement

**bpb**:
Bits per byte on held-out text. The only loss unit comparable across tokenizations, and the
unit every published comparison uses.
_Avoid_: loss, perplexity (per-token quantities are not comparable across tokenizers)

**FLOPs budget**:
The compute allowance for a run, counted analytically and charged per stage at the resolution
that stage runs at. The x-axis of every efficiency comparison.
_Avoid_: compute budget, token budget

**Compute multiplier**:
The ratio of compute a baseline needs to reach a given bpb, relative to the H-Net reaching it.
_Avoid_: speedup, efficiency gain

## The deliverable

**Workable model**:
A trained model that a person could actually use: it trains stably, its quality is verifiable
against published numbers, and it generates usable text.
_Avoid_: working model, shippable model

**Token replacement**:
The product shape: a byte-level model that can stand in for a tokenized model in someone's
pipeline, without them having to accept a vocabulary.
_Avoid_: tokenizer replacement, drop-in (drop-in describes the interface, not the intent)
