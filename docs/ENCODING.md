# Input encoding: does it matter, and how?

Provoked by two comparison tables supplied from another agent (UTF-8 / StreamVByte /
PEP 393 / BOCU-1 / SCSU / FSST, then BOCU-1 / SCSU / FSST). Both tables rank formats by
**processing speed and density**. This note separates those two axes and checks both
against measurement and prior art. Written 2026-10-29.

## Verdict in one line

**Speed is irrelevant to this experiment. Density is real, and it is already a 2026
research area — but for tokenizers, not for learned chunkers, which is our gap.**

---

## 1. The speed axis is measuring a non-bottleneck

Measured numbers for the fastest UTF-8 handling available (Lemire, simdutf, 2026):

| operation | throughput |
|---|---|
| UTF-8 validation, AVX-512, realistic document | **36 GB/s** (14 GB/s on AVX2) |
| UTF-8 → UTF-16 transcode, SIMD | 7.0 GB/s |
| UTF-8 → UTF-16 transcode, scalar loop | 1.7 GB/s |

Our model, measured on a T4 in the `mbdfa` project: **training consumes 14 kB/s**. Forward
pass: 93 kB/s.

So even the worst-case scalar decoder is ~120,000x faster than the model that consumes its
output, and the false premise is starkest at the top: the fastest format in the table
("StreamVByte, >25 GB/s") does not beat UTF-8 *validation*, which is already 36 GB/s.

Arithmetic: 1 GB of text through a scalar decoder is 0.6 s; through SIMD, 0.03 s. Training
on that same GB takes ~20 GPU-hours. The pipeline is 0.001% of the wall clock.

Two corrections to the table's numbers specifically:

- "UTF-8 moderate (3-12 GB/s)" understates it — that figure is roughly right for *scalar*
  transcoding, but validation (the only thing a byte-level model actually needs) is 36 GB/s.
- "StreamVByte >25 GB/s" is real but measured on integers in a different workload. It is not
  a character encoding: it encodes *integer* streams. To apply it to text you first map
  characters to IDs, which requires a vocabulary — i.e. it is a tokenizer wearing a
  byte-encoding costume, and it stacks with the tokenizer question rather than replacing it.

## 2. The density axis is real, and it is a compute tax

UTF-8 spends **3 bytes per character** on CJK and **1 byte** on ASCII. For a byte-level
model that is a 3x sequence-length tax per Chinese character — and Chinese is exactly the
regime our claim lives in. This is not speculation; 2026 papers name it:

- **UBE, NeurIPS 2026** (arXiv:2610.01984, 1 Oct 2026) calls it the *encoding floor*: "when
  no learned merges can be applied, a multibyte character requires multiple byte-derived
  symbols". Their fix: keep 1-2-byte UTF-8 characters on a UTF-8 path and route 3-4-byte
  characters through UTF-16, so 3-byte BMP scripts (CJK, Devanagari, Thai) cost 2 symbols
  instead of 3. Exact decoding preserved; LM quality matched; token counts fall most for
  high-premium scripts and cross-lingual token-budget disparity narrows.
- **BBPE16, ICASSP 2026** (arXiv:2602.01717): UTF-16-based BBPE, −10.4% Chinese token counts,
  −10.3% decoding iterations, comparable or better accuracy.
- **Bit-level BPE** (arXiv:2506.07541): the same redundancy attacked *below* the byte
  boundary, losslessly.

Both UBE and BBPE16 target **tokenizers**. Neither tests a learned dynamic chunker. That is
the gap this project could stand in, and it is a much sharper question than alignment
statistics alone (see §4).

## 3. Why each proposed format is awkward for a byte-level model

| format | what it actually is | problem for us |
|---|---|---|
| **UTF-8** | interchange encoding, stateless, 1 byte for ASCII | 3 bytes/CJK — the whole issue — but universal, stateless, and what every published H-Net result uses |
| **StreamVByte** | varint scheme for **integers** | needs a char→ID map first; density claim (1.25 B/char ASCII) is wrong for text, where ASCII is already 1 |
| **PEP 393 / 1-2-4** | CPython **in-memory** string layout | not a byte stream; to feed a model you pick a width, which is UTF-32 (4 B/char, worse for CJK). Its 2-byte case is just UTF-16 |
| **BOCU-1** | *stateful* delta against previous code point | see below — the statefulness is the killer |
| **SCSU** | *stateful* window shifting | same, plus effectively deprecated |
| **FSST** | per-block learned symbol table (1-8 byte strings) | not a character encoding; symbols have no cross-block semantics |

**The statefulness argument against BOCU-1 and SCSU is the important one.** Under both, a
character's bytes depend on the character before it. For an autoregressive model that is
causally fine in the narrow sense, but it destroys the property that makes byte-level
modeling attractive: a fixed 256-symbol alphabet where a given byte means a stable thing.
It moves the tokenizer's context-dependence *into the encoding* — and context-dependent
compression is precisely what H-Net is trying to **learn**. You would be pre-solving, by
hand, the thing under study. Both also break random access and error recovery, and
**our UTF-8 DFA alignment tool would need rebuilding**.

Their density claims are also better than the modeling case warrants: BOCU-1's "30-50% CJK
reduction" and SCSU's "roughly 1 byte/char for Russian/Arabic/Asian" are real compression
figures, but they come with the statefulness cost, and neither is recommended for
interchange any more. FSST's "2-3x better than UTF-8" is a columnar-database figure for
repetitive data, not natural prose.

## 4. The experiment the caveat actually opens: encoding invariance

Better than "alignment tracks tokenizer fit", because it becomes causal:

> Encode the *same* text two ways — UTF-8 (CJK at 3 bytes) and UTF-16LE (CJK at 2 bytes,
> ASCII at 2). Byte positions differ; character positions are identical. A chunker that
> learns **characters** must place boundaries at the same characters in both. A chunker that
> learns **this encoding** will follow the bytes and drift.

That is a direct test of the paper's own thesis — meaning vs bytes — it costs one extra arm
on the corpus we already have, and it is the kind of test that could *falsify* end-to-end
chunking rather than flatter it. It also serves as the strongest available check on the
alignment metric: if boundaries do not track characters across encodings, the alignment
number was measuring an encoding artefact.

Two caveats to carry into the design:

- **The metric must change.** Bits-per-byte is encoding-relative. Cross-encoding comparisons
  need **bits per character**, with bpb reported alongside for comparability to the paper.
- **UTF-16LE is the honest choice**, not BOCU-1 or UBE. It is standard, stateless, decodable
  by everything, and its 2-symbol floor for CJK is the same win UBE reports. UBE's routing is
  a further refinement we can borrow later.

## 5. What we do

**Keep UTF-8 as the primary.** Every published H-Net result is UTF-8; the paper is UTF-8;
main-horse is UTF-8; our DFA tooling is UTF-8. Switching the primary buys a speed win that
does not exist and costs comparability.

**Add encoding as a possible second factor**, decided after Q1: one extra arm on UTF-16LE,
scored in bits/character, testing invariance. If Q1 shows the chunker engaging, this is
probably the single most valuable follow-on available to us.

**Do not adopt BOCU-1, SCSU, FSST or StreamVByte** for the primary stream. Revisit none of
them before the invariance test says the chunker tracks characters at all.

## Sources

- Kim, Jung. *Universal Byte-Level Encoding: UTF-8/UTF-16 Routing…* NeurIPS 2026.
  arXiv:2610.01984.
- Kim et al. *BBPE16: UTF-16-based byte-level BPE…* ICASSP 2026. arXiv:2602.01717.
- Moon, Hiraoka, Okazaki. *Bit-level BPE: Below the byte boundary.* arXiv:2506.07541.
- Lemire. simdutf benchmarks: 36 GB/s validation (AVX-512), 7 GB/s UTF-8→UTF-16 SIMD vs
  1.7 GB/s scalar, `lemire.me/blog/2026/09/30/…`.
- Local: `mbdfa/RESULTS.md` (14.1 kB/s training, 92.9 kB/s forward, T4).
