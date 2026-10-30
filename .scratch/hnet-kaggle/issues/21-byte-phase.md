# What byte-level structure does the chunker actually learn?

Type: task
Status: resolved
Blocked by: 19

## Question

The invariance test (ticket 19) refuted "the chunker learns characters". The surviving
possibility is that it learns something about the **byte stream**: a phase, a periodicity, a
chunk-length distribution. Two cheap, training-free measurements separate these from each other,
and both reuse checkpoints that already exist.

## The two measurements

**(a) Offset-phase alignment, per script, per encoding.** For a model trained on UTF-8, is the
boundary set concentrated on *even* byte offsets, on odd offsets, or on multiples of some k?
Compute the distribution of boundary offsets modulo 2, 3, 4 for each language and compare it to
the same distribution under a random boundary set. In UTF-8, Russian and Arabic characters are
2 bytes each, so their character starts *coincide with even offsets* — which is exactly what the
"aligns to characters" reading would look like, and also what "aligns to even offsets" would look
like. Chinese characters are 3 bytes, so its character starts are multiples of 3, and the two
hypotheses come apart there.

**(b) Zero-training encoding transfer.** Take the UTF-8 model and feed it the **UTF-16LE bytes
of the same text**, with no retraining. If the chunker is a byte-stream segmenter, it should
place boundaries on some phase of the new byte stream; if it has learned something about
characters, it cannot, because it has never seen this encoding. This costs one forward pass per
window and no training at all.

## Pre-registered predictions

- If the chunker learned **characters**, (a) shows boundary offsets concentrated on character
  starts in *both* encodings, and (b) shows the UTF-8 model failing to align in UTF-16 (it
  already does: -0.228 rus, -0.326 arb). The (b) result is already in, so characters are out.
- If the chunker learned **a byte phase**, (a) shows a consistent non-uniform offset distribution
  in both encodings, concentrated on the encoding's own structure, and (b) shows the UTF-8 model
  still producing boundaries at a *consistent* rate and spacing even on UTF-16 bytes it has never
  seen.
- If the chunker learned **nothing transferable** (a byte-level noise segmenter), (b) shows the
  UTF-8 model's boundary rate collapsing on UTF-16 bytes.

**Kill criterion.** If (a) finds no offset-phase concentration in either encoding beyond chance,
and (b) shows no consistent boundary rate on the unseen encoding, then the honest description is
that the chunker has no stable byte-level structure either, and the study's finding is the
negative one it already has: at this scale the chunker's segmentation is not interpretable, and
no further analysis of the same checkpoints will change that.

## Cost

No training. CPU, minutes. Uses the ten UTF-8 checkpoints and the two UTF-16 checkpoints already
in the dataset.
