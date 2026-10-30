# Alignment metric and DFA stream

Type: research
Status: resolved
Blocked by: 01

## Question

What exactly is "boundary-to-character alignment", and what does it take to compute it on
the new corpus?

Two sub-questions:

1. **The metric.** Given a byte position `t` and a boundary indicator `b_t`, "does this
   boundary fall on a character start" needs a definition that is not gamed by a model that
   simply puts a boundary everywhere (which trivially scores 100% on lead bytes). Candidates:
   precision/recall against the set of UTF-8 lead-byte positions; a lift over the base rate
   of lead bytes in that script (Chinese: 1 in 3 bytes is a lead byte); alignment conditioned
   on realised compression. The ticket must pick one, justify it, and state the chance level
   per script so the number is interpretable.
2. **The stream.** `mbdfa/src/build_corpus.py` builds `*_state.u8` — the UTF-8 DFA state per
   byte — for the Tatoeba corpus. Confirm the same builder runs on the new packed corpus, and
   that the 8-state compression in that file is the right granularity for this measurement
   (a "lead byte" question may only need `state == 0`, i.e. the ASCII/start state).

Also settle: how alignment is tracked **over training** (per-checkpoint curve, not just
final), and whether the English word-alignment metric (±1 byte of a space) belongs in the
same table.

## Answer

**Metric: ΔF1 against a count-matched random boundary set.**

Precision and recall of boundaries on UTF-8 lead-byte positions, the F1 of those, minus the
F1 that a *random* boundary set of the same size achieves on the same text. Positive means
better than chance; near zero means the apparent alignment is a density artefact.

The subtraction is mandatory, and one adversarial case shows why: on Chinese, 1 byte in 3 is a
lead byte, so a chunker that drops a boundary every third byte scores ~100% precision having
learned nothing. Every raw "X% of boundaries fall on character starts" figure is
uninterpretable without that script's chance level, and a count-matched shuffle supplies it
empirically without an analytic per-script baseline.

Per script, per checkpoint:

| field | definition |
|---|---|
| `lead_precision` | P(lead byte \| boundary) |
| `lead_recall` | P(boundary \| lead byte) |
| `F1` | harmonic mean, with realised boundary count |
| `chance_F1` | F1 of a count-matched random set on the same text, 5 draws |
| **`delta_F1`** | `F1 − mean(chance_F1)` — the headline |
| `word_precision` (Latin only) | P(boundary within ±1 byte of a space) |

`chance_F1` inherits boundary density from the run, so the comparison stays fair at every
compression ratio and survives the model drifting toward or away from `N`.

**Stream.** `mbdfa/src/build_corpus.py` already emits `*_state.u8` and its `ALLOWED` table
encodes exactly which bytes are legal starts per state; a lead byte is `state == 0`. No
change needed — the builder must simply run over the new packed corpus, per language, with
the state files shipped alongside the `.bin`.

**Over training.** Evaluated on a held-out slice at every logged checkpoint, as a curve.
Alignment is expected to *fall* as compression rises toward `N` (fewer, longer chunks), so a
single end-of-run number is confounded by whatever ratio the run settled at; the curve shows
both.

**Gap closed at gate time:** the metric's noise floor — the `delta_F1` spread between two
seeds of one config. That spread is the smallest reportable effect, and no kill threshold can
be pre-registered until it is measured, so the config ticket must state its criterion relative
to it rather than as an absolute number.
