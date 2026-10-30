# The delimiter-landmark test: adding spaces to Chinese

Type: task
Status: resolved
Blocked by: 17, 18

## Question

Five candidate causes have been refuted. What is left is that Chinese, Japanese and Thai share
something the aligned scripts do not. The candidate still standing is **byte-phase
recoverability**: a 1-byte ASCII landmark recurring in the stream lets a model find character
starts without tracking UTF-8 phase, and the aligned scripts (Russian, Arabic, Devanagari) all
have spaces -- while Chinese, Japanese and Thai barely do.

The removal test (ticket 14) showed deleting spaces from Russian and Hindi does *not* destroy
alignment, so spaces are not *necessary*. The untested direction is whether they are
*sufficient*: add a space every K characters to Chinese and see whether alignment appears.

`kernel/spaced.py` builds the variant: a space inserted after every 3 (and every 6) characters.
Chinese goes from a 0.036 space rate to 0.268 at K=3 -- above Russian's 0.132 -- with the byte
width essentially unchanged (2.67 -> 3.10 B/char, the space being 1 byte).

**Prediction.** If the missing landmark is what breaks Chinese, `cmn_sp3` aligns. If it does not,
the failure is specific to those scripts and no delimiter story survives.

Controls in the same run: `rus` and `arb` (aligned) and `cmn` (not aligned), so a null result is
readable as a harness result rather than a broken run.

## Answer

**Spaces are not sufficient. Adding them to Chinese does not produce alignment, so the delimiter
hypothesis is dead in both directions.**

Kernel `mauricew/hnet-q1-spaced-cpu`, 2 seeds x 5000 steps, 5-language mixture.

| arm | space rate | mean lift | per-seed |
|---|---|---|---|
| cmn | 0.036 | -0.244 | -0.278, -0.210 |
| **cmn_sp3** | **0.268** | **-0.273** | -0.322, -0.224 |
| cmn_sp6 | 0.168 | -0.225 | -0.247, -0.202 |
| rus (control) | 0.132 | +0.124 | +0.174, +0.075 |
| arb (control) | 0.162 | +0.315 | +0.306, +0.323 |

Chinese goes from a 0.036 space rate to 0.268 -- well above Russian's 0.132 -- and its alignment
does not move (-0.244 -> -0.273, a change of -0.029, inside seed noise). Both controls reproduce,
so the run is a harness result and not a broken one.

**Together with the removal test (ticket 14), whitespace is now excluded in both directions:**
deleting every space from Russian and Hindi left alignment intact, and adding spaces to Chinese
leaves it unchanged. Whitespace is neither necessary nor sufficient.

**And this is what the byte-phase account predicted.** Adding spaces does not change where byte
values change *inside* a Chinese character, so the off1 phase (boundaries one byte into the
character, 2.35x enriched) should persist -- and it does. The delimiter test was designed before
the byte-phase analysis and now reads as a confirmation of it.

**Operational note.** The run crashed in `examples()` both times: the spaced variants had no
UTF-8 DFA state stream, then the kernel mounted a dataset version that predated the fix. The
training results were written before the crash and were recovered from the failed kernel's
output, so no re-run was needed -- but the lesson is that `examples()` sits downstream of the
result and should not be able to take the run down with it.

