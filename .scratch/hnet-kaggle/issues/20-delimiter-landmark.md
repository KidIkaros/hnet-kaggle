# The delimiter-landmark test: adding spaces to Chinese

Type: task
Status: open
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

<!-- filled on resolution -->
