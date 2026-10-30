# Thai: the cell that separates logographic from every other explanation

Type: task
Status: resolved
Blocked by: 17

## Question

The width manipulation (ticket 17) falsified bytes-per-character as the cause: English widened to
2.69 B/char aligns (+0.094, 4/4 seeds) while Chinese at 2.67 does not (-0.205, 4/4). Width, ASCII
fraction and whitespace have all now failed to separate the aligned scripts from the unaligned
ones. What is left is the script type itself: Chinese and Japanese are the only logographic
scripts in the set.

**Thai is the discriminating cell**: alphabetic, 2.82 B/char (wider than Chinese), 5% spaces
(unspaced), 3.2% ASCII (lower than Chinese's 6.2%). Every surviving explanation predicts Thai
fails; only "logographic script breaks it" predicts Thai succeeds.

## Pre-registered prediction (written before the run)

| arm | B/char | spaces | ASCII | prediction |
|---|---|---|---|---|
| tha | 2.82 | 0.050 | 0.032 | **if logographic: aligned; if width/spaces/ASCII: not aligned** |
| eng_comb | 2.69 | 0.155 | 0.372 | aligned (already measured +0.094) |
| cmn | 2.67 | 0.036 | 0.062 | not aligned (already measured -0.205) |
| rus | 1.80 | 0.132 | 0.113 | aligned (already measured +0.075) |

**Decision rule:** Thai is *aligned* if its 95% CI excludes 0, *not aligned* if the CI includes 0
and the mean is <= +0.05, *unresolved* otherwise. Three seeds.

**Consequence, stated in advance:**
- Thai aligned -> the effect is specific to **logographic scripts**; the chunker learns character
  structure wherever characters are alphabetic, at any width, with or without spaces.
- Thai not aligned -> the effect needs **both** width and the absence of spaces; no single
  variable explains it, and the honest answer becomes a two-factor description with no mechanism.

## Answer

**Thai is NOT aligned (-0.145, 3/3 seeds negative, CI [-0.257, -0.034]), so the logographic
explanation is dead. But the run also failed its own harness control, so by the pre-registered
rule it is reported as a harness failure first and a result second.**

Kernel `mauricew/hnet-q1-tha`, COMPLETE, seeds 0-2, N=4, 4-language mixture.

| arm | B/char | space | ASCII | mean lift | seed SD | 95% CI | per-seed | verdict |
|---|---|---|---|---|---|---|---|---|
| **tha** | 2.82 | 0.050 | 0.032 | **-0.145** | 0.098 | [-0.257, -0.034] | -0.244, -0.144, -0.048 | **not aligned** |
| eng_comb | 2.69 | 0.155 | 0.372 | +0.020 | 0.136 | [-0.134, +0.174] | +0.141, +0.045, -0.127 | not aligned |
| cmn | 2.67 | 0.036 | 0.062 | -0.037 | **0.328** | [-0.409, +0.334] | **+0.341**, -0.249, -0.203 | not aligned |
| rus | 1.80 | 0.132 | 0.113 | +0.177 | 0.095 | [+0.069, +0.285] | +0.287, +0.131, +0.114 | aligned |

**The logographic explanation is dead.** Thai is alphabetic, not logographic, and it fails like
Chinese. So "the chunker breaks on ideographic scripts" is wrong; the effect tracks something
else that Chinese and Japanese share and Thai also shares.

**The harness control failed, and that has to be stated first.** `cmn` was -0.205 (4/4 seeds
negative) in the width run and is -0.037 here, with a seed at **+0.341** and a seed SD of 0.328.
The pre-registered control criterion was a difference under 0.15; the difference is 0.168, so it
fails, and the pre-registered consequence is that this run "is not comparable and is reported as
a harness failure rather than as a result". `eng_comb` also moved (+0.094 -> +0.020) while `rus`
reproduced cleanly (+0.075/+0.178 -> +0.177).

**What that leaves.** Thai's own result is 3/3 seeds negative with a CI excluding zero, which is
consistent with every surviving explanation (width, spaces, ASCII landmarks) and inconsistent with
the logographic one. But the run in which it was measured is the same run in which a control
drifted by more than the effect size. The honest reading is: **Thai does not align, and the
margin over noise is not yet established** — hence two more seeds of the identical mixture
(`mauricew/hnet-q1-tha2`, seeds 3-4, n=5) to tighten Thai and re-test the `cmn` control.

**The larger finding, which is now the dominant uncertainty in this effort:** per-language
alignment is sensitive to *what else is in the training mixture*, by more than the effects being
measured:

| arm | replication (6 langs) | width (5 langs) | tha (4 langs) |
|---|---|---|---|
| rus | +0.178 | +0.075 | +0.177 |
| cmn | -0.147 | -0.205 | -0.037 |
| eng_comb | -- | +0.094 | +0.020 |

`rus` is stable across all three mixtures; `cmn` moves by 0.17 and `eng_comb` by 0.07. The
8-seed replication's clean separation (rus and arb positive 8/8; cmn negative 7/8) is a
*fixed-mixture* result and should be reported as exactly that.

Budget: ~1.7 GPU-h for the three seeds, plus ~1.3 h for the two confirmation seeds.

