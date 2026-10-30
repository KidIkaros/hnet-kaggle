# Thai: the cell that separates logographic from every other explanation

Type: task
Status: open
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

<!-- filled on resolution -->
