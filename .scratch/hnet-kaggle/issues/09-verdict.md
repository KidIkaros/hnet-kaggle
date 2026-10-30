# Verdict: what the result means, and whether Q2 follows

Type: grilling
Status: resolved
Blocked by: 08

## Question

Given the Q1 numbers, what do we actually claim — and does the matched-loss transfer
experiment (Q2) earn its GPU-hours?

This is where the three outcomes that look identical in a table get separated, before
anything is written up:

- **the claim is supported** — alignment is above chance and tracks tokenizer fit;
- **the claim is false** — alignment is flat across scripts, or at chance;
- **the claim is untested** — the chunker never engaged, so the measurement describes a
  half-learned router and says nothing about H-Nets.

Decide:

1. Which of the three we are in, using the pre-registered falsifier from *The claim and its
   falsifier* — not a new one invented after seeing the numbers.
2. Whether Q2 is justified. It is a separate claim (loss-to-loss transfer at matched
   validation bpb) and needs its own pre-registered falsifier before it runs.
3. What the honest one-paragraph summary is, in plain language, including what the result
   does *not* show (it will not be evidence about 1.28x, or about 100B-byte regimes).
4. Whether anything here changes the corpus or config decisions enough to warrant a re-run.

## Answer

**Outcome 3, "the claim is untested", with a partial mechanism inside it — and a real,
two-seed sign split that is not noise.**

**Both seeds, last 3 evals averaged**

| script | seed 0 | seed 1 | sign agrees |
|---|---|---|---|
| arb | +0.382 | +0.312 | yes, both well over the +0.15 bar |
| rus | +0.269 | +0.233 | yes, both over the bar |
| jpn | **-0.234** | **-0.003** | yes in sign, but one is 78x the other |
| cmn | **-0.162** | **-0.127** | yes, both negative |
| deu | +0.009 | +0.011 | yes, both ~0 |
| eng | -0.003 | +0.003 | yes, both ~0 |

Both seeds ran the identical frozen config to 5000 steps; neither hit its guard.

**1. Which of the three.** Not "supported": the pre-registered criterion was `>+0.15` on
rus/jpn/arb on **all three** and that fails on `jpn` in both seeds. Not "false" either: rus and
arb clear it twice each, and a flat null would not show a stable two-seed split between
whitespace-delimited 2-byte scripts and CJK. So the honest classification is **untested at the
level the claim needs, with a mechanism visible at the level below it**.

Why "untested" is the right word rather than a convenience: everything about this run is sized
under the regime where the paper's dynamics are supposed to settle. 0.84 M params against the
paper's 760 M, ~7% of one epoch, N fixed at 6 with no character-width control, and a router that
is still moving at step 5000. The measured quantity — where boundaries fall relative to character
starts — is downstream of a compression target the model has not finished negotiating. We are
measuring a half-trained router, exactly the third outcome this ticket names.

**2. Q2 does not follow yet, but only barely.** Two of three non-Latin scripts show the
character-scale structure the claim is about, on both seeds, so the mechanism is real enough to
be worth a second experiment; but `jpn` is at chance and `cmn` is *worse* than chance on both
seeds, so the routing is not converging on CJK. Q2 at matched bpb would compare a model whose
chunker is phase-locked on 2-byte scripts to a BPE baseline, and any transfer advantage or
absence would be confounded by whether the comparison happened to weigh CJK heavily.

**3. The one-paragraph summary, as it should be written.**

> A small (0.84 M parameter) byte-level H-Net trained for 5000 steps on six languages learns to
> put chunk boundaries on character starts in Arabic and Russian — about 31 points and 25 points
> above chance on held-out text, on both seeds — and does not on Chinese and Japanese, where
> boundaries are at or below chance. English and German sit near zero for structural reasons
> (nearly every ASCII byte starts a character; German's multi-byte characters are 2% of its
> bytes). So the architecture does discover character structure, but only on scripts where a
> chunk of the model's preferred 4-5 byte length almost always spans whole characters; when the
> ratio target forces chunk lengths to about 1.5 characters, the boundaries lose the phase. This
> is a statement about a 0.84 M-parameter model on ~20 MB of text, and it is *not* evidence about
> the paper's 1.28x/2.06x compute multipliers, about the 100B-byte crossover, about stage-2
> H-Nets, or about downstream quality.

**4. Does anything change the corpus or config enough to re-run?**

- **Config yes, corpus no.** The corpus is behaving: language purity is high, offsets validate,
  no language is starved. The one real corpus problem (`eng` boilerplate) does not affect an
  alignment claim and only matters for bpb, which Q1 does not use.
- The concrete config gap the data points at is that **`(N, character width)` was never
  controlled**. `cmn`'s decay tracks bytes/chunk falling onto a non-multiple of 3, and `jpn`
  improved (seed 1, 3.91 bytes/chunk) as its chunks got shorter than seed 0's — the opposite of
  `cmn`. The obvious next experiment is a period-matched N: set N to ~2-3 characters per chunk
  (6 for CJK, 4-5 for 2-byte scripts) and see whether the phase returns on `cmn`.
- A re-run also needs the monotonicity check the last run nearly met: bytes/chunk rose 2.06 ->
  4.62 by step 1000 then drifted to 4.29 at 2500 and 4.53 at the end. The ratio never locked in,
  which is itself evidence the run is at the edge of the regime, not inside it.
- Not worth re-running for: size. Going to 8 M is 15.5 h/run and does not address the phase
  problem, which is dimensional, not capacity-bound.

**Still open before any write-up claims anything:** the period-matched N isn't swept, so the
`cmn`/`jpn` explanation remains a hypothesis that happens to fit. It should be labelled as such
wherever it appears.

## Amendment: eval robustness, measured after the fact

The Q1 runs scored alignment on 8 fixed windows x 256 B = 2 KB per language per eval, with no
uncertainty reported. `tests/eval_noise.py` recomputes every language's lift over **all 1953
non-overlapping val windows** (500 KB) and bootstraps the 8-window estimator to recover its
sampling error.

| script | seed 0 reported | seed 0 full-val | seed 0 95% CI | seed 1 reported | seed 1 full-val | seed 1 95% CI |
|---|---|---|---|---|---|---|
| arb | +0.382 | +0.390 | [+0.360, +0.412] | +0.312 | +0.313 | [+0.276, +0.345] |
| rus | +0.269 | +0.257 | [+0.213, +0.297] | +0.233 | +0.243 | [+0.199, +0.281] |
| cmn | -0.162 | -0.193 | [-0.251, -0.114] | -0.127 | -0.099 | [-0.151, -0.045] |
| jpn | -0.234 | -0.229 | [-0.268, -0.180] | -0.003 | -0.045 | **[-0.112, +0.031]** |
| deu | +0.009 | +0.006 | [-0.004, +0.014] | +0.011 | +0.008 | [-0.000, +0.017] |
| eng | -0.003 | +0.001 | [-0.004, +0.005] | +0.003 | +0.001 | [-0.003, +0.005] |

**The eval was underpowered in design but not misleading in outcome**: every reported number sits
inside its own CI, and the estimator's sd is 0.014-0.037 — far tighter than feared for 2 KB of
text, because the windows are long relative to the chunk statistics.

**One conclusion changes.** `jpn` on seed 1 is **not significantly negative** — its CI spans zero.
So `jpn` is not "below chance on both seeds": it is clearly negative on seed 0 (-0.229) and at
chance on seed 1 (-0.045). That is seed variance in a near-zero effect, not a stable sign, and the
verdict's table should be read that way. `cmn` remains negative on both seeds with CIs that
exclude zero, so the CJK claim survives; `rus`/`arb` remain solidly positive on both.

Two things to fix before any of this is published: report the CI next to every lift, and stop
reusing one fixed window set across evals (a fresh draw with more windows is free).

## Amendment 2: the `cmn`/`jpn` explanation in section 1 is refuted

The period-matched N sweep (ticket 12) tested the "chunk length vs character width" hypothesis that
this verdict offered as the reason CJK failed. It does not hold:

| N | rus | arb | cmn | jpn |
|---|---|---|---|---|
| 3 | +0.247 | +0.259 | -0.003 | -0.162 |
| 4 | +0.301 | +0.233 | **-0.206** | -0.280 |
| 6 | +0.317 | +0.405 | -0.109 | -0.230 |
| 9 | +0.304 | +0.383 | -0.111 | -0.197 |

At N=4 `cmn` sits at 2.97 bytes/chunk — **one character per chunk** — and scores its worst lift.
There is no N at which `cmn` or `jpn` is positive, and rus/arb are positive at all four. Matching
the character period does not restore the phase.

**What the verdict should say instead.** Not "the chunker loses the phase when chunks are a
non-multiple of the character width", but: **the chunker aligns to character starts only on
2-byte whitespace-delimited scripts, at every compression target tested, and never on CJK.** The
width explanation and the word-delimitation explanation are confounded in our six languages, so
the mechanism is still unresolved — it needs a 3-byte whitespace-delimited script (Devanagari) to
separate them.

The classification stays **"untested, with a mechanism inside it"**; what changes is which
mechanism is on the table, and that the CJK half is now robust across two seeds and four
compression targets rather than one.

## Amendment 3: the mechanism is now identified — whitespace, not width

Ticket 13 tested the confound left open by amendment 2 by adding Hindi (2.53 bytes/char,
CJK-like width, 0.192 space rate — the highest measured):

| lang | bytes/char | space rate | N=4 | N=6 |
|---|---|---|---|---|
| rus | 1.80 | 0.132 | +0.270 | +0.358 |
| arb | 1.78 | 0.162 | +0.282 | +0.063 |
| hin | 2.53 | 0.192 | +0.229 | +0.325 |
| cmn | 2.67 | 0.036 | -0.022 | -0.209 |
| jpn | 2.68 | 0.020 | -0.023 | -0.264 |

**Width is ruled out; whitespace is the discriminator.** Hindi is 0.14 bytes/char away from
Chinese and Japanese, and positive at both ratios while they are negative.

**The claim, now in its final form:**

> At 0.84 M parameters and 5000 steps, a byte-level 1-stage H-Net places chunk boundaries on
> character starts above chance only in scripts that separate words with spaces — Russian, Arabic
> and Hindi, at both compression targets tested — and not in the two scripts that do not, Chinese
> and Japanese. Matching the chunk length to the character period does not change this. English and
> German are unmeasurable on the metric because nearly every byte is already a lead byte.
>
> The mechanism is most likely that a word boundary is trivially a character start, so a router
> anchored on whitespace reads as character-aligned on any multi-byte script; with no whitespace
> there is no anchor and the router drifts.

**What this means for the "untested" classification.** It stays, but it is now a narrower gap: the
question is no longer "does the chunker learn characters" but "does the router depend on the
whitespace prior". That is directly relevant to Cartesia's own framing in Post II — they argue
pretokenization's role is to align tokens with whitespace-separated words and that "learning basic
concepts such as words from bytes is a nontrivial problem". Our result is evidence for exactly
that at small scale: strip the whitespace prior and the learned chunker does not recover
character structure, let alone words.

**New caveat, and it is a real one.** Adding `hin` to the mix moved `cmn` at N=4 from -0.206 to
-0.022. Per-language alignment is sensitive to the composition of the training mixture, so a
single-language number is not a stable property of that language. Within-run contrasts at equal N
are trustworthy; cross-run comparisons are not. Any write-up must report per-run composition.

## Amendment 4: amendments 2 and 3 are withdrawn; the claim goes back to "no mechanism"

The space ablation (ticket 14) tested amendment 3's mechanism by deleting every `0x20` from Hindi
and Russian and retraining at the same N in one run. Alignment did not collapse:

| language | lift | | language | lift |
|---|---|---|---|---|
| hin | -0.037 | | hin_nospace | **+0.081** |
| rus | +0.162 | | rus_nospace | **+0.211** |
| cmn | -0.225 | | jpn | -0.096 |

Removing the whitespace prior does not remove alignment. **The whitespace explanation is dead**,
as the width explanation was before it.

**Worse for this effort, and more important than the mechanism:** `hin` scored +0.229 at N=4 in
ticket 13's run and -0.037 here; `cmn` moved from -0.022 to -0.225 across the same two runs. The
only differences were the training mixture and a near-duplicate language in the batch. So
single-seed per-language alignment is **not reproducible across mixtures**, and that instability
(0.27 on hin) is larger than any effect claimed anywhere in this ticket.

**What the verdict now says, in full:**

> A 0.84 M-parameter byte-level 1-stage H-Net, trained 5000 steps on six languages, places chunk
> boundaries on character starts above chance in Russian in all three runs that measured it, and at
> or below chance in Chinese in all three. Japanese is at or below chance in all three but with
> wide spread. Hindi and Arabic are positive in some runs and at chance in others and are **not
> established**. No mechanism is established: character width and whitespace delimitation have both
> been tested and both fail.
>
> The honest summary of Q1 is therefore that alignment differs by script, that Russian is the one
> language where the effect has reproduced, and that the per-language measurement is too unstable
> at one seed per config to support a script-level claim about anything else.

**This is the third interpretation in this ticket that a better experiment overturned** — first the
off-by-one probe (a bug, not a finding), then the dimensional hypothesis (refuted by the N sweep),
then the whitespace mechanism (refuted by the ablation). The pattern is that at 0.84 M params and
one seed per config, effects are being read out of variance. The next experiment must be a
variance estimate, not another hypothesis.

**Classification.** Still outcome 3, but for a sharper reason than "the chunker is half-trained":
the *measurement* is not yet capable of supporting the claim. That is a fixable, well-understood
problem (seeds, CIs, fixed mixture) and it is the gate before any write-up.

## Amendment 5 (final): the claim, with replication

The seed replication (ticket 15, 8 seeds, fixed mixture, N=4) settles what the earlier
single-seed runs could not. Every pre-registered check passed, and the explicit kill criterion
(rus and cmn overlapping) was not met.

| script | bytes/char | mean lift | 95% CI | sign test | verdict |
|---|---|---|---|---|---|
| eng | 1.00 | -0.000 | [-0.002, +0.001] | 4/8 | not aligned (structural) |
| deu | 1.02 | +0.002 | [-0.008, +0.011] | 6/8 | not aligned (structural) |
| **rus** | 1.80 | **+0.178** | **[+0.119, +0.236]** | **8/8, p=0.004** | **aligned** |
| cmn | 2.67 | **-0.147** | [-0.225, -0.068] | 7/8 neg, p=0.035 | **not aligned** |
| jpn | 2.68 | -0.123 | [-0.240, -0.006] | 6/8 neg, p=0.14 | **unresolved** |
| **arb** | 1.78 | **+0.266** | **[+0.203, +0.330]** | **8/8, p=0.004** | **aligned** |

**The final claim:**

> At 0.84 M parameters, trained 5000 steps on bytes with a 1-stage H-Net, the chunker places
> boundaries on character starts above chance in Arabic (+0.266) and Russian (+0.178) — both 8/8
> seeds positive, p = 0.004 — and below chance in Chinese (-0.147, 7/8 seeds negative). Japanese
> is unresolved (6/8 negative, p = 0.14). English and German are structurally unmeasurable on this
> metric: 99.6% and 98.5% of their bytes are already lead bytes.
>
> The split tracks **bytes per character** — aligned at 1.78-1.80, not aligned at 2.67 — and not
> whitespace delimitation: deleting every space from Hindi and Russian left their alignment
> intact. It does not track the chunk-length/character-width period either: making chunks exactly
> one character long made Chinese *worse*, not better.
>
> This is a statement about 0.84 M parameters and ~7% of one epoch. It is not evidence about the
> 1.28x/2.06x compute multipliers, the 100B-byte crossover, 2-stage H-Nets, or downstream quality.

**Why the mechanism is still open.** Width is the best predictor, but the evidence is
circumstantial: seven languages, one of which (Hindi) is unreplicated, and no experiment that
manipulates width directly. The one causal test we ran — removing whitespace — came out null.
The next causal test would be to hold the script fixed and change the byte width, e.g. by
re-encoding a single-byte-width script into UTF-16 (2 bytes/char) or padding characters, and
watching alignment move. That is cheap on CPU and would settle the mechanism; it is not needed for
the claim above, which is a claim about what the model does, not about why.

**History of this verdict, stated plainly:** four earlier readings of this data were wrong — an
off-by-one in the alignment probe (a bug), a chunk-length/character-width hypothesis (refuted),
a whitespace hypothesis (refuted), and a single-seed reading of Hindi that did not survive
replication. The claim above is the fifth reading and the only one that has survived replication
with a pre-registered kill criterion that was not met.

## Amendment 6: the width explanation is withdrawn; the cause is unidentified

The width manipulation (ticket 17) falsified bytes-per-character as the cause. English widened to
2.69 B/char — Chinese's exact width, same script, same words, same spaces, same byte count —
aligns (+0.094, every seed); Chinese at 2.67 does not (-0.205, every seed). Same width, opposite
signs. ASCII-landmark density does not separate them either: `rus_comb` (0.058) and `cmn` (0.062)
have nearly identical ASCII shares and opposite signs.

**The claim's mechanism sentence is therefore withdrawn.** What remains is the effect — Russian
and Arabic align, 8/8 seeds, p = 0.004; Chinese does not — with no identified cause. Four
candidate mechanisms have now been tested and refuted: chunk-length period (ticket 12),
whitespace delimitation (ticket 14), bytes per character (ticket 17), ASCII-landmark density
(ticket 17).

**And the pattern across all five amendments is itself the finding.** Every reading of this data
that was proposed as a cause has been overturned by the next experiment, while the effect itself
has replicated every time it was measured with seeds. At 0.84 M parameters, cross-sectional
comparisons between languages keep producing a story that a within-language manipulation then
destroys. The honest summary is: *this model aligns chunk boundaries to character starts in some
scripts and not others, and we have not identified why.*

**Next:** Thai (ticket 18) — alphabetic, wider than Chinese, unspaced, lower ASCII share than
Chinese. Every surviving explanation predicts Thai fails; only "logographic script breaks it"
predicts Thai succeeds. If Thai aligns, the cause is the script type. If it does not, the honest
answer is a two-factor description with no mechanism.

## Amendment 7: the cause is unidentified, and mixture sensitivity is the dominant uncertainty

Thai (ticket 18) is **not** aligned: -0.145, 3/3 seeds negative, CI [-0.257, -0.034]. The
logographic explanation is therefore dead — an alphabetic script fails exactly like Chinese. Five
candidate causes have now been tested and refuted: chunk-length period, whitespace delimitation,
bytes per character, ASCII-landmark density, and script type.

**And the run failed its own harness control.** `cmn` reads -0.205 in the width mixture and
-0.037 in the Thai mixture, with one seed at +0.341 and a seed SD of 0.328 — outside the
pre-registered 0.15 tolerance, so by the rule written down in advance the Thai run is a harness
failure first and a result second. Two more seeds of the identical mixture are running to settle
it.

**The finding that now dominates:** per-language alignment moves with *what else is in the
training mixture* by more than the effects being measured (`cmn`: -0.147, -0.205, -0.037 across
three mixtures; `eng_comb`: +0.094, +0.020 across two), while `rus` is stable (+0.178, +0.075,
+0.177). This is consistent with the Hindi instability found earlier and it upgrades that from an
anecdote to a first-class limitation.

**Final position of the claim.** The effect is real and replicates at fixed mixture — Russian and
Arabic align (8/8 seeds, p = 0.004), Chinese does not — but its cause is unidentified after five
refuted hypotheses, and the per-language numbers are mixture-dependent. The write-up now says
exactly that, and every figure is reported with its mixture.

**What this effort has actually produced:** a replicated measurement, a working Kaggle-only
harness, a published corpus, and an unusually honest record of five refuted explanations. The
causal question is open and is now well-posed: whatever separates Chinese, Japanese and Thai
from Latin, Cyrillic, Arabic and Devanagari is not width, whitespace, ASCII density, script type
or chunk period.
