# Run Q1 and produce the health curve

Type: task
Status: resolved
Blocked by: 05, 06, 07

## Question

Execute the Q1 run on a T4 and produce the artifacts the verdict depends on. Nothing to
decide here — the decisions were made upstream, and this ticket must not re-litigate them
mid-run.

Produce:

1. **The health curve**, logged from step 0: ratio loss, realised compression ratio (not the
   target), mean boundary probability, and the "fake vs real" boundary counts if the
   kth-value threshold is engaged.
2. **Boundary alignment over training**, per script, using the metric fixed in *Alignment
   metric and DFA stream* — a curve, not a single endpoint.
3. **Visualisations**: 20 boundary-marked examples per language, bytes annotated with the
   UTF-8 state so multi-byte characters are visible.
4. **The N-sweep** if the config ticket chose one: compression ratio and alignment per N.
5. Run logs and the checkpoint, as kernel output.

Operational rules: explicit `max_steps` **and** a wall-clock guard; `kernels logs -f` to
watch; remember that pushing over a running kernel does not cancel it.

Resolve with: the numbers, the kernel ref, and an explicit statement of whether the chunker
engaged.

## Answer

**Seed 0 ran the full 5000 steps on a T4 and produced the whole artifact set. The result is
mixed: alignment clears the bar on `rus` (+0.269) and `arb` (+0.382) and fails on `jpn`
(-0.234) and `cmn` (-0.162). Seed 1 is running the identical config.**

Kernel `mauricew/hnet-q1-s0`, COMPLETE, `stop=max_steps`, 5000 steps, 0.84 M params, N=6,
~17 min. Seed 1: `mauricew/hnet-q1-s1`. Kill the earlier combined attempt at step 75 after
measuring 3.25 s/step (see the amendment on *Experiment config*); the retry was the same config
with one seed.

**Align lift (`P(lead | boundary) - P(lead)`), held-out val, by eval step**

| step | eng | deu | rus | cmn | jpn | arb |
|---|---|---|---|---|---|---|
| 1000 | +0.003 | +0.004 | +0.318 | +0.227 | -0.157 | +0.382 |
| 2000 | +0.003 | +0.011 | +0.283 | +0.083 | -0.174 | +0.407 |
| 3000 | +0.003 | +0.012 | +0.274 | -0.066 | -0.191 | +0.390 |
| 4000 | -0.004 | +0.009 | +0.260 | -0.152 | -0.227 | +0.394 |
| 4750 | -0.003 | +0.009 | +0.268 | -0.162 | -0.231 | +0.381 |

Last 3 evals: rus **+0.269**, arb **+0.382**, cmn **-0.162**, jpn **-0.234**, eng -0.003,
deu +0.009. Realised bytes/chunk settled 4.4-5.3 against a target of 6.

**Against the pre-registered criteria from *Gates on Kaggle***

| criterion | verdict |
|---|---|
| bounded in `[0.5N, 2N]` = `[3, 12]` | **pass** — 4.4-5.3 throughout |
| alive: `>1.5x` step-0 (3.09) | **pass** — 4.5 at the end |
| alive: monotone 500 steps | **marginal** — 250: 3.52 -> 1000: 4.62 -> 2500: 4.29 -> 4.53; it rises then wobbles back, so not monotone |
| `>+0.15` on rus/jpn/arb, all three | **fail** — 2 of 3 |
| `align_on_lead > 0.85` on rus/jpn/arb | **fail** — reached on rus/arb, not on jpn |
| eng read by bpb only | honoured |

Not engaged on the letter of the criterion. It is also not a flat null: two of three scripts are
strongly and stably positive.

**The pattern, and the hypothesis it generates.** Positive alignment appears exactly where the
script is 2 bytes per character *and* whitespace-delimited (Cyrillic, Arabic); it is absent or
inverted on the 3-byte CJK scripts and sharpest on `cmn`. The model settles at 4.4-5.3 bytes per
chunk. That is comfortably above one character for 2-byte scripts, where a chunk boundary inside
a character is a minority event, but for `cmn` a chunk of 4.5-5.3 bytes is ~1.5 characters, so
roughly half of all boundaries must land on a continuation byte *by arithmetic alone* once chunks
drift off the 3-byte period. `cmn` decays monotonically (0.227 -> -0.162) rather than sitting
negative, which is what "starts near character level, then drifts to a sub-character period"
looks like.

**Read, therefore, as: the chunker learns character-scale structure on simple prepatterned
scripts and loses the phase on CJK once the ratio term pulls chunk length to a non-multiple of
the character width.** A candidate multiplicand — `target_ratio` and character width — is Q1-adjacent
but was not swept, so this is a hypothesis, not a finding.

**Artifacts** (kernel output): `health.json` equivalent at `seed-0.json` (21 log rows x 19 evals),
`examples-s0.json` (20 boundary- and DFA-state-annotated samples per language, 120 total),
`ckpt-s0.pt`, `q1.json`, run log with the guard unused.

**Operational notes.** 3.25 s/step at the start of the killed attempt versus ~0.2 s/step once
warm; the 8 h guard was never approached and 5000 steps took ~17 min. `cmn` bytes/chunk is the
noisiest column because it has the largest fraction of continuation-byte positions.

