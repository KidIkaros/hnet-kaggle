# Scale ladder: are characters simply out of reach at 0.84 M parameters?

Type: task
Status: resolved
Blocked by: 19, 21

## Question

Every result in this study comes from a 0.84 M-parameter model. The invariance failure is
consistent with two very different stories: (i) the chunker does not learn characters at all, or
(ii) 0.84 M parameters and 5000 steps are not enough to learn characters, and the UTF-16 failure
is a capacity artifact rather than a property of the architecture.

Story (ii) is testable and would be a *positive* result: characters are learnable, and here is
what it costs. The first rung: the same 6-language config at 2.6 M parameters (d_model 256,
3 encoder / 4 main / 3 decoder, d_ff 1024) with the same 5000 steps, plus a UTF-16 twin, and
the same character-alignment and invariance measurement.

## Design

- Arms: UTF-8 and UTF-16, at 2.6 M parameters, N=4, 3 seeds each, fixed 6-language mixture.
- Metrics: the same alignment lift and the same character-space boundary agreement.
- Comparison: the 0.84 M numbers (UTF-8 rus +0.179 / arb +0.250, 0/10 seeds negative; UTF-16 rus
  -0.228 / arb -0.326) are the baseline.

**Pre-registered reading.** If the 2.6 M UTF-8 model aligns *more strongly* in UTF-8 and the
UTF-16 twin aligns *positively* too, then characters are within reach and the earlier failure was
capacity. If the UTF-8 alignment strengthens while the UTF-16 twin stays negative, then capacity
buys better UTF-8 structure but not characters, and the framing "the chunker keys on the
encoding" stands at both scales. If neither changes, 0.84 M was not the limitation.

**Cost.** ~4.5 h GPU per arm per seed at 2.6 M (measured: 0.84 M runs at 0.2 s/step on a T4;
2.6 M scales roughly with parameters). Two arms x 3 seeds is beyond one quota week, so the
first pass is **one seed per arm** (~9 h) and only proceeds to more seeds if the first is
promising.

**Also worth folding in if quota allows.** The ratio control never locked (realised bytes/chunk
2.4-7.1 against a target of 6) and the practitioner guidance is that ratio dynamics need >5000
steps. A 10,000-step rung at 2.6 M is the honest test of "characters need more steps" as against
"characters need more parameters".
