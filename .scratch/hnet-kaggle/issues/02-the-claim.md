# The claim and its falsifier

Type: grilling
Status: resolved
Blocked by: 11

## Question

What is the exact sentence this effort defends, what observation kills it, and what do we
report if the chunker never learns?

The user's instruction was "whatever is needed to make one defensible claim — you pick the
smallest honest scope", so this ticket decides rather than asks. It must settle four things:

1. **The claim, in one sentence**, in a form a table can support or refute.
2. **The measurement** that decides it, and its noise floor (seed spread must be known
   before the number means anything).
3. **The falsifier**: the concrete result that means "no", stated before running.
4. **The fallback claim** if the chunker does not engage — because "the chunker never
   learned" is a result about our scale, not about H-Nets, and the two must not be
   reported as the same thing.

Candidate claim (to be sharpened or replaced here):

> A byte-level H-Net's learned boundaries align with UTF-8 character structure, and the
> strength of that alignment across scripts tracks how badly a fixed BPE tokenizer fits
> each script.

Why this one: it is measurable at 8M params, it needs no 100B-byte crossover, it uses the
`mbdfa` DFA state stream we already have, and nobody has published a per-script alignment
statistic. It is also the mechanism the 2026 post claims qualitatively but never quantifies.

Related: this ticket must decide whether the **NLL-vs-entropy boundary rule** comparison is
inside the claim or left in the fog — the destination says *one* claim, and that is a
different one.

## Answer

**The claim:**

> A byte-level H-Net's learned chunk boundaries align with UTF-8 character starts far above a
> count-matched chance level, and that alignment is **encoding-invariant**: the same text
> re-encoded as UTF-16LE yields boundaries at the same characters, not at the same bytes.

**The measurement.** `delta_F1` per script, as defined in *Alignment metric and DFA stream*:
F1 on lead-byte positions minus the F1 of a count-matched random boundary set on the same
text. Under UTF-16LE the target is even-offset code-unit starts. Invariance is the
boundary-agreement rate between encodings at equal compression (that ticket, amendment A3).
All of it is computed from one run's checkpoints; nothing extra is trained.

**The falsifier, in two stages** — the first kills the run, the second kills the claim:

1. **Mechanism failed.** Realised compression never leaves 0.4-0.5 (target for N=6 is 0.167)
   and no ratio-loss hockey stick appears by 30% of steps. Report *"the mechanism did not
   engage"* and do not interpret alignment at all.
2. **Claim false.** The mechanism engages but `delta_F1` is ≤ the seed spread on every
   script, **or** cross-encoding boundary agreement is at chance. Either alone refutes its
   half: the first says boundaries are not character-aligned, the second says what they track
   is an encoding property rather than a character property.

**If the chunker never learns**, report it as a result about *our regime* — ~8M params, ~50 MB
per language, 1e16-class compute — not as a refutation of H-Net. The paper's own curves have
H-Net behind BPE below ~10B training bytes, so failure here is the expected outcome and is
worth publishing as such. The wording must keep them apart: "did not engage at 8M params on
50 MB/language" ≠ "H-Net does not work".

**Why this claim and not the compute multiplier.** The multiplier needs a crossover our budget
cannot reach (100B bytes, 1-stage). This needs only that the chunker learn *something*, and it
is measured proportionally: `delta_F1` is a difference at matched boundary density, not an
absolute score. It is also the paper's own thesis — meaning rather than bytes — turned into
something that can fail, which the qualitative chunk visualisations cannot do.

**Secondary, carried not claimed:** bits-per-character and bpb per script, plus BPB Gini
across scripts (from UBE), so the compression side stays visible. The NLL-vs-entropy
boundary-rule comparison remains a second claim and stays out of the falsifier.

**Novelty, stated conservatively.** UBE (NeurIPS 2026) changes the byte representation for BPE
and cites H-Net but never tests a chunker; FLEXITOKENS (2026) learns boundaries on ByT5 for
token classification; H-Net++ measures morphological alignment for Persian. None measures
alignment across scripts or across encodings on a dynamic chunker. That is the contribution,
and it is a mechanism contribution — make no compute-efficiency claim.
