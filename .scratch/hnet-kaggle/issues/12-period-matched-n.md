# Period-matched N sweep

Type: task
Status: resolved
Blocked by: 08, 09

## Question

Is the `cmn`/`jpn` alignment failure dimensional — a chunk length that is not a multiple of
the script's character width — or is it a real absence of character structure?

The verdict left this as the one thing standing between a hypothesis and a claim. Both seeds
settled at 4.4-5.3 bytes/chunk against N=6. For 2-byte scripts that is comfortably above one
character, and alignment was positive on both seeds. For `cmn` (3 bytes/char) 4.7 bytes is ~1.5
characters, so roughly half of all boundaries must land mid-character by arithmetic alone once
chunks drift off the 3-byte period — and `cmn` decays monotonically as that happens (0.227 ->
-0.162). `jpn` moved the other way between seeds as its chunks shortened (3.91 vs 4.58), which
cuts against a pure width explanation.

Run the same frozen config with N set so that a chunk is ~2-3 characters, per script:

- `cmn` (3 B/char): N = 6 is 2 chars — already right, so also try N = 9 (3 chars) and N = 3.
- `rus`/`arb` (2 B/char): N = 4-6.
- `jpn`: mixed kana/kanji, so sweep N = 6, 9.
- English/German stay at N = 6 as the within-run control.

Resolve with: per-script alignment lift at period-matched N, and a statement of whether the
`cmn` phase loss disappears when N is a character multiple. If it does not disappear, the
verdict's hypothesis is dead and the honest claim becomes "character structure is found only
where the script is whitespace-delimited", which is a narrower and more interesting result.

Budget: 6 configs x 1 seed x ~20 min = ~2 GPU-h of the 8.92 left this week.
