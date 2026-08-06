---
name: release-parity-refactor
description: This skill should be used when refactoring, rewriting, or reorganizing code that already produced a published artifact — a released dataset, a shipped model, a public export. Use it when the user says "clean up this pipeline", "extract these scripts into a package", "modernize the build", or whenever the old code's output is already in someone else's hands.
---

# Release-Parity Refactor

Refactoring code whose output has already shipped is not ordinary refactoring.
You have a non-negotiable oracle — the published artifact — and both the code
and the documentation may be wrong about how it was made. The job is to
reproduce decisions, not to improve them.

## Core rules

1. **The published artifact is the spec.** Not the code, not the docs, not the
   registry. In one session the pipeline doc disagreed with the code in 12
   places, and a table the agent itself had just written as "authoritative"
   still had a wrong row. Every parameter you adopt must be traceable to
   something that demonstrably produced the release.

2. **Find the real producer from output fingerprints, not from registries.**
   A `SOURCES = {...}` dict mapping name → function is a claim, not evidence.
   Four superseded producers were found this way: a registry entry specified
   `I_MIN=15` with per-frame random sampling, but the release had 30,956 rows
   across 30,956 distinct groups — exactly one row per group. That histogram
   is the fingerprint of a different script (`K_PER_TOUCH=1`, `I_MIN=12`), and
   it settled the question without trusting either script's docstring.

3. **Grade by whether a proof is possible, and enforce the grade in code.**
   Split sources into *provable* (the release can be joined back and compared)
   and *unprovable* (it cannot). Unprovable ones are **moved verbatim, not
   rewritten** — there is no way to show a rewrite preserved them. Make the
   distinction machine-checkable: a test asserting that unprovable modules do
   **not** export the verification entry point stops "merely moved" from ever
   being read as "proven".

4. **Verify the comparison key discriminates before trusting it.** Checking
   that a join column is non-null is not checking that it identifies a row.
   One source had 166,104 released rows collapsing to 31,096 distinct keys;
   matching that key set is equally consistent with emitting 31,096 rows or
   166,104. Confirm uniqueness, or compare multisets and row counts too — and
   state which of the three you actually did.

5. **Exact is the bar on deterministic paths.** "Within tolerance" is not a
   pass when nothing in the path is random. Decide up front which sources are
   deterministic; for those demand set/count equality and investigate every
   difference. One source returned a comfortable "within budget" verdict that
   was really 1,287 wrong keys.

6. **Never tune a parameter to make the comparison pass.** Trying thresholds
   or seeds to see which fits converts verification into curve-fitting, and
   the resulting green check is worse than no check because it manufactures
   confidence. Write the prohibition into every delegated task, naming the
   specific temptation that source invites ("if it misses, do not try the
   other script's `I_MIN=10`"). When a comparison fails, the cause is in the
   iterator or in a pipeline stage — go find it.

7. **"Better" code can be less faithful.** Sorting a glob removes a
   filesystem-order dependency and is the obviously superior choice — and it
   broke parity, because a running quota over the whole source makes order
   load-bearing. The tell was a contrast inside the legacy code itself: a
   sibling iterator *did* sort, its port sorted too, and that one matched
   exactly. Preserve incidental-looking behaviour until a test proves it
   incidental, and comment why it stays.

## Diagnosing a near-miss

A count that is close but wrong usually means a **missing stage**, not a wrong
threshold — and the fastest way to tell them apart is to rebuild the suspect
stage exactly and re-measure.

One source emitted 97,529 rows against a published 48,197. Reconstructing the
legacy baseline verbatim gave a filter pass rate of 0.958 — nearly identical
to the port's 0.957. That ruled the filter *correct* and made the residual
factor `0.473 / 0.958 = 0.494` an obvious survival rate: a dedupe stage that
lived in the shared driver, not in the iterator, and had been dropped in the
move. Had the threshold been "tuned" to close the 2× gap instead, the result
would have been a passing test over a permanently wrong implementation.

Order of suspicion for a near-miss: a dropped pipeline stage, then unit or
grouping granularity (a per-group window reset is behaviour), then iteration
order, then the parameters — which should be last, and are usually innocent.

## Default workflow

1. **Import verbatim first.** Copy the old scripts into the new repo
   byte-for-byte as the initial commit (verify with `cmp`), so every later
   step is diffable against a known-working state.
2. **Inventory the oracle.** For each output record row counts, distinct join
   keys, group-size histograms, and column sets. These are what you fingerprint
   producers against, and they take minutes.
3. **Audit the join key** (rule 4) and assign grades (rule 3) *before* writing
   migration code. The grade table is itself a deliverable.
4. **Build the comparison harness before the first migration**, with a
   two-tier verdict — deterministic exact, stochastic bounded — and make it
   print which tier it applied.
5. **Migrate one source, prove it, then generalize.** The first proven source
   validates the abstraction; the second, added without touching the shared
   driver, proves it is actually general. Check that claim with `git diff` on
   the driver rather than asserting it.
6. **Record every parameter with `file:line` and the quoted line.** A bare
   number in a summary table is how the fifth conflicting value gets born.

## What a passing comparison means

State the strength, not just the verdict:

- **Row-level parity** — unique key, set equality. The strong claim.
- **Key-set + row-count parity** — non-unique key. Rules out gross error, but a
  compensating multiset difference would still pass. Say so.
- **Moved verbatim** — no proof attempted, by design. Never phrase this as
  "migrated" in a summary.

An unqualified "PASS" spanning a mixture of these three is the failure this
skill exists to prevent: the grading scheme was the deliverable's main honesty
claim, and one unlabelled tier quietly voids it.

Record findings and rejected reconstructions in the ledger (see
`debug-ledger`); when the artifact you are reproducing is an estimator's
output rather than a dataset, `gt-validation` covers the scoring side.
