# P8 — Differential test report (Rust acs2-core vs pyalcs)

Harness: `baseline/dump_differential.py` (instruments the unmodified pyalcs core
to emit `fixtures/differential_cases.json` + `fixtures/differential_episode.json`),
`acs2-core/tests/p8_differential.rs`, `acs2-envs/tests/episode_differential.rs`.

## Part A — deterministic learning step (the gate)

800 random `(population, p0, action, p1, time, reward)` inputs were generated.
For each, pyalcs computes the match set, action set, next-state match set, the RL
bootstrap (`get_maximum_fitness`), and the population after one learning step
(`apply_alp` + `apply_reinforcement_learning`). The Rust core is run on identical
inputs and compared field-by-field (population canonically sorted by `(C,A,E)`).

Each generated input falls in exactly one bucket:

| Bucket | Count | Compared exactly? |
|---|---|---|
| Deterministic, RNG-free | **761** | Yes — **0 divergences** |
| RNG-excluded (`expected_case` ≥2 mark candidates) | 24 | No — pick is random by design |
| pyalcs-bug-excluded (`apply_alp` mid-iteration skip) | 15 | No — pyalcs bug, see below |
| **Total generated** | 800 | |

**Result: zero divergence across all 761 deterministic cases**, validating
`form_match_set`, `form_action_set`, `get_maximum_fitness`, `apply_alp`
(expected/unexpected/covering/deletion), and `apply_reinforcement_learning`.

**Why 761 green is trustworthy (one-sided error).** Both filters can only
*over-exclude*, never produce a false pass. A wrongly-kept skip-tainted or RNG case
would diverge from Rust's seed-0 run and **fail** the test; a wrongly-excluded case
simply never reaches Rust. Kept cases are seed-invariant (12-seed check), so Rust's
single seed necessarily reproduces them. The detector's completeness therefore
bounds coverage, not validity — the gate cannot be passed by a filter hiding a
divergence.

Branch coverage among kept cases: non-empty action set 450, expected-case 292,
unexpected-case 158, covering 469, in-loop inadequate deletion 31. The
clean-deletion path (inadequate classifier deletes identically in both) is the 31
kept deletion cases; only deletions that trigger the pyalcs skip below are
excluded, so the deletion+remap logic is still differentially validated where
pyalcs is correct.

### Why the two excluded buckets are not gate failures

- **RNG (24).** `apply_alp`'s only randomness is `expected_case` →
  `PMark.get_differences`, which picks one position at random when ≥2 marked
  positions differ from `p0`. These are filtered analytically and confirmed by
  12-seed invariance. PROJECT_CONTEXT §6 permits divergence only where RNG enters;
  these are exactly those.

- **pyalcs `apply_alp` mid-iteration skip (15) — a newly discovered bug.**
  pyalcs iterates `for cl in action_set:` while calling `action_set.safe_remove(cl)`
  inside the loop when a classifier is inadequate
  (`ClassifiersList.py:145-161`). Deleting an element mid-iteration makes Python's
  list iterator **skip the next element**: that classifier is never processed (no
  experience bump, no marking, no anticipation handling). This is a fourth pyalcs
  hazard beyond the three in PROJECT_CONTEXT §4, and it lives in a **core learning
  primitive** (`apply_alp`), not the orchestration layer.

  Rust does not reproduce it: `apply_alp` iterates a cloned `original_action_set`
  and defers deletion to after the loop (the design ARCHITECTURE.md prescribed to
  avoid index aliasing), so every action-set member is processed. The underlying
  operations agree — `unexpected_case` in isolation marks the skipped classifier
  identically to Rust (verified) — so the Rust loop result is the correct
  composition of validated components; only pyalcs's buggy loop orchestration
  diverges. Per §4/§J the project re-derives the loop and does not transcribe
  pyalcs orchestration bugs, so Rust is the correct side.

  `acs2-core/tests/p8_differential.rs::rust_processes_whole_action_set_where_pyalcs_skips_after_deletion`
  pins this: on a two-classifier action set whose first member is inadequate, Rust
  deletes the first and still processes (marks, increments `exp` of) the second.

## Part B — one full explore episode (the RNG path)

One fixed-seed (42) explore trial per maze. Exact agreement is **not** expected:
the implementations draw from different RNG streams (reset cell, ε-greedy action,
ALP position pick), so trajectories diverge from the first step. Reported for
order-of-magnitude sanity only.

| Maze | pyalcs steps / macro / reliable | Rust steps / macro / reliable |
|---|---|---|
| Woods1-v0 | 1 / 1 / 0 | 11 / 8 / 0 |
| Maze4-v0 | 50 / 26 / 0 | 50 / 38 / 0 |

`knowledge` (the third metric named in BUILD_PLAN P8) is **0 on both sides** after a
single explore trial — it requires ≥1 reliable classifier predicting ground-truth
transitions, and both show `reliable=0`. Computing it properly needs Rust
transition-enumeration that does not exist yet; the substantive knowledge comparison
is deferred to P9.

Same order of magnitude. Woods1 differs most because pyalcs's random reset
happened to land adjacent to the goal (1 step); a single episode is dominated by
reset luck. Both Maze4 runs hit the 50-step truncation cap with no reliable
classifiers yet — expected after only one explore trial. The cross-language
learning-quality comparison is P9 (full protocol, n_exp repeats, exploit window),
not a single episode.

## P9 handoff

Like the §4 bugs, the `apply_alp` mid-iteration skip can cause minor pyalcs-vs-Rust
population/learning differences over a full run — a known semantic gap, not a Rust
regression. Expect Rust to learn slightly *more* (it never drops a classifier the
ALP should have processed). Flag P9 maze-level differences against this when
judging order-of-magnitude parity.

**Thin coverage spot for P9 to watch:** `add_classifier`'s tie-break when **two or
more eligible subsumers** exist for one child (iteration order picks which gets the
numerosity bump). All 761 cases agree, but random populations of 0–5 mostly-distinct
classifiers rarely produce 2+ eligible subsumers for a single child, so this path is
only lightly exercised. Not a gate concern; worth a targeted case if P9 surfaces a
population-count divergence.
