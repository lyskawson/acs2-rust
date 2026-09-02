# BUILD_PLAN.md — sequenced prompts for the Rust ACS2 baseline

Use with `PROJECT_CONTEXT.md` and `SPEC_PYALCS.md`. Work one phase at a time; do not
start a phase until the previous phase's **gate** passes. Every prompt assumes the
agent has both repos cloned locally and read access to the files; for any prompt that
touches pyalcs, point the agent at the local `acs2vcp-python/pyalcs/` tree.

Recommended agent: a coding agent with filesystem + repo access (e.g. Claude Code).
For the heavier reasoning prompts (P1, P5, P7) enable extended/ultrathink reasoning.

A standing rule for every prompt: **no comments inside code; English identifiers;
SOLID + Clean Architecture; the domain crate has no I/O; do not transcribe the pyalcs
orchestration bugs listed in PROJECT_CONTEXT §4.**

---

## P0 — Environment bring-up (Python baseline)

```
Set up an isolated Python environment that can run the pyalcs ACS2 maze scripts
unchanged. Constraints:
- pyalcs depends on gym==0.23.0 (classic Gym). Pin a Python version and dependency set
  in which `acs2vcp-python/pyalcs` and `acs2vcp-python/openai-envs` install and import
  cleanly. Prefer a reproducible setup (uv or a pinned venv); document the exact
  Python version and pinned packages in a README.
- Do NOT modify pyalcs or openai-envs source. The baseline must run as-is.
- Produce a single script `baseline/run_pyalcs_maze.py` that takes a maze id (e.g.
  Maze4-v0), runs the protocol from PROJECT_CONTEXT §5 with a fixed seed (seed both
  `random` and `numpy.random`), and prints per-trial steps-to-goal plus total
  wall-clock time. Verify it runs end-to-end on Maze4-v0 and one Woods maze.
Deliver the env setup, the script, and the README.
```

**Gate P0:** the baseline script runs on at least two mazes and reports steps + time.

---

## P1 — Test oracle (golden vectors)

```
Read SPEC_PYALCS.md and the pyalcs source under acs2vcp-python/pyalcs/lcs/.
Produce a language-neutral set of golden test vectors in JSON under `fixtures/`,
to be consumed later by both Python and Rust. Do NOT write Rust yet.

Generate fixtures covering, at minimum:
- condition matching (wildcard on either side; full/partial/no match);
- effect.does_anticipate_correctly (wildcard pass-through vs specified change);
- mark.get_differences behavior (single-position vs all-positions specialization);
- ALP expected_case output (specialized child or None);
- ALP unexpected_case output (effect specialization + mark);
- ALP covering output;
- q/r/ir/tav update arithmetic for a few step counts (cross the exp < 1/beta boundary);
- subsumption does_subsume true/false cases;
- population numerosity collapse on identical (C,A,E).

Two sources, combined:
1. Extract concrete cases from the pyalcs unit tests under
   tests/lcs/agents/acs2/ and tests/lcs/strategies/. Cite the originating test.
2. Instrument a COPY of pyalcs (never the original) to dump intermediate states to
   JSON for a handful of fixed-seed transitions.

Each fixture: inputs, expected outputs, and a `source` citation (test name or
"instrumented trace, seed=N"). Aim for >= 30 fixtures across the categories above.
Deliver fixtures/ plus a short index README listing what each file covers.
```

**Gate P1:** ≥30 fixtures, each independently verifiable against pyalcs.

---

## P2 — Rust workspace scaffold

```
Read PROJECT_CONTEXT.md §7 and §8. Scaffold the Cargo workspace EXACTLY as pinned
there: crates acs2-core (lib), acs2-envs (lib), acs2-bench (bin); leave acs2-py out
for now. Create the module files listed under each crate with type/trait signatures
and `todo!()` bodies. Define:
- the Symbol representation (wildcard-aware) and Condition/Effect/Mark types;
- the Classifier struct with all fields from SPEC_PYALCS §A;
- the Configuration struct with all parameters from PROJECT_CONTEXT §5;
- the ActionSelector trait (the actor seam) and an empty EpsilonGreedy/BestAction/
  RandomAction;
- an Rng abstraction trait (inject rand::Rng);
- the Environment trait (Gymnasium-style: reset, step -> obs/reward/terminated/
  truncated/info);
- the Agent trial-loop skeleton, with the RL bootstrap value taken as a parameter
  (the critic seam).
No logic yet — only signatures, doc-free, that compile. `cargo check` must pass with
warnings allowed. No comments in code. English identifiers. SOLID; acs2-core has no
dependency on acs2-envs.
Deliver the compiling workspace.
```

**Gate P2:** `cargo check` green; layer boundaries respected (core depends on neither
envs nor bench).

---

## P3 — Core data types + matching

```
Implement, in acs2-core, against SPEC_PYALCS §A and the relevant fixtures:
- Symbol, Condition (does_match, specificity, subsumes), Effect
  (does_anticipate_correctly, is_specializable, specialize), Mark (set_mark,
  is_marked, get_differences), Classifier (fitness=q*r, copy, identity = (C,A,E)).
Use a straightforward element-wise representation (no bit-packing yet — see
PROJECT_CONTEXT §7 representation note). Add unit tests that load the matching /
effect / mark / subsumption fixtures from fixtures/ and assert equality.
No comments; English identifiers.
Deliver the implemented types and passing tests.
```

**Gate P3:** all matching/effect/mark/subsumption fixtures pass.

---

## P4 — Population + covering + action selection

```
Implement, in acs2-core, against SPEC_PYALCS §B, §D.4, §F:
- Population: form_match_set (linear scan over a clean Vec-based store),
  form_action_set, numerosity collapse on identical (C,A,E), subsumption-aware
  insertion, ALP-style removal.
- ALP covering (cover): fresh classifier with executed action, experience=0,
  reward=0, then specialize to the observed change.
- ActionSelector impls: EpsilonGreedy (explore), BestAction (exploit, filters to
  change-anticipating classifiers, ranks by fitness*num, uniform-random fallback),
  RandomAction. RNG is injected.
Add unit tests for numerosity collapse and match/action-set formation using fixtures.
No comments; English identifiers.
Deliver implementation + tests.
```

**Gate P4:** population + covering + selection fixtures pass; numerosity collapse
verified.

---

## P5 — ALP + RL + GA (the learning core)

```
Implement, in acs2-core, against SPEC_PYALCS §C, §D, §E:
- q/r/ir updates and the tav MAM rule (respect the exp < 1/beta boundary).
- ALP: expected_case (specialize toward mark; respect u_max generalization being
  effectively disabled at u_max=100000), unexpected_case (effect specialization +
  mark + decrease_quality), covering trigger, deletion of incorrectly-anticipating
  inadequate classifiers.
- RL update with the bootstrap target supplied as a parameter (critic seam); the
  default caller passes max q*r over change-anticipating next-state match-set
  classifiers, or 0 at episode end.
- GA: q^3-weighted (note: pyalcs weights by q^3 * num) roulette parent selection,
  generalizing mutation (specified -> wildcard with prob mu), two-point crossover on
  the CONDITION only, gated on equal effects; action-set-size deletion against
  theta_as. GA is implemented FULLY but gated behind the runtime config flag `do_ga`
  (default OFF). The agent loop must only invoke GA when `config.do_ga` is true, so GA
  can be switched on/off per run without code changes; the same flag will later be set
  identically on the pyalcs baseline.
Use the q/r/ir/tav, ALP, and subsumption fixtures as tests. Do NOT copy the pyalcs
GA deletion bug (is_marked bound-method); implement the intended tie-break correctly
and note it. No comments; English identifiers.
Deliver implementation + tests.
```

**Gate P5:** all learning-core fixtures pass.

---

## P6 — Agent trial loop

```
Implement, in acs2-core, the single shared trial loop (explore and exploit) against
SPEC_PYALCS §B, re-derived cleanly (do NOT transcribe the pyalcs copy-pasted loops).
- explore: form match set; apply ALP to the previous action set; apply RL; optional
  GA (config flag); select action via the configured ActionSelector; step env; on
  done run the terminal learning pass with bootstrap 0.
- exploit: form match set; apply RL only (no ALP/GA/covering); always BestAction.
- Return per-trial steps-to-goal and accumulated reward.
- The bootstrap value is computed by a small injected component so a critic can
  replace it later without touching the loop.
Add a smoke test on a tiny hand-built maze (or a stub Environment) asserting the loop
runs and the population grows. No comments; English identifiers.
Deliver implementation + smoke test.
```

**Gate P6:** loop runs on a stub env; population grows; no panics.

---

## P7 — Maze environment (Gymnasium-style, 8-sensor)

```
Implement acs2-envs against PROJECT_CONTEXT §5 (maze semantics) and SPEC_PYALCS §G.2.
- Environment trait following Gymnasium conventions: reset() -> observation;
  step(action) -> (observation, reward, terminated, truncated, info). terminated on
  reaching the reward cell; truncated handled by the runner's step cap.
- Maze: 8-sensor perception in order N,NE,E,SE,S,SW,W,NW; cell codes path 0, wall 1,
  reward 9 (own cell never in view); 8 compass actions; move into wall stays put;
  reward 1000 on goal else 0; reset to a uniform random path cell (injected RNG).
- maze_data: port the canonical maze geometries from acs2vcp-python/openai-envs/
  gym_maze (matrices for the standard suite: Maze4, Littman57, Woods1, etc.). Keep
  geometry data separate from logic.
Add a property test: for random (position, action) the Rust maze transition and
perception match the pyalcs gym_maze for the same maze id. (Drive the comparison via
a small Python dumper using the pyalcs env.)
No comments; English identifiers.
Deliver implementation + property test + a dumper script for the comparison.
```

**Gate P7:** Rust maze transitions/perception match pyalcs gym_maze on random probes
for at least three mazes.

---

## P8 — Differential test (Rust vs pyalcs)

```
Build a differential harness:
- For N random (state, small population) inputs, compare match set, action set, and
  population-after-one-learning-step between Rust and a pyalcs instrumented dumper, on
  identical inputs with RNG removed from the path. They must agree exactly.
- For one full episode with a fixed seed on one maze, compare end-of-episode metrics
  (population size, reliable count, knowledge) between Rust and pyalcs; differences are
  acceptable only where RNG enters.
Produce a short report (table of agreements/divergences). Investigate and document any
deterministic-path divergence.
No comments in shipped code; English identifiers.
Deliver harness + report.
```

**Gate P8:** zero divergence on the deterministic path; RNG-only divergences explained.

---

## P9 — Benchmark + comparison

```
Implement acs2-bench:
- Expose `do_ga` as a command-line flag on the bench, and add the same flag to
  baseline/run_pyalcs_maze.py, so a single value sets GA identically on both sides.
- Run the PROJECT_CONTEXT §5 protocol with **GA OFF** for this first comparison over
  the full maze suite with n_exp=10, fixed seed, recording per-maze: exploit
  steps-to-goal (mean/std over the exploit window), macro/micro population, reliable
  counts, and wall-clock time.
- Emit a CSV. Run the SAME protocol/seeds/mazes/flag through the pyalcs baseline.
- Produce a comparison notebook/script: per-maze steps Rust vs pyalcs (sanity: same
  order of magnitude = same algorithm), and speedup t_python/t_rust per maze and
  overall, on the same machine.
Report both tables. Flag any maze where Rust steps differ from pyalcs by more than a
chosen tolerance as a correctness regression to investigate, not a speed win.
A GA-ON run is a separate later invocation of the same binaries (same flag set true on
both), not part of this first comparison.
Deliver the bench binary, the extended baseline script, the CSVs, and the comparison.
```

**Gate P9:** Rust learns each maze to the same order of magnitude as pyalcs; speedup
table produced on one machine.

---

## After P9 (out of scope for this phase)
- Decision point: is the Rust speedup large enough to justify continuing? Document it.
- Only then: introduce the Actor-Critic via the two seams (a new ActionSelector for the
  actor; a learned V(s') at the bootstrap call site for the critic), and optionally the
  acs2-py PyO3 Gymnasium bindings.
- Optional: add the bit-packed Condition/Effect representation behind the existing
  interface as a pure optimization, re-running P3–P5 fixtures to confirm parity.
- Optional secondary baseline: time the supervisor's ALCS cpu_single (with its quirks
  intact) as an "even vs optimized Python" datapoint — never as the AC foundation.

## MPX phases (Task 2) — overview

M1 (build + correctness), M2 (speed/reach), M3 (conditional pack) are coordinated
via per-phase agent prompts, gated like P0–P9. M1 is the correctness gate (the MPX
analogue of P8); M2 measures reach + per-component cost; M3 only fires if M2 proves a
ceiling.

## M3 — Bit-packed representation (CONDITIONAL, in-scope-of-MPX, LAST)

This is NOT MPX preparation and NOT a generic "after everything" optimization. It is
the conditional final phase WITHIN MPX, run ONLY if the M2 reach measurement proves a
specific component is the ceiling (memory or speed).

**Preconditions (all must hold before M3 is written or run):**
- M1 gate passed (MPX correctness: knowledge=1.0 on 6/11/20; analytic oracle green;
  sampled knowledge ≡ exhaustive at k=11/20).
- M2 produced a reach curve WITH per-component memory/time measurement
  (Mark vs condition vs effect) AND named which component is the ceiling. Packing
  without this measurement is forbidden: the likely large-N memory ceiling is
  `Mark = [BTreeSet<Symbol>; N]` (one BTreeSet per attribute), NOT condition/effect,
  so packing condition "because it's obvious" may optimize the wrong component.
  Measurement decides what to pack, not intuition.

**What M3 does (only if triggered):**
- Pack ONLY the component M2 proved is the ceiling, behind the existing acs2-core
  interface (Condition/Effect/Mark surface unchanged; element-wise and packed are
  interchangeable impls).

**Re-validation (mandatory — this is rewriting the validated core, not adding an
optimization):**
- Full P8 re-run: packed matching must reproduce the element-wise differential EXACTLY
  (761/761 zero divergence).
- Full P9 + MPX re-run: existing maze speedup numbers and MPX knowledge/reach numbers
  must NOT move (packed is a representation change, same algorithm).

**May be unnecessary:** if element-wise reaches MPX-137 in M2 without a ceiling, M3 is
NOT run, and "element-wise sufficed to MPX-137" is itself a reportable result.

**Opportunity exception:** if, AFTER MPX is closed, bit-packing is judged a worthwhile
contribution in itself ("packed representation gives an additional X× at preserved
correctness, same algorithm"), it is a legitimate deliberate phase WITH full
re-validation, reported as a separate element-wise-vs-packed result. That is a post-hoc
decision made WITH measurement in hand — never before MPX validation.

**Gate M3 (only if run):** packed component reproduces P8 761/761 exactly; P9 maze
numbers and MPX numbers unchanged; the M2-identified ceiling measurably relieved.
