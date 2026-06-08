# Rust workspace architecture decisions

Decisions that are not recoverable from reading the (P2-scaffold) signatures and
that P3–P8 depend on. PROJECT_CONTEXT.md remains the curated source of truth;
this file records implementation-level choices made while executing the build
plan.

## Crate layout and dependency direction

```
acs2-bench (bin)  ──▶ acs2-core, acs2-envs        (composition root)
acs2-envs  (lib)  ──▶ acs2-core
acs2-core  (lib)  ──▶ (no workspace deps; only rand/rand_chacha)
```

`acs2-core` depends on neither `acs2-envs` nor `acs2-bench` (P2 gate). Verified
with `cargo tree -p acs2-core`.

## The Environment port lives in `acs2-core`, not `acs2-envs`

PROJECT_CONTEXT §7's module map lists `environment` under `acs2-envs`, but the
**trait** (`Environment<N>`, `StepOutcome<N>`) is defined in
`acs2-core::environment`. This is forced, not a preference:

- §7/P6 place the single shared trial loop (`Agent`) in `acs2-core`.
- The loop calls `env.reset()` / `env.step()`, so it must name the `Environment`
  trait.
- The P2 gate forbids `acs2-core` depending on `acs2-envs`.

Therefore the port must live in core (Dependency Inversion: the domain defines
the interface; the outer layer implements it). `acs2-envs::maze::Maze`
implements `acs2_core::environment::Environment<8>`. A generic `reset`/`step`
over `Perception<N>` carries no maze specifics, so core keeps "zero environment
knowledge". This mirrors how RNG is already injected via a core trait (§8).

## `ClassifierRef` is a positional index — deletion contract

Match sets and action sets are `Vec<ClassifierRef>` where
`ClassifierRef = usize` indexes into `Population`'s internal `Vec<Classifier<N>>`
(`population.rs`). This deliberately avoids pyalcs's shared-mutable aliasing of
the same classifier objects across population/match-set/action-set.

Consequence that must be honored in P4/P5: a positional index is invalidated by
any mid-`Vec` removal (every later index shifts). **Append never shifts; only
deletion does.** ALP deletion (incorrectly-anticipating ∧ inadequate, SPEC D.6)
*does* fire within the 500 explore episodes, so:

- `apply_alp` is the **sole owner** of population mutation and set consistency.
- Deletion is **deferred to after** the per-item ALP loop (collect victims,
  delete once), never interleaved with iteration over the action set.
- After `apply_alp` returns, the agent **re-derives the match set** before
  selecting an action (match is a pure function of state + population, so this is
  equivalent to pyalcs's "extend match_set with new matchers" and also absorbs
  any deletions). This is the chosen resolution ("option b": plain `Vec` +
  re-scan) over a tombstoning/slotmap store.

A naive `Vec::remove` mid-loop reintroduces an index-aliasing bug that surfaces
only as a P8 differential mismatch, not a compile error — hence this note.

### P8 discovery: pyalcs `apply_alp` mid-iteration skip (a fourth pyalcs hazard)

The deferred-deletion design above is not just defensive — P8 differential testing
proved pyalcs's in-place variant is **buggy**. pyalcs runs
`for cl in action_set: ... action_set.safe_remove(cl)` (`ClassifiersList.py:145-161`):
deleting an inadequate classifier mid-iteration makes Python's list iterator skip
the **next** action-set element, which is then never processed (no
`increase_experience`, no `set_mark`, no anticipation handling). This is a bug in a
**core learning primitive**, beyond the three orchestration bugs in
PROJECT_CONTEXT §4. Rust iterates a cloned `original_action_set` and defers
deletion, so it processes every member — the correct ACS2 behavior.

Consequences honored here:
- Rust is the **correct side**; do not transcribe the skip (§4/§J: re-derive cleanly).
- The P8 differential **excludes skip-tainted inputs** from the exact-match gate
  (detected via the survivor whose `exp` was not incremented) and counts them as a
  distinct bucket, separate from RNG exclusions. The component operations still
  agree (isolated `unexpected_case` matches Rust), so Rust's loop is the correct
  composition; only pyalcs's loop diverges.
- `p8_differential.rs::rust_processes_whole_action_set_where_pyalcs_skips_after_deletion`
  pins the correct behavior. See `reports/P8_differential.md`.
- **P9 note:** this can make Rust learn marginally more than pyalcs over a full run
  (it never drops an ALP-due classifier) — a known semantic gap, not a regression.

### P8 differential harness contract

`baseline/dump_differential.py` is the instrumented pyalcs oracle; it must run on a
COPY and never modify pyalcs. It emits three buckets — deterministic-kept (the
gate), RNG-excluded (`expected_case` ≥2 `get_differences` candidates, the sole ALP
RNG source, filtered analytically + 12-seed invariance), and pyalcs-bug-excluded
(above). Population comparison is canonical-sorted by `(condition, action, effect)`,
which is a valid unique key because `add_classifier` merges rather than duplicating.
Match/action/next-match sets are compared as ascending index lists (both sides build
from the same input array order). Part B (one explore episode) is **report-only**:
different RNG streams make exact agreement impossible, so it is an order-of-magnitude
sanity check, not an assertion. The cross-language learning-quality comparison is P9.

## P9 benchmark methodology (the speedup claim)

`acs2-bench` (bin) and `baseline/run_pyalcs_maze.py` run the **identical** §5 protocol
(500 explore @ ε=0.8, then 3×200 exploit; final-window mean steps; n_exp=10; GA OFF)
and emit the **same CSV schema** so `baseline/compare_bench.py` can join them.
Methodology constraints that make the timing claim valid:

- **Optimized Rust only.** The timed binary is `target/release/acs2-bench` built with
  `cargo build --release`. A debug build can read *slower* than CPython and would
  invert the result — never time `cargo run`/debug.
- **Sequential, uncontended.** The pyalcs and Rust timed runs never overlap (they
  contend for cores); run one, then the other, on the same machine.
- **Timed region is symmetric.** Only explore+exploit are timed; population/reliable
  metrics are computed *after* the timed region on both sides (pyalcs builds them
  post-`explore`/`exploit`; Rust reads `population()` after the loops).
- **`do_ga` is one flag on both sides** (`--do-ga`), default OFF for the first
  comparison (P9), so a single value sets GA identically (PROJECT_CONTEXT §5).
- **Std is population std (÷N, `pstdev`)** on both sides; **steps std is over the
  n_exp per-repeat final-window means**, not over individual trials.
- **Two speedup readings, both reported:** per-maze `t_py/t_rust` and the total-time
  `Σt_py/Σt_rust` (Woods100-dominated). The headline is the total-time ratio, labelled.
- **Correctness gate = order of magnitude** (`compare_bench.py` flags any maze whose
  exploit-steps ratio exceeds 2× as INVESTIGATE). Rust learning marginally *more*
  (smaller/cleaner population, equal-or-fewer steps) is expected from the P8 `apply_alp`
  skip, **not** a regression.
- **RNG is two injected ChaCha streams** (env reset + agent), both seeded per repeat;
  reproducible within Rust but not bit-identical to pyalcs's `random`/`numpy` (§5
  accepts this — the comparison is wall-clock + learning quality). See
  `reports/P9_comparison.md`.

## Other choices

- **No `cfg` field in `Classifier`** (unlike pyalcs). `beta` and the `theta_*`
  thresholds are threaded as method parameters. This makes `copy_from` a pure
  field copy and keeps the classifier free of a shared-config God object.
- **Alloc-free seams.** `ActionSelector::select` and `BootstrapEstimator::estimate`
  take `(&Population<N>, &[ClassifierRef])` rather than `&[&Classifier<N>]`, to
  avoid building a `Vec<&Classifier>` on every action selection / bootstrap
  estimate (§8: minimize per-step heap churn).
- **Const-generic genome length `N`.** Core carries no environment dimension;
  the maze fixes `N = 8` at the edge (`MAZE_PERCEPTION_LEN`).
- **Subsumption is a relational strategy module** (`subsumption.rs`, added in P3),
  not methods on `Classifier`: free `is_subsumer(&Classifier, theta_exp, theta_r)`
  and `does_subsume(cl, other, theta_exp, theta_r)`, mirroring pyalcs
  `lcs/strategies/subsumption.py` (SPEC F.4). ALP (P4/P5) and GA (P5) reuse it.
  `Condition::subsumes` itself is the **symmetric** per-position permissive rule
  (`a==# || b==# || a==b`, identical to `does_match`) exactly as pyalcs
  `acs/Condition.py`; classifier-level subsumption is made asymmetric by the
  separate `is_more_general` (strict `specificity <`) guard inside `does_subsume`,
  not by `Condition::subsumes`.
- **`Symbol::Token(u8)` assumes a single-byte alphabet** (holds for maze codes
  `'0'/'1'/'9'`; fixtures also use `'2'`). Revisit if a non-maze environment with a
  multi-byte/wider alphabet is ever added.
- **`Symbol = enum { Wildcard, Token(u8) }`** (Copy/Ord). `Mark<N>` is
  `[BTreeSet<Symbol>; N]`. `Classifier` identity (`PartialEq`) is
  `(condition, action, effect)` only — implemented manually, no `Hash`.
- **RNG.** `acs2-core::rng::RandomSource` is the injected trait (object-safe:
  `gen_bool`, `gen_range`, `gen_unit`; `shuffle` is a free function). `ChaChaRandomSource`
  wraps `ChaCha8Rng` seeded via `seed_from_u64`. `rand`/`rand_chacha` use
  `default-features = false` so `getrandom`/OS entropy is **not** linked into the
  domain — determinism comes from the seed (§8), not entropy.

## Truncation contract — the trial loop has NO internal step cap (read before writing `Maze::step`)

Decided in P6, binding on P7. The shared trial loop (`agent.rs`,
`run_explore_trial` / `run_exploit_trial`) follows Gymnasium delegation: it has
**no internal iteration limit** and runs until the `Environment` sets
`terminated || truncated`. Consequences that `acs2-envs::Maze` MUST honor:

- **`Maze::step` MUST set `truncated = true` once the per-episode step count reaches
  that maze's registered `max_episode_steps`.** Per PROJECT_CONTEXT §5: **50** for
  Maze4/5/7 and Woods1; **500** for Woods100/101/102. Use each maze's registered cap.
- **The cap is per-maze data, stored with the geometry**, and must be **identical on
  the Rust runner and the pyalcs baseline** (`baseline/run_pyalcs_maze.py`) for the
  comparison to be valid.
- **If the maze never raises `truncated`, a deterministic exploit trial on a maze the
  frozen policy cannot solve will infinite-loop** — the loop has nothing else to stop
  it. This is not a safety net the agent provides; the environment owns it.
- **On `terminated` OR `truncated` the loop runs one terminal learning pass with
  bootstrap `0`.** This is correct for parity: gym 0.23 folds `TimeLimit` into the
  single `done` flag, so pyalcs also bootstraps `0` at the cap. **P8 asserts this and
  must NOT "fix" it** to bootstrap on truncation — a truncation-bootstrap variant would
  diverge from the baseline being measured.

## P7 maze model — static geometry + agent coordinate (not matrix mutation)

Decided in P7. pyalcs's `gym_maze` mutates the matrix in place (old cell → PATH,
new cell → ANIMAT) and re-derives the agent position by scanning for the ANIMAT
marker (`abstract_maze.py`, `maze_impl.py`). `acs2-envs::Maze` instead keeps the
geometry **immutable** (`Vec<Vec<u8>>` of raw `0/1/9`) and tracks the agent as a
`(agent_row, agent_col)` pair. This is provably equivalent and alloc-free per step:

- The agent's own cell is never in its own 8-neighbour view, and every non-agent
  cell always equals its static geometry value, so perception can read the static
  grid directly. The ANIMAT marker the Python code writes is therefore never
  observed and need not be modeled.
- **Post-goal perception matches** because entering the reward leaves the agent
  coordinate *on* the reward cell (pyalcs's `get_animat_xy` returns the reward
  coordinate once no ANIMAT remains), so both sides read the reward cell's
  neighbours.

### Neighbour / move ordering — confirmed against source, not assumed

The eight offsets `N,NE,E,SE,S,SW,W,NW` use the **`maze_utils.adjacent_cell_values`
convention** (x = row, y = col):
`(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)`. `maze_impl.move` shares
this exact convention, so the **same offset table drives both perception and
transition** (`NEIGHBOUR_OFFSETS` in `maze.rs`). The sibling helper
`maze_utils.get_possible_neighbour_cords` uses a *different* (x = col) ordering but
is reachable only from the off-by-default action-planning path — it is **not** used.

### Cell-code encoding — ASCII byte, matching `map(str, …)`

pyalcs stringifies cell codes (`abstract_maze.py:26` → `'0'/'1'/'9'`) and the agent
consumes `str` perceptions. `acs2-core`'s `Symbol::Token` holds the **ASCII byte**
(`tests/common/mod.rs` builds tokens via `value.as_bytes()[0]`). The maze emits
`Symbol::Token(b'0' + cell)` (= `b'0'/b'1'/b'9'`), **not** numeric `Token(0/1/9)`.
A numeric encoding compiles fine but would be silently rejected by `does_match` as a
P8 divergence — hence this note.

### Truncation off-by-one — gym 0.23 `TimeLimit` semantics

`Maze::step` increments `elapsed_steps`, then sets
`truncated = !terminated && elapsed_steps >= max_episode_steps`. A cap of 50 yields
exactly 50-step episodes; **if goal and cap coincide on the same step, `terminated`
wins and `truncated` stays false** (gym sets `TimeLimit.truncated = not done`). Both
fold to bootstrap-0 so learning is unaffected, but steps-to-goal metrics depend on
this boundary.

### Differential probes (Gate P7)

`baseline/dump_maze_probes.py` (runs in the pyalcs venv; imports maze classes
**directly from `gym_maze.envs`** to sidestep the Woods `gym_woods` registration
entry-point) emits `fixtures/maze_probes.json`: per maze the full grid plus an
**exhaustive, seedless** probe set over (every path cell × 8 actions) — positions
are set explicitly, so no RNG enters the comparison. `done` is captured from the raw
env (no `TimeLimit` wrapper), i.e. termination only; truncation is covered by
separate Rust unit tests. `tests/maze_parity.rs` asserts embedded geometry ==
dumped grid (definitive matrix-match) and perception/transition/reward/terminated
parity. P7 ports **Maze4, Maze5, Maze7, Woods1 (cap 50), Woods100 (cap 500)** —
both cap values exercised; 888 probes total.
