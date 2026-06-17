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

## Deliberate pyalcs-fidelity deviations from canonical ACS2 (Butz & Stolzmann)

These are NOT bugs and NOT representation choices — they are behaviours where pyalcs
under its shipped config diverges from canonical ACS2, and the Rust port reproduces
them ON PURPOSE so the P9 benchmark compares the same algorithm on both sides
(PROJECT_CONTEXT §2: port target is pyalcs-as-code, not the Butz paper). Collected
here as the fidelity ledger for the thesis and as the revisit-checklist for the
Actor-Critic stage, where the port target no longer applies.

- **ALP generalization disabled by `u_max = 100000`** (SPEC J.5). The expected-case
  over-specialization generalization branches never execute; under the shipped config
  ALP only specializes. Canonical ACS2 generalizes via ALP. To restore Butz behaviour:
  lower `u_max` and implement the generalization while-loops (currently omitted, noted
  in expected_case).
- **Exploitation ignores epsilon — always BestAction** (SPEC J.8, PROJECT_CONTEXT §4).
  The epsilon values in the exploit scripts are inert. Kept as a property of the
  measured algorithm.
- **`biased_exploration` is dead config** (SPEC J.3). EpsilonGreedy reads `epsilon` and
  discards the biased-exploration parameter; exploration is pure epsilon-greedy with no
  action-delay / knowledge-array bias, even though those strategy classes exist in
  pyalcs.
- **No generalization pressure under the benchmark config.** With u_max generalization
  dead AND GA OFF (the P9 protocol), structural learning is specialize-only. This is
  faithful to the tested pyalcs maze scripts; it is also why population counts stay
  high. Revisit if GA-ON or AC changes the regime.

Note: RL bootstrap on q·r is NOT in this list — SPEC J.6 confirms it MATCHES Butz, so
it is canonical, not a deviation. The four pyalcs bugs (the §4 hazards + the P8
apply_alp skip) are also NOT here — Rust does not reproduce those; it is the correct
side. This list is only behaviours Rust intentionally mirrors that depart from the
paper.

## M1 — Multiplexer (Task 2, phase 1)

M1 adds the Multiplexer (MPX) task as a NEW environment plus a NEW knowledge-evaluation
capability, reusing the P0–P9-validated `acs2-core` domain unchanged. The maze measured
path is byte-untouched: the only edits to existing files are additive `pub mod` lines
(`acs2-core/src/lib.rs`, `acs2-envs/src/lib.rs`) and an additive `Configuration::mpx()`
(`config.rs`); everything else is new files (`acs2-envs::multiplexer`,
`acs2-core::knowledge`, `acs2-bench/src/bin/mpx.rs`). All P3–P8 + maze-parity tests stay
green.

### MPX is anticipatory; the oracle is analytic, not differential

The percept is `[k input bits] + [1 trailing validation bit]`, `N = k + 1`. `reset()`
samples the k input bits via the injected RNG and pins the validation bit to 0; `step`
is single-step (`terminated = true` always), leaves the input bits unchanged, and on the
correct answer flips the validation bit 0→1 (reward 1000) — otherwise leaves it 0 (reward
0). So there is a real deterministic transition to anticipate. pyalcs's MPX env generates
states randomly per reset and ships NO knowledge harness, so there is no enumerable
transition set to diff against (unlike the maze's P7/P8 probes). M1 therefore validates
against CLOSED-FORM truth: `get_correct_answer` (address-bits-first, MSB-first,
`data[address]`) is pinned by the exact pyalcs `gym_multiplexer/tests/test_utils.py`
3-/6-/11-bit vectors ported as unit tests. "Better accuracy" does not exist for MPX —
knowledge has a hard ceiling at 1.0 reached by every correct implementation.

### `N = k + 1` enforced by a const fn on a monomorphized path

`control_bits_for(N)` (a `const fn`) inverts `N = control_bits + 2^control_bits + 1` and
`panic!`s at const-eval for any N that is not a valid multiplexer size. `Multiplexer<N>`
references `Self::CONTROL_BITS` inside `new()` so the assertion lands post-monomorphization
(associated consts are lazily evaluated — an unreferenced one would not fire). The
`define_multiplexers!` macro instantiates the explicit set by CONTROL BITS, deriving
`N = c + 2^c + 1`, so N is never hand-typed. A runtime backstop test asserts perception
length == k+1 for every instantiated size.

**Size-set correction (137 → 135).** The M1 prompt lists the MPX set as
`{6,11,20,37,70,137}` → `N ∈ {7,12,21,38,71,138}`. `137` is not a valid multiplexer
input size: `k = c + 2^c` gives `135` at `c = 7` (`N = 136`), never `137` — so
`control_bits_for(138)` would (correctly) panic. The canonical LCS multiplexer sequence
is `3,6,11,20,37,70,135`. M1 therefore instantiates `{6,11,20,37,70,135}` →
`N ∈ {7,12,21,38,71,136}` (`Mpx6..Mpx135`). This is the const-assertion doing its job,
and is consistent with PROJECT_CONTEXT §2 (correctness over reproducing a typo) and the
M1 priority clarification (closed-form truth beats matching any artifact).

### Knowledge metric — env-agnostic core fn + MPX transition generators

- `acs2-core::knowledge` (pure, zero env knowledge): `Transition<N> { p0, action, p1 }`
  and `anticipation_fraction(population, theta_r, transitions)` — the fraction of
  transitions for which SOME reliable (`q > theta_r`) classifier `predicts_successfully`
  (`action` match ∧ `does_match(p0)` ∧ `does_anticipate_correctly(p0,p1)`). This mirrors
  pyalcs `_maze_knowledge` / `Classifier.predicts_successfully` exactly, retargeted from
  maze transitions to MPX (input × action) pairs.
- `acs2-envs::multiplexer` builds the MPX transitions: `exhaustive_transitions` (all
  `2^k` inputs × 2 actions) and `sampled_transitions` (fixed-seed, WITH REPLACEMENT).
- **`exhaustive_knowledge` (fast exact bitset).** The naive per-transition × per-classifier
  scan is `O(2^(k+1) × |reliable|)` — pathological for the LARGE specialize-only MPX
  populations (see below). Instead, iterate reliable classifiers and mark the
  `(input, action)` cells each covers into a `2^(k+1)`-bit set; popcount at the end. Cost
  ≈ Σ over reliable of `2^(free input positions)` ≈ `|reliable|` when classifiers are
  specialized (the actual regime) — instant. This is EXACT enumeration (every pair
  evaluated), satisfying the gate's "exhaustive"; a unit test pins
  `exhaustive_knowledge == anticipation_fraction(exhaustive_transitions)` on a trained
  k=6 population and a synthetic k=11 population. Two silent-corruption edges baked in: a
  classifier whose effect specifies any INPUT position covers nothing (inputs never
  change); a classifier whose condition pins validation = 1 never matches `p0`
  (validation is always 0 in `p0`).
- `evaluate_knowledge`: exhaustive (fast bitset) for `k ≤ 20`, sampled for `k ≥ 37`.

**Sampler gate (the only checkable place).** The sampled estimator must agree with
exhaustive at k=11 AND k=20. The test runs on a SYNTHETIC ~50%-coverage population (four
reliable classifiers keyed on input bit 0: a change predictor + a no-change predictor per
action), which yields EXACTLY 0.5 exhaustively (non-vacuous — a converged 1.0 population
would make the test trivially pass). With 200 000 sampled inputs the binomial band is
sub-percent; tolerance 0.005.

### NAMED HAZARD — `does_anticipate_change` vs `does_anticipate_correctly`

These look interchangeable and are NOT. Conflating them is the M2 / Actor-Critic trap.

- `does_anticipate_change` (`effect.specify_change()`) gates ACTION SELECTION:
  `action_selection::best_change_anticipating_action` (exploit/BestAction and the greedy
  branch of EpsilonGreedy) considers ONLY change-anticipating classifiers. No-change
  classifiers are invisible to it.
- `does_anticipate_correctly` is what the KNOWLEDGE metric uses, and it CREDITS the
  no-change (identity) transition: for the wrong action `p1 == p0`, an all-wildcard-effect
  reliable classifier anticipates correctly. Without this, the wrong-action half of the
  pairs is never credited and knowledge silently caps at 0.5 masquerading as a learning
  plateau. The ceiling is 1.0 by construction; `acs2-core::knowledge` tests
  (`change_only_population_caps_at_half`, `wildcard_effect_credits_identity_transition`)
  pin this.

Timed-region invariant. The "knowledge is computed outside the timed region" property
holds SOLELY because the bench computes the metric AFTER the timer wraps explore+exploit
(the P9 timed-region symmetry), and never calls it inside the learner loop. It does NOT
rest on any "reliability is frozen" argument: reliability (`q > theta_r`) is driven by
ALP's q-updates (`q += beta·(1-q)` on correct, `q -= beta·q` on incorrect), which fire
THROUGHOUT explore — the phase where knowledge is actually learned. RL
(`apply_reinforcement_learning`) moves only `r`/`ir`, never `q`; that is true but
irrelevant to the timing invariant (q is merely quiescent in exploit, which runs no ALP).

### M1 results and the k=20 anticipation/covering INVESTIGATE signal

Trial schedule anchored to the published ACS2 boolean-multiplexer protocol (explore
ε = 0.8, single-step trials, GA OFF, `u_max = 100000`). There is no `multiplexer_11.yaml`
in the repo to mirror — that is an M2 cpu_single artifact (GA-on, 1000/−1). Per-size
explore budgets are ordered `6 < 11 < 20`; the literature figures assume GA-ON
generalization, whereas M1 is GA-OFF specialize-only, so MORE explore is expected.

| size | explore (measured) | knowledge | reliable pop | mean spec / N |
|------|--------------------|-----------|--------------|---------------|
| 6    | 20 000             | 1.0 (min 1.0) | ~260      | 5.63 / 7      |
| 11   | 400 000            | 1.0 (min 1.0) | ~4 380     | 9.01 / 12     |
| 20   | —                  | INVESTIGATE   | explodes   | 20.5 / 21     |

(`specificity()` counts over all N condition positions; the compact rule a generalizing
ACS2 would form specifies only ~`control_bits + 1` of them — the measured 9/12 and 20.5/21
are the over-specialization that specialize-only produces.)

k=6 and k=11 reach knowledge = 1.0. **k=20 cannot, in the M1 GA-OFF specialize-only
regime, and this is a measured anticipation/covering signal, not an implementation
defect.** With ALP generalization dead (`u_max = 100000`) and GA OFF, structural learning
is specialize-only: reliable conditions specialize to ~full length (mean specificity
20.5/20 at k=20 — one classifier per input), so the population scales with the `2^k` input
space. Measured at k=20: 168 956 classifiers after just 2 000 explore trials
(knowledge 0.134), with per-trial cost growing as the macro population explodes. Reaching
1.0 would require ~`2^21` reliable classifiers and a budget many orders beyond any generous
multiple of the literature figure — infeasible. Per the M1 tuning boundary, the budget is
NOT inflated; k=20 is reported as INVESTIGATE. This is precisely the spec author's
"compactness is a reach concern (M2)" framing: M2's generalization (lower `u_max` + the
ALP generalization while-loops, and/or GA-ON) is what forms compact MPX rules. The fast
`exhaustive_knowledge` metric stays tractable even on the exploded 169k population (~2 s),
so the evaluator is validated independently of the learning wall. `Mpx20/37/70/135`
compile and instantiate (const-generic story intact, perception-length test green); they
are simply not in the default bench run.

The MPX bench is a SEPARATE binary (`acs2-bench/src/bin/mpx.rs`); the maze bench
(`src/main.rs`) is untouched, so the P9 maze numbers cannot move. Default run = sizes
`{6, 11}`; `--sizes`, `--explore-trials`, `--skip-knowledge` are available for probing.

## M2a — Multiplexer with GA-on generalization (Task 2, phase 2a)

M2a turns genetic generalization ON (`do_ga=true`) keeping every other M1 parameter,
crucially `u_max=100000` (so ALP still ONLY specializes — generalization comes SOLELY
from GA), and asks whether canonical GA-on ACS2 compactifies MPX and how far up in k it
reaches. The metric is COMPACTNESS, not just knowledge. All GA parameters are the
canonical pyalcs ACS2 defaults — M2a is purely the `do_ga` flag, zero new tuning:
`theta_ga=100` (`lcs/agents/acs2/Configuration.py:18`; SPEC §E.1), `mu=0.3`, `chi=0.8`,
`theta_as=20`, `theta_exp=20`, `do_subsumption=true`. Rust does NOT reproduce pyalcs's
inferred GA-deletion bug (SPEC §E.4 `cl.is_marked` bound-method): `ga.rs:116` calls the
method correctly — Rust is the correct side.

Additive changes only: `--do-ga` on `mpx.rs`; a new `acs2-bench/src/bin/mpx_reach.rs`
(reach + per-component memory + 4-way verdict, with `--time-cap-secs`); `libc` dep on the
bench crate (for `getrusage` peak-RSS). `ga.rs`/`subsumption.rs`/`agent.rs` and the maze
measured path are untouched; P5 GA, P3 subsumption, P8 differential, and maze-parity tests
all stay green, and the maze P9 learning metrics reproduce BYTE-IDENTICAL (GA flag keeps
the maze bench on GA-off).

### GA-on compactifies and clears the M1 wall (gate, n_exp=10)

GA-on reaches knowledge=1.0 on MPX-6/11/20 (10/10 each), driving reliable-classifier
condition specificity toward the compact `a+1` and collapsing the reliable population by
orders of magnitude vs M1 specialize-only:

Knowledge on gate sizes 6/11/20 is **exhaustive-exact** (all 2^k×2 pairs enumerated via
the M1 fast bitset, `k≤20`), so "1.0" here means *every* pair is correctly anticipated,
not a sampled estimate:

| k  | knowledge (min, X/10) | reliable spec (→a+1) | reliable pop (GA) | M1 specialize-only |
|----|-----------------------|----------------------|-------------------|--------------------|
| 6  | 1.0 (10/10)           | 3.04±0.07 (a+1=3)    | 28.1±2.3          | 259.8              |
| 11 | 1.0 (10/10)           | 4.15±0.18 (a+1=4)    | 56.7±3.3          | 4380               |
| 20 | 1.0 (10/10)           | 5.24±0.25 (a+1=5)    | 88.6±3.4          | infeasible (exploded) |

This is NOT the HARD FAIL (knowledge=1.0 with full specificity / ~2^k population): the
reliable population is compact (specificity ≈ a+1, count ≈ 3·2^(a+1), four orders below
2^k at k=20). k=20 — which specialize-only structurally could not reach (M1) — is
solved. Budgets (schedule-tuned to reach 1.0, GA params untouched): k6=20k, k11=200k,
k20=300k. GA-on is also FASTER than M1 specialize-only at equal k (k=11: 200k trials in
~2s vs M1 400k in ~20s) because the population stays small. Results: `reports/mpx_rust_ga.csv`
(M1 baseline preserved in `reports/mpx_rust.csv`).

### Reach (k=37/70/135): boundary found — race won through 37, lost at ≥70

Trials cap for the reach 4-way verdict = **M1-empirical specialize-only extrapolation ×10,
a generous non-convergence proxy — NOT a GA-on budget prediction, NOT literature.** Fit:
M1 same-protocol convergence (k=6→20k, k=11→400k, Δk=5 → ×20) modelled as trials ∝ 2^k,
anchored at k=6 → `estimate(k)=20000·2^(k−6)`; cap = `estimate·10 = 200000·2^(k−6)`
(k=37 ≈ 4.3×10¹³; k=70/135 exceed u64 → clamped to u64::MAX). Deliberately so loose that
MEMORY (5.6 GB RSS = 70% of this 8 GB machine) or TIME (per-repeat wall cap) bind first.
Unold's GECCO Companion '26 figure of 500 exploration episodes is the only in-repo
literature budget and is cited as CONTEXT only — it uses a DIFFERENT protocol (`u_max=1`,
ε=0.1), so it is not the anchor. Verdicts are four SEPARATE outcomes, never merged:
SUCCESS / TRIALS-LIMITED / MEMORY-LIMITED / TIME-LIMITED.

Reach knowledge (k=37/70/135) is the **sampled** estimator (50k inputs; exhaustive is
infeasible at these N), gate-validated against exhaustive at k=11/20 (M1). Reliable
spec/count are measured at the verdict by the reach bin:

| k   | verdict (n)        | knowledge | reliable | reliable spec     | peak macro | peak RSS | mechanism |
|-----|--------------------|-----------|----------|-------------------|------------|----------|-----------|
| 37  | SUCCESS (3/3)      | 1.0       | 163–168 (mean 166) | 7.14–8.32/38 (mean 7.91; a+1=6) | ~34k | 0.08–0.11 GB | GA wins — compact, 1.0 |
| 70  | TIME-LIMITED (3/3) | 0.0004–0.0071 | 57–109 (mean 76) | 46.35–67.97/71 (≫ a+1=7) | ~78–81k | 0.27 GB | RACE LOST |
| 135 | TIME-LIMITED (1)   | 0.0000    | 62       | 135.94/136        | 30.4k      | 0.36 GB  | RACE LOST |

(All rows produced by the same current `mpx_reach` binary; k=37 and k=70 are n_exp=3 at a
unified per-row time cap — k=37 480 s, k=70 600 s.) k=37 is MEASURED compact (mean spec
7.91/38 across 3 seeds — far from full 38, modestly above the ideal a+1=6), corroborating
the counting argument: knowledge=1.0 over 2^37 inputs is impossible with per-input
classifiers, so the ~166 reliable rules must each generalize over ~2^29 inputs. k=70 spans
3 distinct seeds {42,43,44} at a unified 600 s cap: all TIME-LIMITED, all knowledge≈0
(0.0004–0.0071), reliable 57–109, all with spec 46.35–67.97/71 — ≫ the compact a+1=7; a
separate fixed-budget diagnostic also showed reliable FROZEN at 58 across 20k/50k/100k
trials at spec 70.84/71. The boundary contrast is crisp and measured:
**k=37 mean spec 7.91/38 → 1.0** vs **k=70 spec 46–68/71 → ≈0**.

**Largest k reaching knowledge=1.0 (compact): k=37.** At k≥70 the generalize-vs-specialize
race is LOST: the few reliable classifiers that stabilize are heavily OVER-SPECIALIZED
(spec 46–68/71 at k=70, 135.94/136 at k=135 — i.e. ≫ the compact a+1, up to nearly
full N, like M1 specialize-only), so only ~57–109 near-per-input rules ever stabilize and
they cover ≈0% of the 2^k space → knowledge≈0. This is the rare-action-set-visitation race
realized: ALP specializes a rule toward full length before GA generalizes it; once
near-fully specialized it matches ≈1/2^k inputs, so its action set is essentially never
revisited, GA's trigger (`time − mean(tga) > theta_ga`) almost never fires on it, and it
can never be generalized back. The q/2 child-halving compounds it. **It is NOT a GA bug**
(GA generalizes correctly through k=37) and **NOT a memory bound**: at both k=70 and k=135
the binding limit is the RACE (TIME-LIMITED at peak RSS 0.22/0.36 GB — far below the 5.6 GB
cap; the population never approaches the N=136 memory pop-threshold of ~1.44M in the time
budget). Precise scope: *canonical GA-on ACS2 compactifies MPX through k=37 (knowledge=1.0,
general rules); the generalize-vs-specialize race is lost at k≥70 (reliable frozen
fully-specialized, knowledge≈0, race-bound not memory/trials-bound).* This is the
pre-authorized FOUND BOUNDARY (a measured result, budget NOT inflated) and it motivates
**M2b: ALP-generalization via lower `u_max`**, which generalizes inside `expected_case`
independent of action-set revisitation — immune to the race that defeats GA here.
Raw logs (all current `mpx_reach` binary): `reports/mpx_reach_37.txt` (n_exp=3, 480 s cap),
`mpx_reach_70.txt` (n_exp=3 seeds {42,43,44}, unified 600 s cap), `mpx_reach_135.txt` (n_exp=1, 360 s cap).

### Per-component memory — the M3 packing target, measured

`size_of` decomposition of `Classifier<N>` at each reached k (`mpx_reach` component bench).
`Mark<N> = [BTreeSet<Symbol>; N]` dominates and its share GROWS with N, while condition and
effect (`[Symbol; N]`, 2N bytes each) stay small:

| k (N)     | condition | effect | Mark (stack) | classifier | Mark share |
|-----------|-----------|--------|--------------|------------|------------|
| 20 (21)   | 42 B      | 42 B   | 504 B        | 672 B      | 75.0%      |
| 37 (38)   | 76 B      | 76 B   | 912 B        | 1152 B     | 79.2%      |
| 70 (71)   | 142 B     | 142 B  | 1704 B       | 2072 B     | 82.2%      |
| 135 (136) | 272 B     | 272 B  | 3264 B       | 3896 B     | 83.8%      |

Measurement (not intuition) confirms the large-N ceiling is `Mark`: at N=136 a per-classifier
`[BTreeSet;136]` is 3264 B of 3896 B (84%), and the empty-`BTreeSet` stack footprint scales
linearly in N. These `size_of` shares are **stack-only and therefore a LOWER bound** on
Mark's true weight: a populated `BTreeSet` allocates heap nodes on top. Empirically at k=135
the reach run held ~30.4k classifiers at peak RSS 0.36 GB ≈ 11.8 KB/classifier vs the 3896 B
stack size — i.e. ~8 KB/classifier of heap, much of it Mark B-tree nodes — so the real Mark
share exceeds 84%. **M3 packing should target `Mark` first** (e.g. a bitset / small-set
representation), not condition/effect. This is also why the reach RSS-cap pop-threshold is
N-dependent (`5.6 GB / size_of(Classifier<N>)`: 8.3M at k=20 → 1.44M at k=135 — itself an
over-estimate of true capacity, since it ignores Mark heap).