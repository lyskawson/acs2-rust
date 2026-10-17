# Rust workspace architecture decisions

Decisions that are not recoverable from reading the (P2-scaffold) signatures and
that P3–P8 depend on. PROJECT_CONTEXT.md remains the curated source of truth;
this file records implementation-level choices made while executing the build
plan.

## Crate layout and dependency direction

```
acs2-bench      ──▶ acs2-agents, acs2-core, acs2-envs
acs2-measure    ──▶ acs2-agents, acs2-core, acs2-envs, acs2-trajectory
acs2-agents     ──▶ acs2-core, acs2-trajectory
acs2-trajectory ──▶ acs2-core
acs2-envs       ──▶ acs2-core
acs2-core      ──▶ (no workspace deps; rand/rand_chacha and sha2)
```

`acs2-core` depends on neither `acs2-envs` nor `acs2-bench` (P2 gate). Verified
with `cargo tree -p acs2-core`.

## The Environment port lives in `acs2-core`, not `acs2-envs`

The **trait** (`Environment<N>`, `StepOutcome<N>`) is defined in
`acs2-core::ports::environment`. This is forced, not a preference:

- The shared `LearningAgent` contract and exploitation trial live in `acs2-core::trial`;
  the learning agents implementing that contract live in `acs2-agents`.
- The loop calls `env.reset()` / `env.step()`, so it must name the `Environment`
  trait.
- The P2 gate forbids `acs2-core` depending on `acs2-envs`.

Therefore the port must live in core (Dependency Inversion: the domain defines
the interface; the outer layer implements it). `acs2-envs::maze::Maze`
implements `acs2_core::ports::environment::Environment<8>`. A generic `reset`/`step`
over `Perception<N>` carries no maze specifics, so core keeps "zero environment
knowledge". This mirrors how RNG is already injected via a core trait (§8).

## `ClassifierRef` is a positional index — deletion contract

Match sets and action sets are `Vec<ClassifierRef>` where
`ClassifierRef = usize` indexes into `Population`'s internal `Vec<Classifier<N>>`
(`acs2-core/src/representation/population.rs`). This deliberately avoids pyalcs's shared-mutable aliasing of
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
- **Subsumption is a relational strategy module** (`acs2-core/src/learning/subsumption.rs`, added in P3),
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
- **RNG.** `acs2-core::support::rng::RandomSource` is the injected trait (object-safe:
  `gen_bool`, `gen_range`, `gen_unit`; `shuffle` is a free function). `ChaChaRandomSource`
  wraps `ChaCha8Rng` seeded via `seed_from_u64`. `rand`/`rand_chacha` use
  `default-features = false` so `getrandom`/OS entropy is **not** linked into the
  domain — determinism comes from the seed (§8), not entropy.

## Truncation contract — the trial loop has NO internal step cap (read before writing `Maze::step`)

Decided in P6, binding on P7. The ACS2 trial loop (`acs2-agents/src/acs2/mod.rs`,
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

### Learning at an episode boundary

`trial::TruncationMode` is an agent-owned option shared by ACS2 and ACS2ER.
`Bootstrap` is the default of both `new` and `with_population`. An ended episode
still stops on either flag, but its learning target is `reward + gamma * V(next)`
when only `truncated` is true, and `reward` whenever `terminated` is true. Success
on the limit step therefore has bootstrap zero, as does a transition raising both
flags. Exploitation uses the same rule and still updates RL.

A truncated ACS2 exploration step forms the next state's match set, applies ALP,
reforms that set after covering/deletion, estimates the bootstrap and applies RL,
then GA. This is the same learning pass as a continuing transition, at the same
ALP/GA timestamp, without selecting another action or resetting first. Termination
keeps the old learning pass. No transition or ALP/GA update is dropped at a cap.

Select historical validation explicitly with
`.with_truncation_mode(TruncationMode::Pyalcs)` before learning or restoring a
checkpoint. This mode initializes the final exploration match set empty and
bootstraps zero on either ending flag, reproducing the former loop exactly.
P8 selects the pyalcs mode in its explicit time-limit regression; its original
761 primitive differential vectors have no ending flags. P11, the episode
differential and both P9 agents also select it in their own code. Maze parity
checks the environment only, so it has no agent mode to select. Single-goal
learning equivalence runs both agents in each mode, including the corrected one.
`tu_probe` selects pyalcs for its online learner; its historical HER sample
construction retains the old `done` rule. All five v2 logs were reproduced exactly
except `wall`, with the existing commands.

ACS2ER stores `done = terminated` by default and
`done = terminated || truncated` in pyalcs mode; replay still gates its bootstrap
and GA match set on that stored bit. The online loop ends on either environment
flag independently of `done`. A mode switch with a nonempty replay buffer is
refused because samples cannot be reinterpreted after their flags were folded.
Direct callers of `replay_learning_step` own the meaning of the `done` they supply.

The mode is absent from `Configuration`, `ReplaySample`, `AgentState`, the codec,
`checkpoint_identity` and `AgentOptions::describe()`. Capture/restore save learning
state, not this option; restore retains the caller's selected mode. A caller
resuming a historical capped run must explicitly choose pyalcs first: the codec
cannot identify the origin of an old `done` bit or convert it. MPX checkpoints
need no special selection because `multiplexer.rs` always returns
`terminated: true, truncated: false`; the unchanged reach regressions pass.

This contract interprets caps as external training limits on a continuing task,
following [Pardo et al. (ICML 2018)](https://proceedings.mlr.press/v80/pardo18a.html)
and [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
It does not turn an intrinsic finite-horizon objective into a continuing task;
such an objective needs remaining time in the observation and terminal horizon
semantics. Goal-task success within a short evaluation cap is a separate metric.

`acs2-agents/tests/truncation.rs` checks all four paths (ACS2 exploration/exploitation and
ACS2ER replay/exploitation), a sample actually replayed twice after warmup,
termination, both flags, success reward, constructor defaults, restore and the
buffer guard. It also compares complete population fields and RNG after truncated
and continuing exploration with GA off/on, and exercises next-match-set deletion
and quality changes. `acs2-agents/tests/goal_maze.rs` adds an actual cap-one goal success in every
path and mode. Deliberate mutations swapping modes, zeroing default truncation,
bootstrapping termination and ignoring ACS2ER truncation each failed a focused
test before being reverted. The P6 unreachable-goal regression keeps its original
step-count assertions in the corrected default; it never asserted a zero target.

### Short paired ACS2ER diagnostic (2026-09-27)

Maze4-v0, all 27 coordinate goals, cap 5, 10,000 environment steps per mode and
seed, agent/environment ChaCha streams 0/1. A final partial episode is truncated
when the exact budget runs out. Configuration is `default_protocol()` with
`epsilon=1`; GA/PEE/action planning off, subsumption on, ALP variant Pyalcs,
`u_max=100000`, beta/gamma 0.05/0.95. Replay buffer 256, warmup 64, three samples
per step. There are no exploit/evaluation episodes or relabeling. Uniform actions
keep the learning inputs paired; both modes have equal episode/success counts,
trajectory hashes and final population sizes for every seed.

The cap metric is the mean maximum change-anticipating fitness `q*r` for the
actual action/state just before truncation, read after that episode's replay.
The classifier metric is the final unweighted mean `r` over change-anticipating
classifiers. Values below are corrected default / pyalcs:

| Seed | Episodes / successes / truncations | Population | Cap action fitness | Final change-classifier r |
|---|---:|---:|---:|---:|
| 42 | 2086 / 192 / 1894 | 1497 | 147.685801 / 99.176314 | 330.047309 / 207.838022 |
| 43 | 2093 / 198 / 1895 | 1581 | 133.210605 / 90.070172 | 312.618406 / 205.506803 |
| 44 | 2076 / 180 / 1896 | 1561 | 121.501090 / 85.766218 | 295.705382 / 202.088062 |

These are short-run value differences under a random policy, not convergence,
policy superiority or steps-to-goal evidence. Equal training success is expected
at epsilon one. The diagnostic source and raw outputs are delivered outside the
checkout; this diagnostic changes no runner or archived measurement.

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
transition** (`NEIGHBOUR_OFFSETS` in `maze/mod.rs`). The sibling helper
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
wins and `truncated` stays false** (gym sets `TimeLimit.truncated = not done`). In
pyalcs mode both episode endings bootstrap zero; the corrected default bootstraps
only truncation. Steps-to-goal metrics also depend on this boundary.

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

- **ALP generalization disabled by `u_max = 100000` ON THE MAZE PATH** (SPEC J.5). The
  expected-case over-specialization generalization branches never execute under the shipped
  maze config, so maze ALP only specializes. This deviation still holds for the MAZE
  measured path (P8/P9 use `u_max = 100000`). The generalization while-loops are NO LONGER
  omitted: **M2b implements them** (pyalcs `alp.py:78-94` faithfully, plus a Butz
  child-generalization variant), self-gated by `u_max` behind `alp_gen_variant`; they fire
  only on the MPX path where `u_max` is lowered, and stay dead on the maze path. See the
  M2b section.
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
(`acs2-core/src/support/config.rs`); everything else is new files (`acs2-envs::multiplexer`,
`acs2-core::support::knowledge`, `acs2-bench/src/bin/mpx.rs`). All P3–P8 + maze-parity tests stay
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

- `acs2-core::support::knowledge` (pure, zero env knowledge): `Transition<N> { p0, action, p1 }`
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
  plateau. The ceiling is 1.0 by construction; `acs2-core::support::knowledge` tests
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
inferred GA-deletion bug (SPEC §E.4 `cl.is_marked` bound-method): `acs2-core/src/learning/ga.rs:116` calls the
method correctly — Rust is the correct side.

Additive changes only: `--do-ga` on `mpx.rs`; a new `acs2-bench/src/bin/mpx_reach.rs`
(reach + per-component memory + 4-way verdict, with `--time-cap-secs`); `libc` dep on the
bench crate (for `getrusage` peak-RSS). `acs2-core/src/learning/ga.rs`/`acs2-core/src/learning/subsumption.rs`/`acs2-agents/src/acs2/mod.rs` and the maze
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
literature budget and is cited as CONTEXT only, so it is not the anchor. (An earlier
version of this note attributed a `u_max=1`, ε=0.1 protocol to that figure — that was
FALSE. The Unold PDFs never mention `u_max`; the `u_max=1` traces to
`reference/ALCS/experiments/configs/multiplexer_11.yaml`, where per ALCS's own
`parameter_guide.md:89` it means "max attributes in covering" — a DIFFERENT semantic that
is inert/print-only in the ALCS backend and never reaches `expected_case`. It is neither
the Unold paper's figure nor a pyalcs `expected_case` `u_max`.) Verdicts are four SEPARATE
outcomes, never merged: SUCCESS / TRIALS-LIMITED / MEMORY-LIMITED / TIME-LIMITED.

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
**M2b: ALP-generalization via lower `u_max`**, which generalizes inside `expected_case`.
(An earlier version of this note claimed ALP-gen is "immune to the race that defeats GA
here" — that was FALSE. ALP runs on the PREVIOUS action set (SPEC D), so a near-fully-
specialized rule reaches ALP as rarely as it reaches GA; ALP-gen is still action-set-
revisitation-bound. Its actual advantage over GA is that it fires on EVERY action-set
visit — no `theta_ga` gate — and generalizes DIRECTIONALLY, mark-driven via
`get_differences`/`u_max`, rather than by random mutation. Whether that pushes the
boundary past k=37 is the empirical question M2b answers, not an assumption either way.)
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
## M2b — Multiplexer with canonical ALP generalization (Task 2, phase 2b)

M2b turns on the canonical ALP over-specialization generalization branch (pyalcs
`alp.py:78-94`), DEAD in M1/M2a because `u_max = 100000`. It is a canonical-PORT phase:
the branch is transcribed faithfully, not modified. Two variants are shipped behind
`Configuration::alp_gen_variant` (`AlpGenVariant::{Pyalcs, Butz}`), self-gated by `u_max`
(no `do_alp_gen` bool — the branch is unreachable at `u_max = 100000`, exactly as pyalcs
expresses it), so the maze path stays on `u_max = 100000` and byte-identical (below):

- **A = Pyalcs (anchor).** Faithful transcription of `alp.py:78-94`: generalizes the
  PARENT classifier IN PLACE (the live population member), counting only
  `specified_unchanging_attributes` (condition-specified ∧ effect-wildcard); the child is
  snapshotted before the loops (`copy_from`, `alp.py:76`) and is NEVER generalized — it is
  only prevented from over-specializing by the shrinking `diff`. The coin-flip asymmetry
  (heads unconditionally decrements `no_spec_new`; tails decrements `no_spec` only if a
  parent generalization succeeded) is ported verbatim.
- **B = Butz (canonical-algorithm variant).** Per `articles/butz_algorithm.pdf` EXPECTED
  CASE: generalizes the CHILD (`child.C`), counting FULL condition specificity, any
  specified position eligible. This is a DELIBERATE DEVIATION from the pyalcs source lines,
  flagged as a separate mode — not a modification of A.

The A-vs-B divergence is a first-class finding, the same class as the four pyalcs bugs in
the deviations ledger: **does the widely-used pyalcs variant (parent-generalization)
compactify the reliable population the knowledge metric measures, or is child-compaction a
property only the Butz variant achieves?** GA is ON and EQUAL across A and B (see below),
so the A−B difference is attributable purely to the ALP variant.

### GA must stay ON: GA-off + ALP-gen is a MEASURED runaway (not run)

M2b keeps M2a's `do_ga = true`. This is not incidental — it was FORCED by measurement.
With GA OFF and ALP-gen ON, the Pyalcs parent-generalization mutates a reliable parent to
be MORE general, so the parent no longer subsumes its own specialized child; the child is
then ADDED instead of absorbed, and with GA off there is NO `theta_as` action-set deletion
pressure at all. The population grows super-linearly and the run becomes computationally
infeasible: MEASURED to not complete even a few hundred trials at k=11 (k=6 survives only
because 2^6 saturates). GA-on restores the deletion pressure and bounds the population
(k=11: 5000 trials in 0.34 s; k=20 300k trials in ~9.7 s). This is an artifact of the
GA-off CONFIG removing deletion pressure, not a pyalcs bug and not a port bug — but it is
why M2b = GA-on + ALP-gen (mirroring M2a), never GA-off.

### u_max is a DERIVED hyperparameter (no citable value exists)

No source in the repo gives a pyalcs-semantic `u_max`. Butz's algorithmic description
defines `umax` but gives NO experimental value; the Unold PDFs never mention `u_max`; the
`u_max=1` seen in `reference/ALCS/.../multiplexer_11.yaml` means "max attributes in
covering" (ALCS `parameter_guide.md:89`), a DIFFERENT semantic that is inert/print-only in
the ALCS backend and never reaches `expected_case`. So `u_max` for MPX is a DERIVED choice,
labelled not-literature, same honesty rule as the M2a trials cap. It bakes in
solution-structure knowledge (the compact MPX rule specifies `a` address + 1 data bit); the
`u_max` sweep is a SEPARATE study, not M2b. Maze keeps `u_max = 100000`.

Derivation principle: branch A fires when the variant's own count `>= u_max`, so set
`u_max` to the tightest value that (i) does NOT fire on the compact rule and (ii) DOES fire
at one-redundant. The two variants count differently, so they derive DIFFERENT `u_max`:

| quantity | Pyalcs (unchanging count) | Butz (full condition count) |
|---|---|---|
| compact no-change rule | `a+1` | `a+1` |
| compact change rule (validation bit in condition) | `a+1` (bit is CHANGING → excluded) | `a+2` (bit counted) |
| **derived `u_max`** | **`a+2`** (`control_bits+2`) | **`a+3`** (`control_bits+3`) |

The change rule specifies the validation/answer bit in its condition (`Classifier::
specialize` sets `condition[N-1] = p0[N-1]`); Butz's full count includes it, pyalcs's
unchanging count does not — hence B's `u_max` is one higher. Per-k values: Pyalcs
{k6→4, k11→5, k20→6, k37→7, k70→8, k135→9}; Butz {5,6,7,8,9,10}.

### Gate (n_exp=10, GA-on + ALP-gen): knowledge=1.0 holds, specificity converges CLEANER

Both variants hold knowledge=1.0 (exhaustive-exact, 10/10) on MPX-6/11/20 — the regression
check passes — AND drive reliable-condition specificity CLOSER to the ideal `a+1` than M2a's
GA-alone random mutation:

| k  | knowledge | Pyalcs spec (u_max=a+2) | Butz spec (u_max=a+3) | M2a GA-alone spec | ideal a+1 |
|----|-----------|-------------------------|-----------------------|-------------------|-----------|
| 6  | 1.0 (10/10 both) | 3.00±0.01 | 3.02±0.03 | 3.04±0.07 | 3 |
| 11 | 1.0 (10/10 both) | 4.00±0.00 | 4.00±0.01 | 4.15±0.18 | 4 |
| 20 | 1.0 (10/10 both) | 5.02±0.03 | 5.00±0.00 | 5.24±0.25 | 5 |

Reliable counts (Pyalcs / Butz / M2a): k6 27.5 / 28.2 / 28.1; k11 56.0 / 57.8 / 56.7;
k20 88.3 / 92.3 / 88.6. Both variants converge to `a+1` essentially exactly (Butz marginally
tighter at k20: 5.00 vs 5.02), vs GA-alone's ~0.04–0.24 excess. **Cleaner generalization at
equal k is CONFIRMED** — canonical ALP-gen approaches the ideal more tightly than GA's
random mutation, as theory predicts. Results: `reports/mpx_m2b_pyalcs.csv`,
`reports/mpx_m2b_butz.csv`.

### Reach k=37: SUCCESS, and MUCH cleaner than GA-alone (spec ~6.0 vs 7.91)

At the M2a boundary k=37 (a+1=6), both variants reach knowledge=1.0 (3/3 SUCCESS) with
specificity at the IDEAL, vs M2a GA-alone's mean 7.91/38 (~2 redundant):

| variant | verdict | knowledge | reliable | spec (/38) | vs M2a GA-alone |
|---|---|---|---|---|---|
| Pyalcs (u_max=7) | SUCCESS 3/3 | 1.0 | 144–156 | 6.01–6.05 (mean ~6.03) | 7.91 |
| Butz (u_max=8)   | SUCCESS 3/3 | 1.0 | 144–161 | 6.00–6.01 (mean ~6.00) | 7.91 |

k=37 reach delta vs GA-on baseline: reach UNCHANGED at k=37 (both SUCCESS, as GA-alone),
but specificity delta is large and favourable: **−1.9 spec (6.0 vs 7.91)** — ALP-gen removes
the ~2 redundant specializations GA leaves. Logs: `reports/mpx_m2b_reach37.log`.

### Reach k=70: the M2a freeze is BROKEN — compact rules, coverage time-bound (not race-lost)

M2a GA-alone LOST the generalize-vs-specialize race at k=70: reliables froze
fully-specialized (spec 46–68/71, ≫ a+1=8) with knowledge≈0 (0.0004–0.0071). Both M2b
ALP-gen variants shatter that freeze — reliables stay COMPACT near the ideal a+1=8, and
knowledge is orders of magnitude higher, though neither reaches 1.0 in the 600 s cap
(TIME-LIMITED, 3/3 ALL AGREE, both variants):

| variant | verdict (n=3) | knowledge | reliable | spec (/71) | peak RSS |
|---|---|---|---|---|---|
| M2a GA-alone | TIME-LIMITED | 0.0004–0.0071 | 57–109 | 46.35–67.97 (frozen) | 0.27 GB |
| Pyalcs (u_max=8) | TIME-LIMITED | 0.135–0.241 (mean ~0.17) | 75–110 | 8.63–9.79 (mean ~9.2) | 0.13 GB |
| Butz (u_max=9)   | TIME-LIMITED | 0.208–0.268 (mean ~0.24) | 203–223 | 8.52–8.75 (mean ~8.6) | 0.12 GB |

This is the M2b headline. The k=70 boundary CHANGES CHARACTER: under GA-alone it is a
race-lost freeze (reliables fully specialized, ~1/2^70 match fraction, action set never
revisited, knowledge≈0); under ALP-gen it is compact-but-coverage-incomplete (reliables at
spec ~a+1, knowledge climbing to 0.17–0.24 in the same wall-clock, still TIME-LIMITED). The
peak RSS (0.12–0.13 GB, far below the 5.6 GB cap) confirms the limit is genuinely TIME, not
memory — as it was for M2a, but now with compact rather than frozen rules. This is exactly
the mechanism predicted: ALP-gen fires on EVERY action-set visit (no theta_ga gate) and
generalizes DIRECTIONALLY (mark-driven), so a rule that specializes toward full length is
generalized back before it freezes — the freeze GA could not escape (theta_ga almost never
fires on a near-fully-specialized rule whose action set is never revisited).

**A-vs-B at k=70 — the raw gap is a `u_max` + seed CONFOUND; freeze-break is variant-
independent.** The 3-seed means (Pyalcs know ~0.17 spec ~9.2 rel ~90; Butz know ~0.24 spec
~8.6 rel ~215) ran Pyalcs at `u_max=a+2` and Butz at `u_max=a+3`, so they differ in BOTH the
variant AND the threshold. A confound-controlled isolation (k=70, seed 42, both variants ×
both thresholds; `reports/mpx_m2b_reach70_confound.txt`) resolves it:

| config (seed 42) | knowledge | reliable | spec (/71) |
|---|---|---|---|
| Pyalcs @ a+2 (correct) | 0.241 | 110 | 8.63 |
| Pyalcs @ a+3           | 0.246 | 257 | 9.80 |
| Butz @ a+2             | 0.024 | 8   | 7.38 (collapse) |
| Butz @ a+3 (correct)   | 0.256 | 203 | 8.58 |

Findings: (1) **knowledge is variant-independent** (~0.24–0.26 across all valid cells — the
"0.24 vs 0.17" gap was seed noise; the freeze-break does NOT depend on parent-vs-child).
(2) **Butz collapses at a+2** (knowledge 0.024, 8 reliable) — its full count fires branch A
on its own compact change rule and strips a needed bit, CONFIRMING why Butz derives a+3;
Pyalcs works at the tighter a+2, so **Pyalcs tolerates a tighter threshold**. (3) At equal
`u_max=a+3` Butz gives tighter conditions (8.58 vs 9.80) but FEWER reliables (203 vs 257) —
so "more reliable rules" was a threshold artifact that flips. Net: the parent-vs-child
divergence is a first-class *implementation* choice (pick one, derive its `u_max`), but the
separable variant effect is threshold tolerance + modestly tighter Butz conditions at equal
threshold — NOT a Butz sweep, and NOT the source of the freeze-break.
Logs: `reports/mpx_m2b_reach70.log`, `reports/mpx_m2b_reach70_confound.txt`.

### Reach k=135: beyond reach — no reliable rules form in budget (both variants)

At k=135 (a+1=9), neither variant forms ANY reliable classifier within the 360 s cap
(n=1, TIME-LIMITED): Pyalcs 324.5k trials, Butz 188k trials, both knowledge=0,
**reliable=0** (the reported spec=0.00/136 is the empty-set mean — no reliable rules, NOT
spec 0). Peak RSS 0.26 / 0.38 GB, far below the 5.6 GB cap: TIME-bound, not memory-bound.
Contrast M2a GA-alone, which formed 62 FULLY-SPECIALIZED reliable rules (spec 135.94/136,
knowledge 0) — different mechanism, same knowledge=0: GA froze 62 per-input rules; ALP-gen
churns general candidates none of which accumulate enough experience (q>theta_r=0.9) over
2^135 in 188–324k trials. k=135 is beyond reach for both under this budget.
Logs: `reports/mpx_m2b_reach135.log`.

### Reach summary and the boundary delta vs GA-on k=37

| k   | GA-alone (M2a)                  | Pyalcs ALP-gen              | Butz ALP-gen               |
|-----|---------------------------------|-----------------------------|----------------------------|
| 37  | SUCCESS, spec 7.91              | SUCCESS, spec ~6.03         | SUCCESS, spec ~6.00        |
| 70  | TIME-LIM, spec ~57, know ≈0     | TIME-LIM, spec ~9.2, know ~0.17 | TIME-LIM, spec ~8.6, know ~0.24 |
| 135 | TIME-LIM, 62 rel spec 135.9     | TIME-LIM, 0 reliable        | TIME-LIM, 0 reliable       |

**Largest k at knowledge=1.0: k=37 for both variants — the SAME k as GA-alone.** So the
reach-to-1.0 boundary is NOT pushed by k. The M2b gains are on the OTHER two axes the gate
asks about:
1. **Specificity delta at equal k (k=37): −1.9 (spec ~6.0 vs GA-alone 7.91).** ALP-gen
   converges to the ideal a+1=6; GA-alone leaves ~2 redundant. Cleaner generalization
   confirmed, both variants, tightest for Butz.
2. **k=70 boundary character delta:** GA-alone's race-lost freeze (spec ~57, knowledge ≈0)
   becomes ALP-gen's compact-but-coverage-incomplete (spec ~a+1, knowledge 0.17–0.24).
   Same TIME-LIMITED verdict, radically healthier state — a change in KIND, not just degree.

This is the pre-authorized FOUND BOUNDARY, reported as measured (budget NOT inflated). The
precise scope: *canonical ALP-gen (both variants) reaches knowledge=1.0 through k=37 with
IDEAL specificity (a+1), strictly cleaner than GA-alone; at k=70 it BREAKS the
generalize-vs-specialize freeze that defeats GA (compact rules, knowledge lifted from ≈0 to
~0.2) but remains TIME-bound short of 1.0; at k=135 no reliable rule forms in budget.* The
KNOWN SUBTLETY is borne out: ALP-gen runs on the previous action set, so it is NOT free of
action-set revisitation — k=70/135 remain time-bound — but its every-visit, mark-directional
generalization removes the FREEZE that made GA's boundary a hard wall, which is the
motivation for the thesis's prioritization/ER mechanism.

### Maze P8/P9 UNCHANGED — byte-identical by construction

The `u_max` flag keeps the maze path on `u_max = 100000`, so the ALP-gen branch is
unreachable and adds ZERO RNG draws on the maze path (the `else` while-loop runs zero
iterations; verified by the m2b_alp_generalization `disabled_u_max` fixture). Reproduced:
- **P8 differential: 761/761 cases, zero divergence** (2519 classifier assertions;
  `p8_differential` green under `default_protocol`, `u_max = 100000`).
- **P9 maze learning metrics: byte-identical** — `acs2-bench` default run (GA-off, n_exp=10,
  seed 42, 500 + 3×200) matches `reports/bench_rust.csv` exactly on all learning columns
  (`exploit_steps_mean/std`, `macro_pop_mean`, `numerosity_mean`, `reliable_mean`); only the
  wall-clock columns differ, as expected.
- All 59 workspace tests green, incl. P3 subsumption, P5 expected_case/unexpected_case, GA,
  maze parity, and the three new `m2b_alp_generalization` fixtures (parent-vs-child
  divergence + disabled-u_max inertness locked).

### Per-component memory — unchanged from M2a (no struct changes)

M2b adds no fields to `Classifier<N>`, so the `size_of` decomposition is IDENTICAL to M2a's
table above (Mark dominates, 75%→84% share k=20→135). Runtime peak RSS at each reached k
(ALP-gen reach runs, far below the 5.6 GB cap, confirming TIME- not memory-binding): k=37
0.03–0.06 GB; k=70 0.11–0.14 GB; k=135 0.26–0.38 GB. `Mark` remains the M3 packing target.


---

## Analysis tooling — logs reduce to committed CSVs, figures read only CSVs

Thesis provenance requires every figure to regenerate from the repository alone, but
the raw `mpx_reach` logs are large, arrive in several vintages, and mostly live on the
WCSS cluster. So the pipeline has a deliberate seam:

```
logs (.log/.out) --> tools/parse_mpx_logs.py --> reports/mpx_*.csv --> tools/plot_mpx.py --> reports/figures/
                          stdlib only                committed              matplotlib
```

**Nothing downstream of the CSV reads a log.** The CSVs are small and reviewable, so a
supervisor or examiner can check the numbers behind a figure without cluster access.

### Why `tools/` is a separate uv project from `baseline/`

`baseline/` pins an unmodified pyalcs (Python 3.10, `gym==0.23.0`, `numpy==1.23.5`) and
its lockfile is part of the P8 differential-validation evidence. Plotting dependencies
must not perturb an environment whose job is to stay fixed, so `tools/` carries its own
lock and a much looser Python floor.

### NAMED HAZARD — `seed = base_seed + repeat`

`mpx_reach` runs repeat `r` at `options.seed + r`. An `n_exp=3` log headed `seed=42`
therefore contains seeds **42, 43 and 44**, not three runs of seed 42. Collapsing them
into one series would have hidden the effect that turned out to dominate at k=70 (see
`docs/AGENT_HANDOFF.md` §4a): trials-to-success varies ~2.5x across seeds, and seed 42
is a fast outlier. The parser reconstructs the effective seed for exactly this reason.

Two related log-format facts the parser absorbs: trajectory lines carry no repeat index
(they are attributed to the repeat closed by the next verdict line, and an unclosed tail
becomes the next, unfinished repeat), and older logs predate `u_max` / `alp_gen_variant`
while a single file may hold several runs behind banner lines.

### Chart conventions

No dual-axis plots — the `anatomy` figure is stacked panels sharing one x-axis, because
knowledge, specificity and population have unrelated units and twin axes would invite
reading meaningless crossings. Colour follows the seed rather than its rank, mapped from
every seed in the CSV so that filtering never repaints survivors. A run cut off by its
time cap gets a hollow end marker and an explicit footnote: truncation is a budget
artifact and must never read as a failure to converge.

## ACS2ER — ACS2 with experience replay

Added on `feature/acs2er`. Reference implementation (source of truth):
`pyalcs/lcs/agents/acs2er/` (`ACS2ER.py`, `ReplayMemory.py`,
`ReplayMemorySample.py`, `Configuration.py`). The accompanying MSc thesis
(Ł. Śmierzchała, supervisor O. Unold, PWr 2022, ch. 4–5) motivates the
extension; where the text and the code disagree, the code wins — the
divergences found are listed at the end of this section.

### NAMED HAZARD — ACS2ER deletes current-step learning

**In ACS2ER the learning on the current step is REMOVED. All learning goes
exclusively through replay from the buffer.**

This contradicts the intuition that "ER adds replay *on top of* ordinary
learning", and that intuition produces a wrong implementation. `ACS2ER.py:44-99`
does only `form_match_set` → action selection → `env.step` →
`replay_memory.update(...)` → the replay block. There is **no `apply_alp` /
`apply_reinforcement_learning` / `apply_ga` on the current transition at all.**

Consequences that the implementation must exhibit, and that
`tests/p10_acs2er.rs` pins:

- While the buffer holds fewer than `min_samples` samples the agent **does not
  learn at all** — the population stays empty through the whole warmup window.
- Learning on a sample is **immediate**: match set from `sample.state`, action
  set from `sample.action`, next match set from `sample.next_state`. It is
  **not** deferred by one step the way `Agent::run_explore_trial` defers it.
  The deferred ACS2 loop was deliberately **not** reused or generalised —
  ACS2ER has a different structure and sharing that loop would be wrong.
- The timestamp handed to ALP and GA for every replayed sample is the
  **current** `time + steps`, never the time at which the sample was stored.
- `random.sample(range(0, len(buffer)), er_samples_number)` draws indices
  **without replacement**; `ReplayMemory::sample_indices` reproduces that with
  rejection redraws (the same selection-set strategy CPython uses at these
  `k ≪ n` sizes), not sampling with replacement.
- `_run_trial_exploit` is **identical** to ACS2's — no replay. It is therefore
  shared code, not a copy (see below).

### `LearningAgent` — the agent abstraction

`acs2-core::trial` holds what both agents share: `TrialMetrics`, the
`LearningAgent<N>` trait (`run_explore_trial`, `run_exploit_trial`,
`population()`, `config()`) and the free `run_exploit_trial` function that is
the single implementation of the exploitation loop. `Agent` (ACS2) and
`Acs2ErAgent` both implement the trait and both delegate exploitation to it.

`Agent`'s previous inherent methods were **moved** into its trait impl rather
than kept alongside it. Keeping both would have made every call site silently
resolve to the inherent method (inherent beats trait in method resolution),
so the trait could rot without any call site noticing. Call sites gained a
`use acs2_core::trial::LearningAgent;` import and nothing else; the P9 gate
confirms the move changed no behaviour.

The trait has generic methods (`E`, `S`, `B`), so it is **not object-safe** and
cannot be boxed. This is deliberate: the alloc-free `(&Population<N>,
&[ClassifierRef])` seams are worth more than `dyn` dispatch, and the benches
need static dispatch anyway. Agent choice therefore resolves in a `match` over
`AgentChoice` at construction, with the experiment body written once as a
function generic over `A: LearningAgent<N>`. This composes with the existing
`match size` monomorphisation in `run_reach_dispatch` without multiplying it:
size dispatch stays four arms, and the agent branch lives inside the
per-`N` function.

### ER configuration is a separate type, not new `Configuration` fields

`ReplayConfiguration { buffer_size, min_samples, samples_number }` lives in
`acs2-agents::acs2er::replay`; `Configuration` is **untouched**.

pyalcs models this by inheritance (`acs2er.Configuration` extends
`acs2.Configuration`). The Rust equivalent of that is composition, and it is
also the safer choice here: invariant 1 of this branch is that ACS2 behaviour
stays bit-identical, and a `Configuration` that cannot grow an ER field cannot
leak one into an ACS2 code path. It is Interface Segregation applied literally
— ACS2 does not depend on parameters it never reads. Defaults
(`10000 / 1000 / 3`) match `acs2er/Configuration.py`; the thesis calls the same
three knobs `N`, `N_warmup` and `m`.

### Replay memory

`acs2-agents::acs2er::replay::ReplayMemory<N>` is a `VecDeque<ReplaySample<N>>`
bounded by `max_size`, with
`update` evicting the front (oldest) before pushing, matching
`ReplayMemory.update`'s `pop(0)` exactly. Index `0` is the oldest sample on both
sides, so replayed indices mean the same thing. `ReplaySample<N>` is `Copy`
(`Perception<N>` already is), so replaying a sample does not clone the buffer. The sample
is defined in `acs2-core::learning::replay`, shared with trajectory storage and checkpoints.

All index draws go through the injected `RandomSource` (invariant: no
`thread_rng`, no global state). `sample_indices` clamps the request to the
buffer length instead of panicking; pyalcs would raise `ValueError` there, and
the clamp is unreachable in any configuration with
`min_samples >= samples_number`.

### Cost of the deliberate re-derive

`acs2-core::learning::replay::replay_learning_step` re-derives the next match set with
`form_match_set` after `apply_alp`, exactly as `Agent::run_explore_trial` does,
rather than using the set `apply_alp` maintains in place. The two are provably
equal (`apply_alp` appends every new `p1`-matcher and remaps deletions, and the
input set already held every other matcher), so this costs one extra population
scan per replayed sample and buys consistency with the documented deletion
contract above. Given that ER already performs three `form_match_set` calls per
sample, the deviation was not worth the divergence in style between two agents
that must be read side by side.

### What the differential covers

`baseline/dump_acs2er_differential.py` + `acs2-agents/tests/acs2er_differential.rs`
run the same 60-trial, 886-step explore experiment on both implementations and
compare the final population field for field. Both sources of randomness in the
pyalcs path (the replay index draw, and `PMark.get_differences`' position pick)
are recorded as one chronological event log and served back to Rust through its
`RandomSource`; action selection is scripted on both sides. The test asserts
Rust consumed exactly as many draws, with the same bounds, as pyalcs did —
so a structural divergence fails on the RNG stream even before the population
is compared.

The gate configuration sets `theta_i = 0`. ALP deletion is the only trigger for
the pyalcs mid-iteration skip documented above, and that bug would make an
end-to-end population comparison meaningless. The dumper counts deletions
regardless and the Rust test refuses to compare when the count is non-zero.
ALP deletion itself is covered by the P8 differential, which is unaffected.

`do_ga = false` in the gate, so the GA path through `replay_learning_step`
(including its `sample.done` empty-match-set gating) is **not** differentially
covered — it is transcribed from `ACS2ER.py:83-99` and reviewed, not proven.

### Reference code vs the thesis text

- **§5.3 calls the eviction policy "LIFO"** ("usunięcie najstarszego w buforze
  (ang. Last In First Out, LIFO)"). The prose is right and the label is wrong:
  dropping the oldest entry is FIFO, and `ReplayMemory.update`'s `pop(0)`
  drops the oldest. Implemented as FIFO, per the code.
- **Algorithm 5 shows no `done` handling in the replay block.** The code gates
  both the RL bootstrap (`0 if sample.done else ...get_maximum_fitness()`) and
  the GA match set (`ClassifiersList() if sample.done else er_next_match_set`)
  on the stored `done` flag. Implemented per the code; the algorithm listing is
  incomplete here.
- **§5.3's iteration-count formula, `n * m − N_warmup`,** does not follow from
  the loop. Learning runs on every step from the `N_warmup`-th onward, giving
  `m * (n − N_warmup + 1)` applications. Presentational only — no code depends
  on it.
- `_run_trial_explore` draws `env.action_space.sample()` into `action` before
  the loop and immediately overwrites it. Dead code in pyalcs that consumes a
  draw from the environment's RNG; not reproduced, and irrelevant here because
  the two implementations already draw from different streams (see the P9
  methodology note above).
- `TrialMetrics.reward` is the **total** episode reward in the Rust port and
  the **last** step's reward in pyalcs. This is a pre-existing convention of
  the Rust `Agent`, kept for `Acs2ErAgent` so both agents behind the trait
  report comparable metrics. No benchmark reads it.

## Checkpointing — save and restore of a run in flight

Added on `feature/checkpointing`. The blocker it removes is measured, not argued: one
k=264 seed is 700-4500 CPU-hours against a hard **504 h** per job in the longest queue,
so no single job can finish one. Everything else about it follows from one property.

### The property: a resumed run is identical, not similar

A run saved at trial *n*, exited, and resumed must produce the same trajectory as an
uninterrupted run of the same seed and configuration — trial for trial, not
approximately. `acs2-bench/tests/reach_regressions.rs::a_resumed_run_reproduces_an_uninterrupted_trajectory`
drives three separate processes (whole / first half / resumed half) through the real
`run_reach_repeat`, for **both ACS2 and ACS2ER**, and asserts two things: the
concatenated trajectory lines match the uninterrupted ones, and the checkpoint the
resumed run finally writes is byte-identical to the uninterrupted one's once wall-clock,
RSS and the checksum derived from those fields are removed. All learning fields remain
in the comparison; the resume path verifies integrity separately.

That second assertion is what makes the test hard to pass by accident, but it is not the
whole property: it compares evaluation snapshots, and both arms run the same
implementation. No-flag parity is established separately, by running the pre-change
binary from a git worktree and diffing its output line for line. It was verified
by sabotage: dropping the environment RNG, the agent RNG, the ALP clock,
`trials_since_eval`, the mark, `talp`, `tga`, `tav`, `exp`, population order, the replay
buffer, replay order, replay reward or the replay `done` flag each makes it fail, and so
does serialising floats as six-decimal text instead of bit patterns.

**`ee` round-trips through the codec and is checked there, but no test can cover its
effect on learning.** It is written (`set_mark` clears it) and never read: PEE is not
implemented, and `do_pee` only flips `leave_specialized` in `apply_alp`. It is serialised so
the checkpoint stays complete if PEE lands, not because a trajectory depends on it today.

### What is saved, and why each piece

| Saved | Because |
|---|---|
| Population, in order | `ClassifierRef` is a positional index; reordering it changes deletion and match sets |
| Agent RNG state | every ALP, GA and exploration draw comes from it |
| **Environment RNG state** | `Multiplexer::reset` draws the input bits — a fresh env RNG replays the same inputs from the top |
| `time` (the ALP clock) | `talp` / `tga` are relative to it, so `update_application_average` diverges without it |
| `trials_used` | the x-axis of every archived trajectory |
| `trials_since_eval` | otherwise the resumed run measures at different trial counts and the archive gets an inconsistent series |
| Peak macro population, peak RSS | run-level maxima, meaningless if they restart per job |
| Accumulated wall-clock | elapsed time along the retained checkpoint history; discarded work requires SLURM accounting |
| Replay buffer (ACS2ER) | it *is* the learning input; a fresh buffer relearns from different transitions |

### `ChaChaRandomSource` state — seed, stream, word position

`rand_chacha` 0.3 has no serde in this build and none is added. `ChaCha8Rng` exposes
`get_seed` / `get_stream` / `get_word_pos` and the matching setters, and those three
values determine the generator exactly, including a capture taken mid-block —
`get_word_pos` accounts for the position inside the buffered 4-block window. Both are
pinned in `acs2-core/src/support/rng.rs`.

`RandomSource` gains `capture_state` / `restore_state` with defaults that return
`None` / `false`, so every existing test double compiles unchanged and a source that
cannot be checkpointed fails **loudly** at save time rather than resuming from a
differently-positioned stream. That failure mode — a resumed run drawing from the wrong
place — is the one this design exists to make impossible.

**No container's iteration order became load-bearing.** `Population` is a `Vec` and is
written in order; `Mark` is a `[BTreeSet<Symbol>; N]` and is already sorted; the replay
buffer is a `VecDeque` written front to back. There is no `HashMap` or `HashSet` on the
learning path.

### The file format

Line-oriented text, hand-rolled in `acs2-core/src/checkpoint/codec.rs`. The format
lives beside the shared checkpoint contract. Two choices worth stating:

- **Floats are stored as `to_bits` hex, not decimal.** Rust's shortest-round-trip
  `Display` would also be exact, but exactness by construction is cheaper to review than
  exactness by argument, and a 500-hour run is a bad place to find out.
- **Symbols are two hex digits, wildcard `--`.** 2N characters per condition against a
  printable encoding's N, which at k=264 is ~1.5 KB per classifier and a 12.5 MB file
  for 8,107 classifiers — measured, not estimated. Reviewability wins over size here;
  the disk grant is 200 GB against 27 MB used.

Format 3 adds a required final `sha256 <64 lowercase hex digits>` line. The digest covers
every preceding byte, from the version header through `end\n`. Verification happens before
decoding state. Versions 1 and 2 are refused; there is no implicit unchecked legacy path.
Continue existing version-2 runs with their matching executable, or arrange an explicit
conversion after independently checking the source state.

SHA-256 is supplied by RustCrypto's `sha2` crate in `acs2-core`. One extra linear pass
over the already-rendered bytes is justified for an
artifact representing hundreds of hours: valid-looking changes to an RNG position or a
float previously passed every structural check and silently changed the resumed run.
The file remains inspectable text. This detects accidental corruption; it does not
authenticate the writer or prevent someone recomputing the digest after editing a file.

### What the independent review changed

The change was reviewed by a second model against the acceptance test's blind spots.
Five defects were confirmed against the code and fixed; two challenges to stated
decisions were accepted. The one that mattered:

**A resource cap could swallow an evaluation that was due.** The loop checks the RSS and
wall-clock caps *before* the evaluation block, so a job can stop on the very batch a
measurement was due, leaving the checkpoint owing it. Resuming straight into another
batch shifted that measurement and every later one. Reproduced at k=20 with
`--eval-interval 1000`: evaluations landed at 1500/2500/3500 against 1000/2000/3000, and
**SUCCESS was reported at 66,500 trials instead of 67,000** — the headline metric of the
thesis, moved by a checkpoint. Fixed by paying an owed evaluation before the next batch
(`train_before_measuring`); a fresh run never takes that branch.

The others, in descending order:

- **`--strict-resource-limits` can stop a job between measuring knowledge 1.0 and
  declaring SUCCESS.** Only a stored `SUCCESS` verdict short-circuited a resume, so such
  a run trained on and reported a later success trial. A run is solved at the trial its
  measurement names.
- **Nothing sequences the jobs of a chain.** One checkpoint is one learning state and
  there is no lock; the README had invited submitting a chain in one batch. It now
  prescribes `--dependency=afterany`. A staging file named from the destination plus the
  process id keeps two writers off one another, but that is defence in depth, not a
  substitute for sequencing.
- **`with_extension("partial")` could stage onto the destination itself** when the
  destination already ended in `.partial`, destroying the only recoverable copy, and
  `run.ckpt` and `run.backup` shared one staging file.
- **Booleans were decoded as `text == "1"`,** so anything else read as `false`. A file
  cut immediately after the space before a trailing flag still yields a field — an empty
  one — silently turning a terminal replay sample into a bootstrapped one. Booleans are
  now strict and the file carries an `end` terminator, which also rejects trailing junk.
- **Reopening a closed run inflated its archived cost** with the new process's start-up
  time and footprint.

Two challenges accepted:

- **A changed `--eval-interval` is now refused,** not warned about, unless
  `--checkpoint-allow-eval-change` says otherwise. It does not change what the agent
  learns, only which trials can be observed — which is the number this project reports.
- **The identity now carries the whole `Configuration`** through its `Debug` rendering
  rather than a hand-listed subset of flags. Nothing on the command line reaches `beta`
  or `theta_r`, so a later executable could change one and still satisfy a list; a field
  added later gates resumes without anyone remembering to add it.

One finding was **not** acted on as reported: no ownership lock was added. A stale lock
left by a killed job would block exactly the disaster recovery that
`a_periodic_checkpoint_outlives_a_killed_process` exists to prove works. Job dependencies
sequence the chain; the lock would only convert one failure mode into another.

### Two decisions that are not obvious from the code

**Wall-clock has two readings.** `--time-cap-secs` bounds *this process*, because that
is what the queue kills; the reported and archived `wall=` follows the retained checkpoint
history. Work lost after a save is absent from that number; SLURM accounting measures spend.

**The identity gate excludes the stopping limits.** Seed, encoding, epsilon, `u_max`,
GA, ALP variant and agent settings must match or the resume aborts — resuming under a
different one of those would splice two unrelated runs into one trajectory. Trials, wall
and RSS caps are deliberately absent: a chained run raises them per job, and they change
when a run stops, not what it learns.

`a_periodic_checkpoint_outlives_a_killed_process` covers the disaster path the other
tests miss: every one of them resumes from a *clean* stop, which records a verdict. A
node failure kills a job outright and the last periodic save is all that survives. The
test spawns a real run, waits for a periodic checkpoint, `SIGKILL`s it, and requires the
resumed run to rejoin the uninterrupted trajectory.

### What the second review round changed

The fixes went back to the same reviewer. It confirmed them and found five more, all real,
four of them in the archive stitching that had been written between the two rounds and
reviewed by nobody:

- **Diagnostic rows key `trials` as text and trajectory rows as an int**, so stitching a
  chain with diagnostics from two segments compared `"4000" > 4000` and raised. The
  stitching key is now cast; the stored value is not, because sorting the diagnostics CSV
  by a string is pre-existing behaviour and changing it would reorder the archive.
- **A segment that recorded nothing could not discard the work it superseded.** Boundaries
  were inferred from the rows, so a job that resumed and stopped before its next
  evaluation left the previous job's discarded measurements standing. `parse_log` now
  returns the segments it saw whether or not they produced rows.
- **Segments were ordered by resume point, which is not chronology.** Two jobs resume at
  the same trial when the first dies before saving again, and ties broke lexicographically
  — `_seg10` before `_seg9`, letting the older job overwrite the newer. A restart from
  zero after a deleted checkpoint sorted first and revived history it should have
  replaced. Ordering is now the SLURM job id.
- **A `run-segment:` marker relabelled rows still pending from an earlier block.** Pending
  rows are flushed before the marker takes effect.
- **A SUCCESS could vanish from the archive entirely.** The terminal checkpoint is written
  before the verdict is printed; a job killed between the two recorded no verdict, and
  every later job printed a line the parser did not read as one. A reopening job now
  restates the verdict marked `reopened=true`, and stitching keeps one per run. Confirmed
  by reverting: without it a reopened run prints no verdict line at all.

It also corrected three tests that asserted less than they claimed. The `SIGKILL` test read
the surviving checkpoint position *before* the kill, so a save landing in between made the
comparison wrong; it required no staging file to survive, which a kill during a staged
write legitimately leaves; and it filtered both sides of the comparison above the resume
point, which would have let a resume that restarted from zero pass, since that run
reproduces the same trajectory. The closed-run RSS assertion could pass without the fix
whenever the reopening process happened not to grow, and now forces the saved figure below
anything the process could be using. The staging test checked a filename the writer no
longer produces.

Two more it was right about: the `end` terminator makes the format incompatible, so it is
version 2 rather than version 1; and the process id in the staging name is unique on a
node, not across them, so the README no longer claims two processes never share one.

### What the third review round changed

Two more, both in the archive and both silent:

- **Concatenated segment logs dropped a run without a word.** Boundaries are built once
  per file from the context that survives to EOF, so a file holding two segments emitted
  both under the last one's name and the first's rows found no boundary and vanished.
  Reproduced. Rejected rather than handled: `sync_runs.sh` copies each job's log as its
  own file and nothing produces a concatenated one, so a second `run-segment:` marker or
  a second header block in a checkpointed log now raises.
- **A restart from zero could leave an obsolete SUCCESS standing.** Verdict selection
  only considered segments that recorded a verdict, so a job that restarted the run and
  was killed before printing invalidated the trajectory but not the verdict. The archive
  would have reported success from work the restart replaced. Verdicts now obey the same
  invalidation as measurements.

It also noted that the reopened-verdict test was written in Python against a handwritten
line, so it would have passed if the binary stopped printing one.
`a_reopened_run_restates_its_verdict_on_stdout` runs the real binary through
`CARGO_BIN_EXE_mpx_reach` and pins that it does.

`block` stays in `tools/mpx_selection.py`'s run identity and out of the parser's chained
run key. They are not in conflict: downstream, `block` separates runs *within* one file,
which is what a multi-block non-chained log needs; the parser's key spans files, where a
chained run's segments always carry block 1 — now enforced.

Three rounds, twelve defects, no false positives. Every round after the first found most
of its defects in code written between rounds, which is the argument for reviewing the
fixes and not only the change.

### What a fresh reviewer found

A fifth review, by a model with no history of the previous four, read the whole feature
rather than a diff. It found no missing learning state and no incorrectly restored RNG
position, and five operational or reporting defects. All five were confirmed against the
code; two were reproduced.

- **Trailing arguments overrode the flags the wrapper controls.** Extra arguments are
  appended after the wrapper's own and `mpx_reach` takes the last value of a repeated flag,
  so a trailing `--time-cap-secs` sailed past the allocation check and a trailing
  `--checkpoint-path` defeated the derived path that keeps two jobs off one learning state.
  Reproduced: the guard saw 600,000 and the binary received 1,800,000. Those flags are now
  refused in the tail.
- **An allocation the wrapper could not read silently disabled the time check.** No
  `scontrol` at all means this is not a compute node and skipping is right; `scontrol`
  present and unable to answer is a failure on a node where the check matters, and now
  refuses unless `CHECKPOINT_SKIP_TIME_CHECK=1`.
- **Job ids are not a clock.** SLURM wraps them from `MaxJobId` back to `FirstJobId`, so a
  chain crossing a rollover would order backwards and the older segment would discard the
  newer one's measurements and its SUCCESS. Ordering is now the wrapper's `started=`
  timestamp, with job id and restart count as a tiebreak — and only when *every* segment of
  the run carries one, because ordering a mixed set on it would sort the segments missing it
  ahead of all the others, which is the same silent destruction it exists to prevent.
- **A verdict a later job ran past was still rendered as the run's.** A stopped at 4,000 and
  recorded TIME-LIMITED; B resumed and reached 5,000 with no verdict yet. Stitching kept
  both, and `summarize_mpx.collect()` let the verdict replace the newer point, so the table
  read TIME-LIMITED at 4,000 — a mid-chain reading presented as where the run ended, which
  is the failure §7 of the handoff catalogues four times over. A verdict superseded by later
  progress is now dropped and the run renders as in flight.
- **`wall=` is not the chain's whole compute.** It is the elapsed time along the *retained*
  checkpoint history: work a job did after its last save and then lost to a kill is not in
  it, and `trials_per_s` is correspondingly optimistic. Not fixed, narrowed — see below.

It also corrected a claim in this document: `ee` *is* covered by the codec round-trip test.
What no test can cover is its effect on learning, because nothing reads it.

### Two claims narrowed rather than fixed

**`wall=` is elapsed time along the retained checkpoint history.** Making it the chain's
true expenditure would need per-attempt accounting including work that was discarded, and
the authoritative figure for spend already exists outside the archive: SLURM's own
accounting, which §5 of the handoff uses against the grant. Trials-to-success is the
machine-independent metric this project reports; wall-clock is colour.

**A checkpoint's durability, stated precisely.** The staging file is `sync_all`ed before
publication. The previous primary file is then renamed to `<name>.prev`, and the staging
file is renamed to the primary path. Each rename is atomic, but the pair is not: a process
killed between them leaves the primary absent. `run_reach_protocol` tests only that path's
existence, so it would start fresh on the next invocation. Recovery from `.prev` is manual;
restore it to the primary path before restarting. Directory synchronization is attempted
only for an explicit parent path, and both open and sync failures are ignored. These
existing limitations remain outside the checksum change; this implementation does not
guarantee automatic recovery or directory-entry durability through every node failure.

The version-3 checksum closes the separate content-integrity gap. Truncation, flags,
record counts and trailing records retain their structural checks. Their tests deliberately
recompute the checksum after damaging structure, so a green checksum test cannot hide a
permissive parser. The new corruption test first demonstrates that each changed RNG, clock
or float field parses with a recomputed digest, then requires rejection under its original
digest. The checksum-envelope test covers a missing or altered checksum, extra bytes,
missing final newline and a legacy header.

The checksum and adjusted structural/resume tests were checked with ten individual Rust
mutations: disabled verification, ignored trailing records/counts, omitted agent restore,
skipped owed evaluation, ignored measured success, inflated reopened RSS and suppressed
reopened stdout. Each compiled, selected exactly one test and failed an assertion. The
hard-kill test now parses and verifies the surviving file instead of only checking its
last word. Tests that deliberately alter valid checkpoint fixtures use read/modify/write,
so they still exercise resume behavior rather than failing on stale checksums.

### Budgeting memory for a save

Measured at k=264: a `Classifier<265>` is **7,504 B** resident, of which the mark is 6,360 B
(84.8%), and **1,542 B** as text. `Checkpointed::capture` clones the population and `render`
builds the whole file as a `String`, so a save roughly **doubles the population's footprint**
while it runs:

| population | live | at the moment of a save |
|---|---|---|
| 8,107 (measured at 500 trials) | 0.06 GB | 0.13 GB |
| 100,000 | 0.75 GB | 1.65 GB |
| 300,000 | 2.25 GB | 4.96 GB |

At 500 trials the difference is below the reported resolution — 0.35 GB peak RSS with and
without checkpointing. Budget `--mem` for at least **2.2x the population's resident size**.
If a k=264 population reaches the hundreds of thousands, that is the moment to stream the
render into the staging file and borrow the population instead of cloning it; until then the
simpler code is worth more than the headroom.

**Measured on the first production segment (job 5872532).** The checkpoint at 100,000
trials is **24,905,571 B for 13,394 classifiers — 1,859 B each**, above the 1,542 B measured
at 500 trials because the mark fills as a run proceeds and it is 84.8% of a classifier.
Peak RSS over the same stretch is **397.50 MB** (`sstat -j <id>.batch`). Anchoring the
model to that reading:

```
RSS(pop) ~ 172 MB + pop x 16,867 B        # 7,504 live + 7,504 clone + 1,859 text
```

which puts a population of 300,000 at 5.23 GB and 750,000 at 12.8 GB.

**A save also costs time, and it scales with the population.** Separable from the first
production segment's own log, because only every twentieth interval carries one:

| population | 5,000 trials, no save | with a save | the save |
|---|---|---|---|
| 40-70k | 199 s | 482 s | 284 s |
| 70-95k | 305 s | 711 s | 406 s |
| 95-120k | 411 s | 1,081 s | **670 s** |

That is 5-6 ms per classifier, and it is the clone plus the per-classifier text rendering
rather than the I/O — 209 MB written in 670 s would be 0.3 MB/s, which no disk is. At
`CHECKPOINT_EVERY=100000` and 11.7 trials/s it costs **7.8%** of the run. The lever that
does not touch code is `CHECKPOINT_EVERY`: doubling it halves the overhead and doubles the
work a node fault can take, which at 100,000 trials is about 2.4 h against 500 h segments.

**The binary's own cap is not `--mem`.** `DEFAULT_RSS_CAP_BYTES` is 5.6 GB and the loop
breaks on it with a `MEMORY-LIMITED` verdict, so before `RSS_CAP_GB` existed every cluster
run carried 5.6 GB however much memory it had been allocated — a k=264 population near
300,000 would have stopped a job holding 16 GB. Set the internal cap below `--mem`: SLURM
kills a job that exceeds the allocation and loses everything since the last periodic save,
while the internal cap stops cleanly and writes a checkpoint. At `--mem=16G` and
`RSS_CAP_GB=13` the headroom reaches roughly 750,000 classifiers, above the largest
population this project has recorded (459,284 at k=135, `u_max=16`).

### Budgeting the measurement itself

An evaluation is two sweeps over 50,000 sampled inputs, and they do not cost the same.

`evaluate_knowledge` collects the **reliable** classifiers first and asks each transition
whether any of them anticipates it, so its cost scales with the reliable set, not the
population. At k=264 that set is empty for the whole measured range — every archived
trajectory point reads `reliable=0` — and `.any()` over an empty slice returns immediately.
What remains is building the 100,000 transitions, which is the same work at every call.

`answer_accuracy` forms a match set over the **whole** population for each sampled input.
That is the only `O(population x 50,000)` term in the loop, and at k=264 it is effectively
the entire cost of a measurement.

Both used to hang off `--eval-interval`, so a knowledge grid fine enough to watch a run
climb bought a sweep that does not need to be that dense. `--accuracy-every N` separates
them: knowledge every `--eval-interval` trials, accuracy every `N` of those.

**Measured at k=264, seed 42, `outcome`, `u_max=12`,** on three runs submitted as a
dependency chain so no two shared a node. Same seed means the same population at the same
trial, so the wall-clock difference is the measurement and nothing else. Three
configurations separate the two costs: `t` and `c` pay the same number of sweeps and differ
only in knowledge evaluations, `f` and `t` differ only in sweeps.

| trials | population | knowledge eval | accuracy sweep | ratio |
|---|---|---|---|---|
| 40,000 | 9,520 | 0.43 s | 31.6 s | 74x |
| 80,000 | 12,535 | 0.36 s | 36.6 s | 103x |
| 120,000 | 14,932 | 0.38 s | 42.1 s | 110x |
| 200,000 | 21,800 | 0.23 s | 54.2 s | 237x |
| 240,000 | 22,568 | 0.26 s | 58.9 s | 225x |

**A knowledge evaluation costs about 0.3 s and does not grow with the population**, because
`reliable` is empty and what remains is building the 100,000 transitions. The sweep grows
linearly and is two orders of magnitude dearer.

**The slope from those populations understates it.** Fitting 9.5k-22.6k gives 2.09 ms per
classifier, and the first production segment measures **4.90 ms** — the intervals carrying
a sweep are separable in its own log, because only every twentieth does:

| population | 5,000 trials, no sweep | with a sweep | sweep |
|---|---|---|---|
| 20-40k | 107.6 s | 253.0 s | 145.4 s |
| 40-60k | 184.6 s | 440.3 s | 255.8 s |
| 60-80k | 253.9 s | 594.1 s | 340.2 s |

At 77,607 classifiers a sweep is 380 s against 265 s for 5,000 trials of learning: at
`--accuracy-every 1` the measurement would cost **144%** of the learning, and at 20 it costs
7.2%. Both terms scale with the population, so that ratio is roughly stable as a run grows.
Over the segment's first 1,900,000 trials the thinning is worth **2.22x** — the same
learning would have taken 45.9 h instead of 20.6 h, and after 20.6 h the run would have been
at 1,140,000 trials instead of 1,900,000.

`N=1` is the default and reproduces the previous output exactly — checked against the pre-change binary
at k=20, where every learning column, knowledge value and accuracy value matches line for
line and SUCCESS lands at the same 67,000 trials.

**The cadence is derived, not stored.** The next accuracy trial is recomputed from
`trials_used` as `(trials_used / step + 1) * step`, so the checkpoint gains no field and the
format is unchanged, and a resumed segment lands on the grid an uninterrupted run would have
produced. Carrying a countdown instead would have reset it at the boundary and shifted every
later sweep — the same defect the first review round found on the knowledge grid, where a
resume moved SUCCESS from 67,000 trials to 66,500.

`an_accuracy_cadence_survives_a_resume` pins this. It runs on the k=20 fixture rather than
the k=6 one the other checkpoint tests use, because k=6 reaches knowledge 1.0 on its second
measurement and ends the run before a break can sit between two sweeps. It was verified by
sabotage: resetting the threshold on resume, advancing it by the evaluation interval instead
of the step, and dropping the `+1` are each caught, the first as an extra sweep the
uninterrupted run does not have.

A caveat the flag does not remove: nothing gates `--log-accuracy` or `--accuracy-every`
across a resume the way `--eval-interval` is gated. Changing either mid-chain thins or
thickens the accuracy grid of a single run without complaint.

### What the fresh reviewer's second pass changed

Its verification round confirmed the flag-override and stale-verdict fixes and found four
more, two of them in the fix for its own finding:

- **`started` was compared as text.** `date -Is` writes local time with an offset, so across
  a daylight-saving change the later job carries the smaller string —
  `02:30:00+02:00` sorts after `02:00:00+01:00` although the second is half an hour later.
  Poland changes clocks on 2026-10-25 and the grant runs into 2027, so a k=264 chain reaches
  it. Ordering now compares parsed instants, and a `started` that cannot be parsed no longer
  counts as a clock.
- **One missing `started` reinstated the wrapped-job-id bug.** The all-or-nothing fallback
  meant a single segment without a timestamp threw the whole run back onto ids. Guessing an
  order has now produced silent loss twice, so an unorderable chained run **raises** instead:
  ordering decides which segment's measurements survive and is not inferred.
- **A missing `scontrol` still disabled the time check inside an allocation.** "No scontrol
  means this is not a compute node" is not true — a module environment can strip it. Every
  way of failing to establish the limit now takes the same explicit override.
- **The durability claim, again** — see above.

It also noted a comment in `mpx_reach.rs` still saying "the whole chain's compute" after the
docs had been narrowed, and that a test which used to pin the numeric fallback had stopped
doing so once the fixture gained timestamps. Both fixed.

### Clock order needs supporting evidence (2026-09-11)

`boundaries_by_run` still uses parsed instants to propose chronology, but no longer accepts
that proposal without checking the chain. Its first segment must start at zero; callers
must provide the initial log, not a suffix of the chain. Each later resume point must be
at or below the preceding segment's furthest recorded progress. Progress includes trajectory,
accuracy, coverage, diagnostic and verdict trials, and the segment's own resume point.
The latter lets an attempt that printed nothing new hand the same checkpoint onwards.
Restarts at zero remain valid and invalidate the history they replace.

The largest logged trial is a **lower bound**, not an upper bound. Periodic saves can land
between evaluations, and a killed process prints no terminal verdict. A link beyond the
recorded bound is therefore refused as *unsupported*, with a message naming both logs and
trial counts; it is not proof that the clocks are wrong. This deliberately rejects some
valid sparse chains rather than deleting measurements under an unverified order. Recover
missing logs or independently verify their chronology and progress before repairing an
archive input; do not fabricate an evaluation row to satisfy the check.

Timezone-free timestamps are refused rather than interpreted in the parser machine's
timezone. Equal instants can be resolved only by distinct numeric restart counts of the
same SLURM job. Job ids and filenames cannot break a tie across jobs. A job's attempt
counts must increase in clock order, including when its instants differ. Validation also
runs when every segment is empty, before any CSV is written.

These checks detect contradictions and unresolved timestamp ties, not arbitrary clock
skew: two attempts resuming from the same point can interlock in either order, and distinct
but wrong timestamps are not distinguishable from correct ones using the current logs.
Synchronized clocks remain an operational requirement. Removing that assumption requires
explicit predecessor identities in future checkpoint/log records, not sorting by trial
counts. The retry regression is renamed `test_clock_order_wins_when_retries_share_a_resume_point`;
it pins replacement of a larger discarded trial count and catches filename ordering.

The wrapper now checks `scontrol`'s exit status before parsing its output. Previously a
failed query could print `TimeLimit=UNLIMITED` and still bypass the guard. Its regression
checks both empty and plausible output with exit status 1, preservation of an existing log,
and the explicit override. Nine affected Python tests were individually run with their
protection removed; each selected exactly one test and failed an assertion, rather than
failing to import the test module.

### Verifying the reviewer's own implementation

The reviewer that found the last nine defects was handed the remaining open items to
implement rather than report — it has repository access, and the earlier insistence on
read-only was my constraint, not its limitation. Its two commits were verified here rather
than accepted: all four gates re-run independently (101 Rust tests, 45 Python tests, P9
byte-identical, the archive unchanged), every check it added sabotaged one at a time under a
harness that asserts the test *ran*, and the twelve-segment chain re-measured end to end.
Everything it claimed holds.

Two things the verification added.

**The new dependency had to be checked against the target that matters.** The checksum
originally brought `sha2` and its tree into `acs2-bench`; the dependency now follows the
unchanged codec into `acs2-core`. The cluster binary is built `--target x86_64-unknown-linux-musl`, and nothing had
compiled that tree for musl. `cargo check --target x86_64-unknown-linux-musl` passes, which
covers compilation but not linking; the cluster remains where the real build happens.

**A loud refusal must not silence the sync.** The parser now refuses a chain whose ordering
it cannot support, which is right on its own — but `tools/sync_runs.sh` runs the parser
under `set -e` *before* it commits, so a refusal would abort with a freshly pulled log
sitting uncommitted, in exactly the one-copy state that script exists to end. That is this
repository's other named failure, traded against the first. The rebuild is now fenced: if
the parser refuses, the logs are committed with a message saying the archive was not rebuilt,
and the refusal is reported afterwards. The irreplaceable half is never held hostage to the
derived half.

### The sabotage harness was broken, and what that cost

Every fix in this feature was checked by reverting it and requiring a test to fail. On the
Rust side that worked: re-run with an explicit check for "compiled but failed" rather than
just "did not print `1 passed`", all eight core sabotages genuinely compile and genuinely
fail.

**The Python harness did not.** It invoked
`python -m unittest tools.test_mpx_archive.ArchiveTests.<name>`, which cannot import
`parse_mpx_logs` — `tools/` is only on the path under `unittest discover`. Every such run
errored before reaching a test, and the harness read any non-`OK` result as a catch. Every
Python-side sabotage reported during this work proved nothing.

Re-run correctly with `unittest discover -s tools -p 'test_*.py' -k <name>`, and requiring
that a test actually ran, all of them hold except one, which was a real gap:

**`test_work_a_killed_job_did_after_its_last_checkpoint_is_superseded` did not test
superseding.** Its resumed segment re-measured every trial the killed one had, so
overwriting by trial gave the same answer as discarding and re-inserting. The drop only
matters when the resumed job does *not* reach a trial again — it stops earlier — and then a
measurement of discarded work survives. The fixture now has the resumed job stop at 7,000
after the killed one recorded 8,000, and keeps the overwrite case alongside it.

One sabotage that "still passed" is not a gap: choosing the verdict by maximum trial count
*after* invalidation is provably the same as choosing the last chronologically, because a
job cannot end with fewer trials than it resumed at. Selection by maximum taken independent
of the boundaries is caught.

The lesson is narrow and worth keeping: a sabotage harness must assert that the test **ran**,
not merely that the command failed.

### What the fourth review round changed

Nothing in the diff it was given — the round-3 fixes hold. It found one defect anyway, in
the wrapper, by following a question the brief asked it to consider.

**SLURM requeues a job under the same id**, which is exactly the case checkpointing exists
for: node failure, preemption. The log name carried only the job id and the wrapper
truncates it, so the second attempt erased the first's. Measured: three attempts under one
job id left a single log with one measurement at 6,000 trials, the first 4,000 trials of
trajectory gone — while the run itself continued correctly from its checkpoint. Neither
rejection fires, because the surviving file is a perfectly well-formed single segment. The
name now carries `SLURM_RESTART_COUNT` and the provenance line carries `attempt=`, which
also orders the attempts of one job within `segment_order`.

This is the archive failure this repository keeps having: the run is fine, the record is
not, and nothing says so.

It also pointed out that the second-marker guard was never pinned on its own — both
concatenation fixtures carry two headers, so they would still raise through the header
check with the marker guard removed. My sabotage of that guard had "passed" for that
reason. Pinned now with two markers and one header.

### How the archive reads a chained run

A checkpointed run spans several jobs and `slurm/mpx_reach.sh` gives each its own log
(`..._seg<jobid>.out`) — a single filename would leave only the last segment, and the
segments are where the trajectory lives. Downstream tools key a run on `source`, so
without help a chain reads as **independent runs sharing a seed**, which is what
`plot_mpx.py` was hardened to refuse. Measured on a three-segment k=20 chain before the
fix: two verdict rows for one run.

The wrapper emits the provenance and `tools/parse_mpx_logs.py` acts on it:

```
run-segment: base=<stable run name> checkpoint=<path> checkpoint_every=<n> resumed=yes|no
```

- **`source` becomes the run, `segment` the job.** A new `segment` column carries the file
  a row came from; it is empty for every log written before checkpointing, and the rebuilt
  archive is unchanged apart from that empty column — 79 verdict, 68,730 trajectory and
  63,471 diagnostic rows, none differing on any existing field.
- **A later segment supersedes an earlier one above the trial it resumed at.** A resumed
  job re-runs whatever the previous one did after its last checkpoint, so the same trial
  can appear twice with different values and the earlier one is of work that was
  discarded. Verified against a real `SIGKILL`: a job logged measurements at 6,000 and
  8,000 trials but checkpointed at 4,000, and the stitched series reproduces an
  uninterrupted run's values exactly.
- **One verdict per run, not one per job.** Intermediate jobs stop on their own wall clock
  and record TIME-LIMITED; that says a job stopped, not that the run did. The run's verdict
  is the closing segment's — last in validated clock order, not by trial count, because a job
  killed after its final checkpoint can report more trials than the job that legitimately
  superseded it.

Superseding is applied once per segment, not once per record. Doing it per record makes
each of a segment's own rows discard the one before it, which is a mistake that reads as
correct and was caught only by running a real chain through the parser.

`--eval-interval` must stay constant across a chain — `mpx_reach` refuses a resume that
changes it without `--checkpoint-allow-eval-change` — or the stitched series carries two
sampling rates and trials-to-success stops meaning anything.

## Goal port

Added on `feature/trajectory-utility`. HER, trajectory utility and every agent compared with
them need what `Environment<N>` does not have: a goal, and the reward and termination of a
stored transition recomputed for a goal the transition was not collected under.
`acs2_core::ports::goal` supplies that as a second port. It is additive — `Environment<N>`,
`StepOutcome`, `Info`, `Configuration`, `Classifier` and the checkpoint codec are unchanged,
and the maze and multiplexer paths do not see it.

### Core module groups and agent ownership

Core source is grouped by role; paths in this table are relative to `acs2-core/src/`:

| Group | Modules |
|---|---|
| `representation/` | `symbol.rs`, `perception.rs`, `condition.rs`, `effect.rs`, `mark.rs`, `classifier.rs`, `population.rs`. |
| `learning/` | `alp.rs`, `ga.rs`, `rl.rs`, `subsumption.rs`, `action_selection.rs`, `replay.rs` (`ReplaySample` and the shared full replay learning step). |
| `ports/` | `environment.rs`, `goal.rs`. |
| `support/` | `config.rs`, `rng.rs`, `knowledge.rs`, `measurement.rs`. |
| `trial.rs` | `LearningAgent`, `TrialMetrics`, `TruncationMode` and the shared exploitation trial. |
| `checkpoint.rs`, `checkpoint/codec.rs` | Agent checkpoint contract/state and the unchanged format-3 codec. |

Each group exposes its defining modules through `mod.rs`; `lib.rs` declares the groups,
trial and checkpoint modules. There are no former-path re-exports or facade modules.
All callers, including `mpx_reach.rs`, P9, measurement, tests and examples, import the
defining paths. Source locations and declarations change; learning, parameters,
defaults, checkpoint identities and serialized bytes do not.

Every learning agent belongs to `acs2-agents`, one module directory per family:
`acs2/`, `acs2er/` (FIFO replay memory and `ReplayConfiguration` in `replay.rs`, diagnostic
one-update variant in `one_update.rs`, episodic variant in `episodic.rs`) and `acs2her/`.
Their shared episodic collection and
copy-update dispatch live in `episodic_replay.rs`. The full replay step stays in core
because ACS2ER and episodic/HER learners use it; `ReplaySample` stays there because
trajectory relabeling and checkpoint storage also use it. Agent parameters stay in their
family. Agent-dependent test suites live in `acs2-agents`, with environment construction
as a development dependency; core and environments have no dependency on agents.

The running MPX chain builds from `feature/mpx264` in its own clone and does not merge
these changes during the chain. Its code has not changed since that branch split from
`main`, so this branch can use the clean layout without preserving its former imports.

### Agent run names (2026-10-14)

Every agent type is named `Agent` in its defining module. Callers using several agents
qualify those modules or alias imports, for example `acs2er::Agent as Acs2ErAgent`.
ACS2ER owns both replay schedules and the one-update variant; episodic replay without
goal substitution is a variant of ACS2ER. ACS2HER owns its named hindsight presets
and the test-only `pyalcs_differential.rs` oracle.

| Archived run name | Current run name | Module in `acs2_agents` | Type |
|---|---|---|---|
| `acs2` | `acs2` | `acs2` | `Agent` |
| `acs2er` | `acs2er` | `acs2er` | `Agent` |
| `acs2er_u1` | `acs2er_one_update` | `acs2er::one_update` | `Agent` |
| `er_episodic` | `acs2er_episodic` | `acs2er::episodic` | `Agent` |
| `her_full` | `acs2her` | `acs2her` | `Agent`, preset `Acs2Her` |
| `her_values` | `acs2her_values` | `acs2her` | `Agent`, preset `Acs2HerValues` |
| `her_values_shortened` | `acs2her_values_shortened` | `acs2her` | `Agent`, preset `Acs2HerValuesShortened` |

The executable accepts and writes current names only. An archived CLI name is refused
with its replacement. The sole machine-readable mapping is
[`experiments/grid/agent_names.json`](../experiments/grid/agent_names.json).
`acs2-measure::agent_names` embeds those bytes for the pinned fixture readers and CLI
diagnostics. The adjacent Python module supplies the shared collector, replay-rule
validator, duplicate/pilot learning-field reader and cross-version identity comparator.
Readers normalize a copy of the `agent` field and resolve per-agent settings under
either spelling; raw rows, manifests and their checksums remain unchanged.

New comparisons with archived rows use `grid_identity.compare`, which compares binary64
bits after excluding the existing provenance/timing fields and the analysis-only `arm`.
The registered phase-5 code, plan, fixtures and rows retain archived names and use their
pinned binaries. Its analysis label `her_values_p2of3` remains unchanged and is absent
from the run-name mapping. The agent and arm names in the implementation/results
sections below use current names, except that closed analysis label.

### Commit-history labels

These labels occur in older thesis-line commit messages. History is not rewritten;
the sections below are named by their implementation or measurement:

| Historical commit label | Section |
|---|---|
| phase 1 | [Goal port](#goal-port). |
| phase 2 | [Goal environments and geometry provenance](#goal-environments-and-geometry-provenance), including the environment interface and random-policy sparsity. |
| phase 2b | [Learning at an episode boundary](#learning-at-an-episode-boundary). |
| phase 3a | [Goal measurement layer](#goal-measurement-layer). |
| phase 3b | [Pilot grid: ACS2 and ACS2ER on ten goal configurations](#pilot-grid-acs2-and-acs2er-on-ten-goal-configurations). |
| phase 4 | [Trajectory storage and relabeling](#trajectory-storage-and-relabeling). |
| phase 5 | [Hindsight copy updates and held-pair replay results](#hindsight-copy-updates-and-held-pair-replay-results). |

### Checkpoint and multiplexer compatibility

The checkpoint identity renders the whole `Configuration` through `Debug`
(`checkpoint_identity` in `mpx_reach.rs`), and a resume asserts that identity. A new
`Configuration` field therefore stops every checkpointed chain on its next segment once the
cluster binary is rebuilt, and the reader accepts format 3 only. Parameters of goal agents
live in their own types, as `ReplayConfiguration` does for ACS2ER, and the goal line runs
through its own binaries rather than `mpx_reach`, whose source `acs2-bench/tests/reach_regressions.rs`
includes verbatim.

`Configuration`'s derived `Debug` contains its bare type name, without a module path.
Grouping the source therefore leaves that identity string unchanged; neither
`type_name` nor `module_path!` is introduced into identity or checkpoint handling.

### Reward and termination are derived, not reported

A goal environment reports what it cannot avoid knowing — the observation, the goal the step
**achieved**, whether the new state ends the episode under every goal (`terminal_state`),
and whether the time limit was hit (`time_limit_reached`). `GoalStep::outcome(objective,
desired)` derives the rest:

- `reward = objective.reward(achieved, desired)`
- `terminated = terminal_state || objective.is_reached(achieved, desired)`
- `truncated = !terminated && time_limit_reached`

The last rule is the gym `TimeLimit` exclusivity that `Maze::step` already follows: a goal
reached on the limit terminates. Because the environment never states a reward, relabeling a
transition with another goal cannot disagree with what the environment would have said.
pyalcs is the counterexample: its three ACS2HER variants relabel with 1/0, 0/−1 and a
user function, none on the environment's 1000/0 scale, and mark every relabeled sample
`done=False` even when it reaches its goal. The price is a restriction: a goal task's reward
must be a pure function of (achieved, desired).

**NAMED HAZARD — goal rewards must be non-negative.** `Population::get_maximum_fitness`
folds from `0.0`, so a bootstrap over classifiers with negative fitness is clipped to zero.
The 0/−1 scheme of the HER literature (and TUCA-HER, Eq. 1) would bias every ACS2 value
estimate; goal tasks here pay `ExactMatch { reward_on_reach: 1000.0 }` and zero otherwise.

### The achieved goal comes from the environment

A goal is not assumed to be a function of the observation. Perception goals alias: in 15 of
the 27 mazes in this repository the goal cell's 8-neighbour perception also occurs at another
walkable cell (MazeF3 and the pyalcs Woods100 among them), so `achieved == desired` on
perceptions is not the environment's success there. Maze4/5/7 and Woods1 are free of it.
Each environment chooses a goal representation it can report exactly.

### `M = S + G` without generic const expressions

`[Symbol; S + G]` in a generic signature is rejected on stable Rust (rustc 1.96: "generic
parameters may not be used in const operations"). `GoalLayout<S, G, M>` takes all three as
independent parameters and asserts `S + G == M` in an associated constant, evaluated after
monomorphization — the pattern `Multiplexer::CONTROL_BITS` already uses. `GoalConditioned::new`
references it, so a wrong triple is a compile error, not a runtime one. Agents stay generic
over one length `M` and never learn that part of their perception is a goal;
`GoalLayout::goal_positions()` is there for code that must know, such as diagnostics.

### ACS2 and ACS2ER run on goal tasks unchanged

`GoalConditioned<E, S, G, M>` implements `Environment<M>` for any `GoalEnvironment<S, G>` by
appending the desired goal to every observation, so an agent that has never heard of goals
learns a goal-conditioned task through it. `acs2-agents/tests/goal_port.rs` runs the unmodified `Agent` on
a goal corridor and pins determinism from the seed. `reset_with_goal` starts an episode on a
chosen goal, which evaluation needs.

The adapter owns the desired goal of the running episode and nothing else may change it: there
is no mutable access to the wrapped environment, because resetting it behind the adapter's back
would leave every later observation labelled, and every reward computed, for the old goal. The
goal is forgotten when an episode ends, so a step without a fresh reset is refused rather than
scored against a goal that no longer applies.

### Goal environments and geometry provenance

`acs2-envs/src/roles.rs` is the role registry. `VALIDATION_MAZES` pins P9's exact five
geometries and order (Maze4, Maze5, Maze7, Woods1, Woods100); it is independent of the
provenance catalogue, so adding a geometry cannot silently extend P9.
`PERFORMANCE_MAZES` is the existing 22-geometry ALCS comparison.
`RESEARCH_MAZES` names Maze4/5/6/7, Woods14, MazeMA and the pyalcs MazeF3/MazeB research geometries.
`RESEARCH_TASKS`, `VALIDATION_ORACLES` and `PERFORMANCE_BENCHMARKS` state the task families.
All four goal task families are implemented under `acs2-envs/src/goal/`.

Geometry and role are independent: the same Maze4 data drives the validated single-goal
`Maze` and the research goal maze. The data tree records kind and provenance, with one copy
of each geometry, while the role sets choose uses. The previous `maze_data` re-export shim
is removed; geometry types and lookup now live at `acs2_envs::maze::geometries`.
The `Maze` and multiplexer public paths stay unchanged.

```
acs2-envs/src/
  maze/mod.rs                 validated single-goal dynamics
  maze/topology.rs            walkable cells, pristine perception, shortest distances
  maze/knowledge.rs           pyalcs and multi-goal transition sets
  maze/geometries/mod.rs      MazeGeometry, MazeSource::{Pyalcs, Alcs}, lookup
  maze/geometries/pyalcs/     gym_maze geometry data
  maze/geometries/alcs/       ALCS geometry data
  goal/                       maze, bit_flipping, hand_eye, taxi, knowledge helpers
  roles.rs                    role sets, independent of provenance
  multiplexer.rs              MPX dynamics and knowledge, unchanged
```

**Geometry IDs are archive keys.** IDs including `-v0` and `-ounold` remain exactly as
committed in `reports/bench_rust.csv`, `reports/bench_rust_unold.csv` and `reports/unold_smoke.csv`.
Do not rename them to match the new module names. Directory and Rust identifier renames
are independent of these persisted keys. Historical documents in `docs/archive/` retain
the old names.

Paths below are relative to `acs2-envs/src/` unless stated otherwise.

| Geometry | Role | Definition |
|---|---|---|
| `Maze4-v0` | Validation (P7, P8, P11), performance (P9); Research (multi-goal; coordinate goals for F3/B) | `maze/geometries/pyalcs/maze4.rs` |
| `Maze5-v0` | Validation (P7, P8, P11), performance (P9); Research (multi-goal; coordinate goals for F3/B) | `maze/geometries/pyalcs/maze5.rs` |
| `Maze6-v0` | Research (multi-goal); added parity oracle | `maze/geometries/pyalcs/maze6.rs` |
| `Woods14-v0` | Research (multi-goal corridor); added parity oracle | `maze/geometries/pyalcs/woods14.rs` |
| `MazeMA-v0` | Research (multi-goal branches); added parity oracle | `maze/geometries/pyalcs/mazema.rs` |
| `MazeB-v0` | Research (coordinate goals); added parity oracle | `maze/geometries/pyalcs/mazeb.rs` |
| `MazeF3-v0` | Research (coordinate goals); added parity oracle | `maze/geometries/pyalcs/mazef3.rs` |
| `Maze7-v0` | Validation (P7, P8, P11), performance (P9); Research (multi-goal; coordinate goals for F3/B) | `maze/geometries/pyalcs/maze7.rs` |
| `Woods1-v0` | Validation (P7, P8, P11), performance (P9) | `maze/geometries/pyalcs/woods1.rs` |
| `Woods100-v0` | Validation (P7, P8, P11), performance (P9) | `maze/geometries/pyalcs/woods100.rs` |
| `Cassandra4x4-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/cassandra4x4.rs` |
| `Littman57-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/littman57.rs` |
| `Littman89-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/littman89.rs` |
| `Maze10-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/maze10.rs` |
| `Maze4-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/maze4.rs` |
| `Maze7-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/maze7.rs` |
| `MazeA-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazea.rs` |
| `MazeB-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazeb.rs` |
| `MazeD-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazed.rs` |
| `MazeE1-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazee1.rs` |
| `MazeE2-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazee2.rs` |
| `MazeE3-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazee3.rs` |
| `MazeF2-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazef2.rs` |
| `MazeF3-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazef3.rs` |
| `MazeF4-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/mazef4.rs` |
| `MiyazakiA-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/miyazakia.rs` |
| `MiyazakiB-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/miyazakib.rs` |
| `Woods1-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/woods1.rs` |
| `Woods100-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/woods100.rs` |
| `Woods101-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/woods101.rs` |
| `Woods101_5-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/woods1015.rs` |
| `Woods102-ounold` | Performance (ALCS comparison) | `maze/geometries/alcs/woods102.rs` |

| Environment or task | Role | Definition |
|---|---|---|
| `Maze` | Validation oracle; P9 and ALCS performance | `maze/mod.rs`, `roles::VALIDATION_MAZES`, `roles::PERFORMANCE_MAZES` |
| Multi-goal GoalMaze (Maze4/5/6/7) | Thesis research tasks | `goal/maze.rs`, `roles::RESEARCH_MAZES` |
| GoalMaze (MazeF3/MazeB, coordinates) | Research bridge to predecessor results | `goal/maze.rs`, `roles::RESEARCH_MAZES` |
| Single-goal GoalMaze | Validation bridge to `Maze` | `goal/maze.rs`, `acs2-agents/tests/goal_maze.rs` |
| BitFlipping | Research into the goal-conditioning limitation; HER relabeling oracle | `goal/bit_flipping.rs`, `acs2-agents/tests/bit_flipping.rs`, `roles::RESEARCH_TASKS` |
| HandEye | Thesis research; simulator and knowledge parity oracle | `goal/hand_eye.rs`, `acs2-envs/tests/hand_eye.rs`, `roles::RESEARCH_TASKS`, `roles::VALIDATION_ORACLES` |
| Taxi | Thesis research; gym transition-table parity oracle | `goal/taxi.rs`, `acs2-envs/tests/taxi.rs`, `roles::RESEARCH_TASKS`, `roles::VALIDATION_ORACLES` |
| Goal-port corridor | Validation of goal layout, objective and adapter | `acs2-agents/tests/goal_port.rs` |
| Multiplexer | Additional MPX scaling research and performance | `multiplexer.rs`, `roles::PERFORMANCE_BENCHMARKS` |

The geometry reorganization changes no dynamics, matrix, cap, fixture or archived measurement.
The ALCS geometry tests keep their assertions and warning baseline, now under
`acs2-envs/tests/performance_geometry.rs`.

### GoalMaze — task, representation and starts

`goal::maze::GoalMaze<E, G>` implements `GoalEnvironment<8, G>` with the sealed encodings
`Coordinates` and `NeighbourPerception`. The convenient aliases are `CoordinateGoalMaze`
(`G=2`, joined length 10) and `PerceptionGoalMaze` (`G=8`, joined length 16).
Observation always has pyalcs's eight neighbour symbols, including the original `9` landmark.
Coordinate goal symbols are raw `Token(row as u8), Token(column as u8)`; perception goal
symbols use pyalcs's ASCII digit alphabet. No learning code or `Configuration` changes.

`multi_goal` takes an explicit cell pool and positive cap. `all_walkable` includes both `0`
and `9` cells. The desired goal is uniform in the pool and the start uniform over all
walkable cells except that goal, using one bounded draw and skipping the excluded index.
The pool does not restrict starts. `reset_with_goal` accepts every pool goal; `reset_at` is
a deterministic start/goal entry point. `goal_pool`, `goal_cells`, `position`, `step_cap`
and `distance_to_goal` give the runner the data it needs. None is an agent goal-set port.
The shortest distance is over the eight compass moves, including diagonal moves permitted
by pyalcs, and also supports unambiguous goals outside the real pool; `None` means a
wall/invalid start, an unknown or ambiguous goal, or no route. The unchanged
ALCS Woods101_5 and Woods102 geometries contain disconnected walkable areas.

In multi-goal mode `9` is traversable and can be entered and left. It remains visible as `9`
in every pristine perception but has no native reward or terminal status. Converting it to
`0` would alter the validated perception alphabet; making it absorbing would remove real
start/goal pairs. Every step reports an achieved goal encoded from the actual cell, never
read back from the observation. `terminal_state` is false, and the objective alone decides
success (`ExactMatch`, 1000/0). The cap sets `time_limit_reached`; the port and adapter keep
termination and truncation separate and give success priority on the limit step. Direct
steps after either kind of episode end are refused until reset.

For perception goals, construction checks each pool cell against **all** walkable cells.
A twin outside the pool is still grounds for refusal; silently dropping ambiguous goals
would alter the caller's task distribution. Coordinates retain the full pool on every
geometry. Maze4/5/6/7 have 27/37/37/36 walkable cells respectively, all with unique
perceptions. MazeB has 20 of 26 unique, MazeF3 7 of 9. In the original 27-geometry
catalogue, 15 geometries have a reward-cell twin: 81 twin cells in all, 40 of them in
MazeE3, confirmed exhaustively. MazeF3's `(1,4)` and `(3,3)` both see
`11111101`; the coordinate twin test enters `(3,3)` with desired `(1,4)` and gets zero.
`goal_at` can encode an achieved cell outside the real pool; it does not authorize that
encoding as a desired pool goal or make ambiguous perception relabeling safe.

`single_goal` fixes the pool to the reward cell and uses the geometry's cap;
`single_goal_with_cap` permits an explicit cap. Its reset consumes **no goal draw** and uses
exactly `Maze::reset`'s row-major path-cell list and one bounded start draw. This is a
validation bridge, separate from the uniform multi-goal start distribution. Coordinate
single-goal mode works on every geometry; perception single-goal mode still refuses twins
(including Woods100 and MazeF3).

### Added pyalcs geometries and reference provenance

Maze6 is copied from `gym_maze.envs.Maze6`, with cap 50. Exhaustive probes and a direct
matrix comparison pin its only difference from Maze7: `(3,5)` changes from wall to path.
The pyalcs MazeF3 and MazeB matrices equal the respective ALCS matrices byte for byte, but
pyalcs caps are 50 versus ALCS's 200. The research set chooses new `MazeF3-v0` and `MazeB-v0`
data with the reference caps; the archived `-ounold` versions stay in the performance set.
Explicit research caps are still constructor parameters. Reusing the ALCS variants would
hide the inherited cap difference behind the geometry ID. These additions do not enter
`VALIDATION_MAZES` or the 22-geometry performance set.

`baseline/dump_goal_maze_references.py`, run in the locked baseline environment, creates
`fixtures/goal_maze_probes.json` for these three additions. The original P7 and episode
fixtures remain byte-untouched. New tests compare every path cell times eight actions,
including perceptions, rewards and terminal flags.

### Maze knowledge — two denominators with explicit names

`maze::knowledge::pyalcs_transitions` (`MazeKnowledgeSet::PyalcsPaths`) reproduces
`gym_maze/utils/utils.py:get_all_possible_transitions`: all moves from `0` cells to
non-wall cells, no wall bumps, no starts at `9`. Duplicate perceptual transitions are kept,
so geometry aliases retain pyalcs's weighting. A physical move with identical before/after
perceptions is still included. Maze4 has exactly 115 transitions.

`multi_goal_transitions` (`MultiGoalWalkable`) adds moves **out of `9`**, which a multi-goal
agent can now make. It still excludes wall bumps to keep the measure comparable. This is
the appropriate world-model denominator for the multi-goal task; the pyalcs denominator
remains the literature reference. `goal_transitions<G, M>` appends wildcard goals to both
ends through `GoalLayout<8, G, M>`, matching the ACS2HER maze scripts' knowledge convention.
The multiplexer generator and its inclusion of no-change transitions are unchanged.

`fixtures/maze_knowledge.json` stores 3073 reference transitions for all 30 geometry
variants plus networkx shortest distances between all walkable cells. Rust comparison
uses sorted **multisets**, not sets. Each entry records its reference ID, native cap,
matrix comparison and whether the reference class can initialize. For differing ALCS
matrices (Maze7, MiyazakiA, Woods1, Woods101_5, Woods102), the unmodified pyalcs utility
runs on the exact ALCS matrix; the native pyalcs matrix is retained separately. This proves
the generator convention without pretending that the geometries are identical.

Two native reference classes, Woods101demi and Woods102, assert during construction because
their matrices contain multiple reward cells. The dumper extracts those literal native
matrices only for comparison and records `reference_initializable=false`; it does not patch
the reference. Their ALCS geometries have one reward and can be passed to the unmodified
knowledge utility normally. Native environment parity is not claimed for these two.

### BitFlipping — canonical HER task and the ACS2 limitation

`goal::bit_flipping::BitFlipping<N>` implements `GoalEnvironment<N, N>`. State and goal
use ASCII `0/1` symbols, action `i` flips exactly position `i`, and the goal stays fixed.
`new` uses cap `N`; `with_step_cap` takes a positive explicit cap. Start and goal are
uniform binary vectors conditioned on being different. `reset_with_goal` samples that
same conditional start distribution. `reset_at` supports deterministic oracles, and
`distance_to_goal` is the exact Hamming distance. The whole real goal pool is exposed as
a lazy `BitGoals<N>` iterator to avoid allocating `2^N` vectors in an environment.
`knowledge_transitions` lazily enumerates every binary state times every action.

The task follows [Andrychowicz et al., HER, §3.1 and Appendix A](https://arxiv.org/abs/1707.01495)
in its bit dynamics, episode cap and success at any visited step. Two deliberate differences:
our objective pays 1000 for reach and zero otherwise rather than the paper's 0/−1, because
ACS2 ranks by `q*r`; and reach immediately terminates, whereas the paper describes
fixed-length episodes with an any-step success measure. Starts already satisfying the goal
are excluded explicitly. The cap truncates an unsuccessful episode and success wins on
its final step.

This is a research task about a limitation, not a claim that every canonical HER task
supports an unchanged ACS2 goal-conditioned value. Covering specifies positions that
changed; subsequent specialization comes from failed anticipations and their marks.
In BitFlipping, the transition rules never fail in another goal context, so no goal
position is added to a condition and reward estimates remain on goal-agnostic classifiers.
Walls in a maze, or the held/unheld context in HandEye, can create those failures and make
a goal position a mark difference. HER only changes goals and rewards, not transitions;
therefore it cannot itself supply the missing failed anticipations. This is a working
criterion for selecting tasks, not a theorem about all ACS2 variants or all configurations.

`no_failed_anticipation_means_acs2_cannot_specialize_on_bit_flipping_goals` runs unchanged
ACS2 through the adapter, N=4/6/8, seeds 42–44, GA off and on, 300 exploration episodes.
After each episode it requires wildcard goal condition/effect positions and no marks;
the final population is exactly `2*N`. Exhaustive N=4 tests check every start/goal/action,
non-negative objective rewards, termination versus truncation at cap one, flip index,
and Hamming distance. Seeded adapter tests pin the goal suffix and `reset_with_goal`.
The task remains useful for exact relabeling oracles even though unchanged ACS2 cannot
learn goal-dependent values here. This task remains outside the maze HER comparison.

### HandEye — simulator parity and block-position goals

`goal::hand_eye::HandEye<SIDE, S>` enforces `S=SIDE*SIDE+1`; aliases `HandEye3`,
`HandEye4`, `HandEye5` have state lengths 10/17/26 and joined lengths 12/19/28.
The pure `HandEyeState` holds gripper `(x,y)`, block `(x,y)` and held status. A held block
must occupy the gripper cell. Actions are N, E, S, W, grip, release in simulator order;
borders block moves. The row-major observation uses `w/b/g`; the gripper hides an unheld
block below it, a held block appears as `b`, and the tactile symbol is `0/1/2` with
`note_in_hand=True`. Coordinates are raw token values `(x,y)`, unlike the maze's `(row,col)`.
The achieved goal always comes from the physical block position, including when hidden.

The desired goal is uniform over all `SIDE*SIDE` cells except the block's start.
The default start distribution is exactly the simulator's distribution: uniform block,
half held (gripper at block), otherwise an independently uniform gripper. This is **not**
uniform over the reachable states: a particular held state is `SIDE*SIDE` times as likely
as a particular unheld state. `reset_with_goal` conditions that same distribution on
block position differing from the desired goal. Constructor caps are positive and explicit.
Seeded draw-order tests pin the default distribution; evaluation never needs an agent goal pool.

Success uses the Fetch block-position convention: transport into the goal while still
holding the block pays 1000 and terminates, with no release requirement. Requiring a release
would add an extra predicate and make the task different from the chosen position goal.
Distance is zero if the block is already at the goal; otherwise held starts need block-to-goal
Manhattan distance, and unheld starts need gripper-to-block distance + one grip + transport.
All distances are checked against networkx BFS over the independent simulator fixture.
`states`, `goal_pool`, `state`, `distance_to_goal`, `reset_at` and `reset_with_goal` provide the
runner and deterministic oracles with world state and real goals.

`baseline/dump_hand_eye_references.py` runs the unmodified simulator on every reachable
state times six actions: 90/272/650 states and 540/1632/3900 probes for grids 3/4/5.
`fixtures/hand_eye.json` records complete before/after perceptions and physical states.
Knowledge comes directly from `gym_handeye/utils/utils.py:get_all_possible_transitions`:
only perception-changing transitions, 258/848/2130 respectively. Rust compares multisets,
keeping multiplicities, and checks every admissible start/goal/action's objective and flags
at cap one. A held block reaching the goal on a later cap step also terminates exclusively.
The fixture catches a held block not moving with the gripper, an x/y swap, changed action
order, a visible gripper hiding/holding the wrong symbol, and a tactile sensor missing `2`.
The reference's sequential goal generator is deliberately absent: this task uses block
position goals, not its move/grip/release sequence. No claim covers `note_in_hand=False`.

### Taxi — a four-goal delivery task

`goal::taxi::Taxi` implements `GoalEnvironment<3,1>` (joined length 4). Observation is
raw numeric tokens `(row,column,passenger_location)`, with 0–3 for R/G/Y/B and 4 for in
Taxi; the desired goal is a stand token, and achieved is the actual passenger location.
Being in the taxi at the goal stand is not success: dropoff must place the passenger there.
The pool has exactly four goals; its small size is a task property, not grounds for adding
artificial destinations. All caps are explicit positive constructor parameters.

The pure `TaxiState::after_action` implements gym 0.23's 5×5 map and action order
south, north, east, west, pickup, dropoff. Dropoff at any stand places the passenger there;
only the desired stand ends the goal task. Illegal actions and wall bumps leave state
unchanged. Native gym rewards −1/−10/+20 are replaced by the shared 0/1000 objective;
no native penalty enters ACS2. Each state-changing transition is independent of the desired
stand, so it remains valid under goal relabeling. Gym 0.23's successful dropoff explicitly
sets passenger location to the destination; it does not leave the passenger in the taxi.

Reset is uniform over gym's 300 admissible initial tuples: taxi in any of 25 cells,
passenger at one of four stands and destination at a different stand. `reset_with_goal`
conditions on the chosen destination and `reset_at` permits deterministic admissible starts.
`goal_pool`, `state` and BFS `distance_to_goal` expose what the goal measurement layer needs. Distances count
pickup and dropoff actions as well as movement and include all internal walls.

`baseline/dump_taxi_references.py` reads the pinned gym transition table `P`: 500 encoded
states × six actions = 3000 probes. All decoded next states match Rust exactly, including
states outside the initial distribution. Native done flags agree with the goal objective
for starts whose passenger is not already at the desired stand; starts already satisfying
the goal are refused by task resets. Networkx distances agree for all 125 physical states
and four goals. The fixture also pins the 300 equally weighted initial states.
Importing gym 0.23's toy-text package requires `pygame` even with no rendering, so baseline
now pins `pygame==2.6.1`; gym and numpy pins are unchanged. Reference source is unmodified.

There is no pyalcs Taxi knowledge utility. `knowledge_transitions` enumerates all 125
physical states × six actions and keeps only observation-changing transitions, following
HandEye's convention. There are 348 transitions. It does not multiply them by the four
unused destination labels. The exhaustive `P` comparison validates this set independently;
no claim is made that it is a literature denominator.

The tests catch missing taxi walls, wrong action ordering or stand coordinates, passenger
removal on wrong-stand dropoff, rewarding proximity while still carrying the passenger,
negative native rewards leaking into the task, cap/goal flag swaps, and a wrong suffix offset.
Wrong-stand dropoff stays active before the cap; cap-one success terminates exclusively.
Seeded goal evaluation covers every real goal. Native reward parity is deliberately not
claimed: those rewards are replaced by the objective.

### Goal environment interface and exact random-policy sparsity

The core port stays unchanged. The runner can read each concrete environment's `step_cap`,
`goal_pool`, current physical state (`position` for mazes, `state` for the other tasks), and
`distance_to_goal`; every task implements `reset_with_goal` for its real pool and provides
`reset_at` for deterministic oracles. BitFlipping's full binary pool is lazy; other pools
are slices. Coordinate maze distances also work for valid achieved goals outside the real
pool. Perception distances refuse ambiguity. `goal::knowledge::with_wildcard_goals<S,G,M>`
lifts any state-only knowledge iterator to joined perceptions, with wildcard suffixes on
both ends. Maze's named generators keep their own convenience wrapper.

The pool belongs to the runner, not the agent. TU still collects desired goals from the
agent's episodes. These environment implementations provide task dynamics; the goal-agent
contract, budget accounting and evaluation live in the measurement layer below. HER agents are constructed by the measurement layer, as described below. Randomness enters only via
the injected `RandomSource`; callers use separate seeded ChaCha streams for environment,
agent and evaluation. Tests pin same-seed resets and complete trajectories for every task.

`baseline/compute_random_policy.py --out <outside-checkout.json>` computes the following
success probabilities **exactly as rational numbers**, storing those fractions alongside
rounded decimal views. This is an analytical environment tool, not a training runner.
For an A-action kernel, count sequences of length t hitting the goal by t: start with
`C_0(s,g)=1[s reaches g]`; at a reached state `C_t=A^t`, otherwise
`C_t(s,g)=sum_a C_(t-1)(next(s,a),g)`. Integer counts divided by `A^t` give the absorbing-state
probability. Average with the exact task start and desired-goal weights. No simulation,
RNG, fitted model or floating-point recurrence enters the calculation. BitFlipping uses
its Hamming-distance birth/death chain and binomial initial weights instead of all `2^n`
states. Python `Fraction` retains exactness at caps up to 200.

All tables are **percentages**. Maze starts/goals range over all walkable cells, including
`9`, with start != goal and eight uniformly sampled actions. BitFlipping conditions out
already-reached starts. HandEye uses the simulator's half-held distribution, then conditions
out the desired block location, and succeeds without release. Taxi uses gym's exact
300-state initial distribution and six actions. These are baselines for the pilot grid to choose
configurations from, not evidence about ACS2 or HER policy quality.

Maze cap columns, descending:

| Task | 50 | 25 | 20 | 10 | 5 | 2 | 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maze4-v0 | 46.302073 | 29.331725 | 25.065609 | 15.005831 | 8.645630 | 3.939637 | 2.101140 |
| Maze5-v0 | 31.587719 | 19.421485 | 16.521240 | 9.844277 | 5.688921 | 2.604167 | 1.388889 |
| Maze6-v0 | 32.504069 | 20.172753 | 17.190554 | 10.264440 | 5.927366 | 2.709741 | 1.445195 |
| Maze7-v0 | 30.863918 | 19.652193 | 16.846240 | 10.188956 | 5.923050 | 2.715774 | 1.448413 |

| Bits (= cap) | Success (%) |
|---:|---:|
| 4 | 21.041667 |
| 5 | 13.135484 |
| 6 | 7.935691 |
| 7 | 4.696347 |
| 8 | 2.720795 |
| 9 | 1.551231 |
| 10 | 0.871945 |
| 11 | 0.484560 |
| 12 | 0.266658 |
| 13 | 0.145554 |
| 14 | 0.078896 |
| 15 | 0.042509 |
| 16 | 0.022786 |

| Cap | HandEye 3 (%) | HandEye 4 (%) | HandEye 5 (%) |
|---:|---:|---:|---:|
| 1 | 2.777778 | 1.666667 | 1.111111 |
| 2 | 4.681070 | 2.795139 | 1.859259 |
| 5 | 8.554813 | 5.097683 | 3.389626 |
| 10 | 12.315083 | 7.271381 | 4.811615 |
| 20 | 16.594623 | 9.545986 | 6.215131 |
| 30 | 19.610849 | 10.972070 | 7.018367 |
| 50 | 24.739549 | 13.196164 | 8.167293 |
| 100 | 35.772248 | 18.023233 | 10.510389 |

| Cap | Taxi success (%) |
|---:|---:|
| 50 | 0.333072 |
| 75 | 0.809786 |
| 100 | 1.426356 |
| 150 | 2.938676 |
| 200 | 4.673316 |

Maze4 cap 50 is **46.302073%**, rather than the probe's finite-sample 49–53% range; it is
still not a sparse task. Caps around 2–5 reduce the four maze baselines to roughly 3–9%.
Taxi cap 50 is **0.333072%**, not a mathematical zero; seeing no successes in a finite probe
is consistent with that probability. At cap 200 it is 4.673316%. BitFlipping n=8 is
2.720795% and n=16 is 0.022786%. HandEye g=3 cap 30 is 19.610849%, g=4 cap 50 is
13.196164%, and g=5 cap 50 is 8.167293%. Changing HandEye's initial holding probability or
requiring release would invalidate these numbers.

`baseline/test_random_policy.py` validates the counting method against every action
sequence for four-bit flipping, the maze one-step edge count, the HandEye one-step
half-held distribution and Taxi's impossible delivery in its first two steps. These four
analytical tests are separate from the 45 existing tests under `tools/`. The JSON table and
one-off report stay outside the checkout; the reproducible calculator and key numbers live
here. No archived measurement under `reports/` is edited.

### Single-goal equivalence and mutation coverage

`acs2-agents/tests/goal_maze.rs` compares coordinate single-goal `GoalConditioned` against `Maze` on all
30 geometries, seeds 42–44, 20 complete random-action episodes per case: reset state and
position, every next observation's state part, reward, termination and truncation.
For learning, all eight pyalcs geometries, the same seeds and 100 exploration episodes are
compared after **every episode**, with GA off and both agents in the same
truncation mode (each mode is checked): steps, total reward, population order and
size, first-eight condition/effect/mark attributes, action, numerosity, experience, ALP/GA
timestamps, `ee`, and bit patterns of `q`, `r`, `ir`, `tav`, plus agent RNG state.
Coordinate suffixes always pass; perception suffixes pass wherever construction is safe.
Suffix condition/effect positions remain wildcards and each suffix mark holds at most one
symbol. Full marks across dimensions are not equal: a constant suffix may be recorded in a
mark, but cannot contribute a difference. No equality claim is made for GA-on learning.

The tests catch: goal derived from the perception (MazeF3 coordinate twin), swapped terminal
and truncation flags, wrong goal join offset (adapter suffix and bridge state), extra goal
RNG drawing or altered start enumeration in single-goal mode, lost/relabelled `9` landmarks,
perception-twin checks restricted to the pool, omitted reward-cell departures, inclusion of
wall bumps or starts at `9` in the pyalcs set, collapsed duplicate transitions, and incorrect
shortest distances including disconnected cells. Every coordinate transition on every
geometry is checked against physical cell success and the objective at cap one; this also
pins non-negative rewards and success winning on the limit step. Seeded reset/move parity
and complete start-index enumeration pin determinism and the conditional uniform start.
The tests do not prove arbitrary future objectives, GA-on equivalence, or safe HER relabeling
with ambiguous perception goals; those are outside this bridge's contract.

### Separate random streams

`ChaChaRandomSource::from_seed_and_stream(seed, stream)` gives each consumer of one seed its
own ChaCha stream; stream 0 is `from_seed(seed)` exactly. The existing runners hand the
environment and the agent the same seed on stream 0, so both draw the same word sequence.
That stays as it is — changing it would change every archived result and the P9 gate — and
the goal line gives agent, environment and evaluation separate streams.

### Why a plain goal suffix is enough — measured before building

The worry was that ALP never specializes on goal positions, because dynamics do not depend on
the goal, leaving values goal-agnostic and trajectory utility tied across goals. `tu_probe`
measured it on Maze4 with a goal drawn per episode from all 27 walkable cells or from a fixed
subset of 4, perception and coordinate goals, goal-conditioned ACS2 and a store-time HER,
3 seeds, up to 2000 episodes.

**What counts as the agent's value.** `Q(s,a|g)` is the maximum fitness over the
change-anticipating classifiers of the action set for `s‖g`, and `V(s|g)` its maximum over
actions — exactly the classifiers `best_change_anticipating_action` and
`Population::get_maximum_fitness` consult. The first version of the probe took every matching
classifier; an independent review pointed out that a goal-specific no-change classifier the
agent never consults could then create goal dependence on its own. The probe now reports both
scopes, and `a_goal_specific_classifier_the_agent_ignores_moves_only_the_all_scope` pins the
difference. Chance baselines are averaged over the same trajectories as the hits they are
compared with — those whose utility has a unique maximum.

After 2000 episodes, agent scope: 66–87% of all classifiers and 69–89% of change-anticipating
ones specify a goal symbol (0% with a fixed goal — the variation is the cause: marks record the
goal of a failure like any other wildcard position, and `get_differences` specializes on it);
`V(s|g)` differs across goals on 37–99% of steps; the utility argmax is unique for 77–99% of
trajectories and names the goal nearest the trajectory at 0.64–0.80 against a chance rate of
0.37–0.58, with rank correlation to proximity 0.25–0.52. Utility is not flat. Per seed, 23 of
24 seed-configurations beat chance by at least 0.10; one sits at chance with a positive rank
correlation. Its margins are small (2–5% of the maximum). Early in training it is weak rather
than inverted — at 125 episodes the mean rank correlation is 0.04–0.35 and the 27-goal
perception ACS2 configuration is at chance, with single seeds down to −0.19 against chance.
The goal specialization that produces it grows the population 3–7x against the single-goal
task. The all-classifier scope reproduces the first version exactly, 120 rows by 19 fields.

HER relabels with achieved cells whether or not they belong to the goal subset. That is the
definition of HER — its goals are the non-target states a trajectory reached — and it is what
trajectory utility is meant to complement with goals from the real set.

What is deliberately absent: an agent goal-set API (the real goal set is the set of desired goals an
agent has been given — TUCA-HER collects it from episodes), checkpointing of goal agents, and
any change to `Classifier`.

## Goal measurement layer

`acs2-measure` is a separate workspace package. Its CLI takes a typed research-task name,
positive episode cap, an explicit goal pool (`full` or listed real goals), encoding, agent,
seeds, ordered environment-step evaluation points and an output path outside the checkout.
The task registry is `acs2-envs::roles::ResearchTask`; validation oracles and performance
benchmarks are typed there too. `VALIDATION_MAZES` and geometry IDs remain unchanged. This
package keeps experiment policy and output formats out of `acs2-core` and the MPX binaries;
adding these to `acs2-bench` would mix thesis metrics with the multiplexer path.

The runner owns the environment-step counter. One `GoalAgent::train_episode` call receives
one `TrainingEnvironment` episode; a second reset, a step outside the episode or an
unfinished episode fails. Evaluation is taken after the first completed episode meeting
each nominal point, without cutting it. The actual count and nominal point are both in the
row; overshoot is strictly less than the configured cap. An agent's returned trial metrics
never set the budget. This excludes the K-heads/K-episodes-per-trial comparison defect.
`MeasuredEnvironment` also exposes the desired goal and the last `GoalTransition`: its raw
`GoalStep` contains the achieved goal and time-limit state, while its `GoalOutcome` keeps
termination and truncation separate. An agent can retain steps until the episode ends and
call `relabel` on any retained step. That method calls the task environment's own objective
through `GoalStep::outcome`, the same function used for the original transition. A MazeF3
coordinate test checks that relabeling the perception twin `(3,3)` with `(1,4)` neither
pays nor terminates, while entering `(1,4)` does both. `MeasuredEnvironment::episode_start`
exposes the original `GoalStart` after reset, including the achieved goal of the initial
state. The runner does not give the agent the full goal pool; agents needing a candidate
set collect desired goals from their own episodes. Deriving achieved goals from
observations would confuse MazeF3's twins; copying reward logic into agents would let
training and relabeling disagree. The unchanged core `LearningAgent` API still receives
an `Environment` from the wrapper.

`GoalAgent` separates training, a named evaluation policy, read-only action and value
selection, update counts, and component-size reports. `CoreAgent` adapts the unchanged
`LearningAgent` implementations of ACS2 and ACS2ER. `TrainingEnvironment` supplies their
goal-concatenated observations while retaining the raw goal step. `CoreAgent` declares
`greedy_change_anticipating_population`: `BestAction` selects from the population on
state concatenated with desired goal, and the reported first-action estimate is the
`MaxFitnessBootstrap` maximum over that action set. A future multi-population agent must
implement `eval_action` for its actual deployed policy and aggregate its population costs;
the runner neither selects head zero nor assumes one population. It does not call
`run_exploit_trial`, whose reinforcement-learning update would invalidate evaluation.

Evaluation creates a separate environment and tie-break generator, never borrows the
training generator or mutable population. It resets to each specified start–goal pair,
then reports cap success, mean steps among successes, mean ratio of successful steps to
shortest distance, and weighted means of first-action estimate, realized discounted return,
and their difference for all pairs and for successful pairs. The successful-pair means are
absent when there are no successes. Per-start actions, estimates and returns are available
with `--per-start`; the CLI omits them by default so long grids stay compact. Small tasks
use all pairs with their task-defined weights; BitFlipping with more
than 20,000 pairs uses a fixed 8,192-pair sample and reports a Bernoulli standard error.
The seed and stream of that sample are fixed and recorded. A random rollout would be
cheaper but would add avoidable noise for the small candidate tasks. An exploitation trial
would learn, so it is not an evaluation substitute.

The random-policy floor uses absorbing-state dynamic programming on the task's complete
deterministic transition kernel, with uniform actions and exactly the evaluation's
start–goal weighting. BitFlipping uses an equivalent Hamming-distance recurrence. Exhaustive
BitFlipping uses binomial start-distance weights; sampled BitFlipping weights the recurrence
by the realized fixed sample. Reachability uses that sample's pair weights too. Consequently
the floor and ceiling in every row describe the same evaluated distribution as success;
the full-law analytical values are not emitted for sampled rows. At BitFlipping16 cap 1 the
fixed sample has one distance-one pair, giving a floor of `1/131072` and a ceiling of
`1/8192`, unlike the full-law `1/65535` and `16/65535`. These are exact probability
algorithms represented as
`f64` in output; the fixed rational oracles independently test key values. Maze4 cap 5
has floor `2833/32768` with all 27 goals and `334385/3407872` with the restricted
`[(2,5),(5,5),(6,3),(6,4)]` pool. HandEye4 cap 50 has floor
`0.13196163920912146`. The ceiling is the weighted share whose exact shortest distance
is within the cap. Two other fields distinguish a reachable pair beyond the cap from a
missing or ambiguous distance. A simulation-derived floor or a ceiling based only on
whether the goal is ever reachable would misstate short-cap tasks.

The preset lists every `Configuration` field instead of silently inheriting
`default_protocol()`: eight or six task actions, β `0.05`, γ `0.95`, θi `0.1`, θr `0.9`,
θexp and θas `20`, θga `100`, μ `0.3`, χ `0.8`, `u_max=100000`, ε `0.8`, initial
q/r/ir `0.5/0.5/0`, GA/PEE/planning off, subsumption on, Pyalcs ALP generalization,
and corrected truncation bootstrap. ACS2ER uses a 10,000-sample FIFO, warmup one and
three updates per environment step, a volume reserved for replay-based successors.
Streams from one seed are agent `1`, training environment `2`, evaluation policy `3`,
restricted-pool draw `4`, evaluation-pair sample `5`, evaluation environment `6`.
The full-pool tasks keep their native reset draw order; restricted HandEye, Taxi and
BitFlipping draw a real goal then call the task's `reset_with_goal`, retaining its
goal-conditioned start law. A hidden default, shared random generator or a restricted
pool that silently falls back to all goals would make cross-agent comparisons opaque.

Costs are cumulative at each evaluation point. Environment steps and completed episodes
come from the environment wrapper. ACS2 makes one online RL update per stepped
transition; ACS2ER's replay count follows its buffer length, warmup and sampling rule
exactly. `Population::form_match_set` increments an opt-in thread-local counter by one
formation and the current population length in classifier–perception tests. This existing
core hook does no matching, learning or random draw and is unchanged in this round. A
guard disables counting on normal exit and during unwinding, including a caught agent
panic. Evaluation pauses the counter, so training cost is not inflated by exhaustive
measurement. The output keeps population classifier count and numerosity
separate from inline classifier bytes, mark-entry count and a known-byte lower bound;
replay sample count and payload bytes are separate, as is future trajectory storage.
Allocator overhead, `BTreeSet` node overhead and unused vector capacity are not included
in these logical sizes. Peak RSS would not separate components and is not used. Cumulative
training and evaluation wall seconds are separate fields, with their sum provided for
convenience. Every row includes the runtime host name and CPU model; time remains secondary
because shared-node load changes it. A reusable read-only test runs identical training
with and without evaluation points and compares the full agent snapshot. It covers ACS2
and ACS2ER and rejects an agent that mutates interior state through `eval_action(&self)`;
the receiver type alone cannot guarantee read-only behavior.

One schema-2 JSONL row represents configuration × agent × seed × evaluation point. The
runner emits it immediately after evaluation; the CLI writes and flushes it before the
next training episode. It repeats the build-time commit and source state (`clean`, `dirty`
or `not_git_checkout`), preset, pool, encoding, distribution, floor and reachability
fractions, costs, component sizes and aggregate diagnostics. Build provenance comes from
the `acs2-measure` build script rather than a Git query in the run directory. The output
uses `goal_pool: "full"` for a task's complete pool and lists goals only for a restricted
pool; this avoids repeating the 65,536 BitFlipping16 goals in every row. The output
guard rejects output inside any Git repository, including a linked worktree, by checking
the output path's runtime ancestors for `.git`; it does not require the source path used
at build time to exist. Holding all rows until completion would lose finished evaluation
points on interruption; querying Git at run time could name a different checkout or no
commit. A process killed during a write can leave an incomplete final JSONL line, which
readers must discard. Each complete row remains interpretable without its siblings. Raw
experiment files stay outside Git repositories and are never added to `reports/`. Tests
pin the floor against independent rational references, read-only evaluation,
environment-owned budget, declared policy, fixed streams, replay volume and goal-pool
conditioning.

## Pilot grid: ACS2 and ACS2ER on ten goal configurations

This registered grid predates the supervisor's train/validation/test goal protocol.
Its maze results are a pilot, not the final thesis comparison. HandEye, Taxi and
BitFlipping results describe limitations outside the main discrete multi-goal maze
comparison. The registration, data contracts and original tooling remain fixed;
`experiments/README.md` indexes the question, provenance and reproduction commands.

The accepted grid fixes seeds 42–61, the thesis preset, equal interaction budgets and
points for both agents, and the goal pools below. Plan SHA-256:
`862f77785f1cf73fc58e81a88fabeac4b708ee908082f227a499d5cb2e4efb98`.
Measurement commit `af3ce49b2df0f5a764efb49be9c48da54ebe0040`, clean musl binary
SHA-256 `1643a9e5c365cbc43fb119cc5d5d136b680303f66bd77d4ff06c765ad0a7ec72`.
All 400 selected runs and 3,800 rows passed manifest/provenance checks; both seed-42
agents in every configuration match the pilot in 8,550 learning fields. Raw rows and
full analysis remain outside the checkout in the archived pilot grid.

The primary score is the arithmetic mean success over registered points **within each
seed**, with equal point weights. It is not a step-weighted integral. Values below are
mean ± SE across 20 seeds; paired ER−ACS2 intervals use Student t(19). These are marginal
95% intervals, without simultaneous multiplicity adjustment. They do not identify a
causal mechanism or establish convergence.

| Configuration | ACS2 score ± SE | ACS2ER score ± SE | Paired ER−ACS2 [95% CI] | Registered verdict |
|---|---|---|---|---|
| bitflip8_c8_full | 0.0064453 ± 0.0003737 | 0.0075884 ± 0.0002823 | 0.0011431 [0.0001608; 0.0021254] | floor_baselines_retain_for_successors |
| handeye4_c50_p4 | 0.3070677 ± 0.0145898 | 0.3081432 ± 0.0175754 | 0.0010755 [-0.0376943; 0.0398453] | compare |
| maze7_c10_full | 0.1871944 ± 0.0033240 | 0.1463968 ± 0.0037976 | -0.0407976 [-0.0511991; -0.0303961] | compare |
| handeye4_c50_full | 0.2492734 ± 0.0097329 | 0.2211413 ± 0.0069083 | -0.0281322 [-0.0537877; -0.0024766] | compare |
| maze6_c10_full | 0.1866517 ± 0.0031975 | 0.1462012 ± 0.0044128 | -0.0404505 [-0.0535013; -0.0273996] | compare |
| maze6_c10_p4 | 0.4071875 ± 0.0099269 | 0.3809375 ± 0.0077742 | -0.0262500 [-0.0583797; 0.0058797] | compare |
| handeye5_c50_full | 0.1170219 ± 0.0028896 | 0.1070583 ± 0.0027437 | -0.0099635 [-0.0185153; -0.0014118] | compare |
| maze4_c5_p4 | 0.6241346 ± 0.0062924 | 0.6200481 ± 0.0140070 | -0.0040865 [-0.0364193; 0.0282463] | compare |
| maze4_c5_full | 0.2196296 ± 0.0040457 | 0.1938604 ± 0.0034072 | -0.0257692 [-0.0373519; -0.0141865] | compare |
| taxi_c200_full | 0.0029500 ± 0.0007747 | 0.0070833 ± 0.0014061 | 0.0041333 [0.0006426; 0.0076241] | floor_baselines_retain_for_successors |

The registered floor verdict uses the **final-point** lower confidence bounds relative
to the row's exact random-policy success; retain these configurations for HER/TU/VCP
without ranking the baselines. Otherwise ACS2 at the first point reaching at least 90%
of task reachability would be too easy; no configuration meets that condition. All
remaining configurations are compared. Final-budget success is secondary:

| Configuration | Steps | Floor | Reachable within cap | ACS2 final ± SE | ACS2ER final ± SE | Verdict | Train steps/CPU-s ≈ ACS2 / ER |
|---|---|---|---|---|---|---|---|
| bitflip8_c8_full | 1000000 | 0.0270763 | 1.0000000 | 0.0073303 ± 0.0009888 | 0.0072388 ± 0.0007424 | floor_baselines_retain_for_successors | 1515853 / 403656 |
| handeye4_c50_p4 | 10000000 | 0.1203966 | 1.0000000 | 0.4747396 ± 0.0284969 | 0.4115885 ± 0.0299420 | compare | 57099 / 9481 |
| maze7_c10_full | 10000000 | 0.1018896 | 1.0000000 | 0.2920238 ± 0.0109073 | 0.1767857 ± 0.0068345 | compare | 41418 / 10233 |
| handeye4_c50_full | 10000000 | 0.1319616 | 1.0000000 | 0.3738997 ± 0.0195017 | 0.2931055 ± 0.0164063 | compare | 49201 / 9052 |
| maze6_c10_full | 10000000 | 0.1026444 | 1.0000000 | 0.2911787 ± 0.0071204 | 0.1760511 ± 0.0067531 | compare | 45591 / 11718 |
| maze6_c10_p4 | 10000000 | 0.0936023 | 1.0000000 | 0.5204861 ± 0.0221299 | 0.4711806 ± 0.0190932 | compare | 102650 / 14410 |
| handeye5_c50_full | 2000000 | 0.0816729 | 1.0000000 | 0.2314667 ± 0.0098668 | 0.1719050 ± 0.0069123 | compare | 19053 / 3923 |
| maze4_c5_p4 | 10000000 | 0.0981213 | 1.0000000 | 0.7504808 ± 0.0128527 | 0.7091346 ± 0.0204435 | compare | 154858 / 23454 |
| maze4_c5_full | 10000000 | 0.0864563 | 0.9971510 | 0.3098291 ± 0.0079370 | 0.2399573 ± 0.0058747 | compare | 80143 / 20156 |
| taxi_c200_full | 10000000 | 0.0467332 | 1.0000000 | 0.0000000 ± 0.0000000 | 0.0000000 ± 0.0000000 | floor_baselines_retain_for_successors | 950352 / 206112 |

Throughput is the mean of per-run ratios on **AMD EPYC 9554 64-Core Processor**,
`lem-cpu-normal`, one CPU per run. Training/evaluation CPU attribution estimates 20 ms
schedstat samples aligned to runner wall intervals; total process CPU and peak RSS come
from wait4. Shared-host load and timing alignment limit cost comparisons. BitFlipping8
uses its fixed evaluation sample; between-seed intervals exclude that sample uncertainty.
The full pointwise statistics, paired seed scores, resource counters and attempt audit
are produced by `experiments/baseline-pilot/analyze_grid.py`.

Points are 1k, 3k, 10k, 30k, 100k, 300k, 1M, 2M, 5M, 10M, truncated at the listed
budget. The restricted pools in CLI row:column coordinates are Maze4
`2:5,5:5,6:3,6:4`, Maze6 `3:1,3:6,4:3,7:7`, and HandEye4 `0:0,0:2,2:3,3:1`.
Other pools are full. Pool selection was fixed without outcomes. HandEye row encoding
uses x,y; preserve the runner's CLI conversion when reusing these pools.

All 667 attempts are retained: 400 complete selections, six lease-refused starts,
260 zero-CPU cancellations and one nine-row incomplete attempt stopped by an operational
watchdog error. The complete replacement agrees on all 405 shared learning fields.
No measurement binary was changed. Allocated cost was 161,965 CPU-seconds for the grid
plus 7,905 for the pilot: **47.186111 CPU-hours**, below 200. No active reservation remains.
Operations are independently versioned; the final launcher selects the exact Slurm array
element, retains its allocation response and enforces node-side checks. Future comparisons
must preserve these scientific configurations and report floor cases, finite budgets and
all attempts; a flat observed interval does not establish a learning ceiling.

## Trajectory storage and relabeling

`acs2-trajectory` depends only on `acs2-core`. It owns storage, selection, scoring,
sampling and exact composition, without learning, task construction or output policy.
HER and future TU/VCP agents live in `acs2-agents` and are constructed by
`acs2-measure`. Benchmarks also construct ACS2/ACS2ER from that package. There is no
dependency cycle: trajectory storage and core/environment packages never depend on
agents or measurement. Nothing changes `Configuration`, `ReplaySample`, agent descriptions,
checkpoint identities or codecs.

### Raw history, identifiers and capacity

`TrajectoryStore<S,G>` stores whole `StoredEpisode`s: the environment's `GoalStart`
(observation, achieved goal, desired goal), then each action and unmodified `GoalStep`
(observation, achieved goal, terminal-state and time-limit flags). It never stores an
authoritative reward, terminated/truncated flag, done bit or joined perception.
`begin_episode` registers the desired goal immediately, including the current episode;
`insert` registers it for callers that built an `Episode` separately. Candidates are
unique desired goals in symbol order and survive eviction. No constructor accepts the
runner's real-goal pool. Achieved goals, including s_0, come from the environment.

Capacity is a positive, strict limit in environment steps. Insertion evicts the oldest
whole episodes until retained steps plus the new episode fit. Every retained episode
keeps its full prefix, so counterfactual admissibility stays exact. After capacity has
first forced eviction, with episode lengths at most L, occupancy is greater than
capacity minus L and at most capacity. An episode exceeding capacity returns
`StoreError::EpisodeTooLong` and does not evict history; a future agent must configure
capacity at least as large as its environment cap. Empty episodes are rejected.
Identifiers monotonically increase on successful insertion and never get reused;
evicted identifiers cannot resolve. Episode payloads are immutable through the store.
The caller records exactly one complete environment episode before insertion; the
measurement boundary enforces completion. `Episode` refuses steps after a raw terminal
or time-limit flag, but cannot independently infer goal termination without an objective.

`logical_bytes` counts the inline store header, retained episode headers and starts,
step payloads and persistent candidate-goal payloads. Unused capacity and allocator
metadata are excluded. The agent reports this through `trajectory_logical_bytes`,
separately from replay samples and population sizes. Candidate storage can grow after
the step buffer fills, so the step bound is not a bound on all logical bytes.

### One objective, one goal and three meanings of admissibility

`GoalEvaluator` supplies scoring and reach predicates. `ObjectiveEvaluator` calls
`GoalStep::outcome` and `GoalObjective::is_reached`; `MeasuredGoalEvaluator` calls
`MeasuredEnvironment::relabel` and its task-owned `goal_reached` predicate. Thus reward
and termination use exactly the same task objective as collection. No observation
comparison or reward scheme is copied into this crate. `build_sample` joins the same
goal to s_t and s_(t+1), preserves action and derives done with
`TruncationMode::is_terminal`. Negative and non-finite rewards return explicit errors.
There is no relabeling-specific done override or pyalcs reward mode.

Termination is objective reach or a goal-independent terminal state. Truncation is a
time-limit flag on that step without termination. Reaching the new goal on the cap step
therefore terminates; missing it truncates. In `Bootstrap`, truncation does not make
`done` true. Ending the original episode on its own goal supplies no extra ending flag
under another goal: the last relabeled step can be neither terminated nor truncated.
If original success occurs exactly on the cap, however, the raw time-limit flag is
true: under a different unreached goal that last step truncates. The statement that
an original-goal ending always becomes a cut needs this exception; the task's
`GoalStep::outcome` formula is authoritative. The cap/cut test checks both cases.

States are s_0 through s_T; transition t moves s_t to s_(t+1). Strategies are:

- `Original`: the episode's desired goal.
- `Final`: ach(s_T).
- `Future`: uniformly weighted state indices t+1 through T.
- `Episode`: uniformly weighted state indices 1 through T, excluding s_0.
- `UniformReal`: uniform over desired goals observed by this agent at draw time.

Repeated achieved goals keep their state-index multiplicities; they are not made
uniform over unique goals. `GoalDistribution` exposes exact integer weights, their
normalized probabilities, source and admitted counts, and a draw using injected RNG.
Choices have deterministic symbol order. The optional candidate-set filter composes
with every rule, removes goals outside the observed desired set before objective
queries, and renormalizes the remaining index weights.

`EveryTransition` admits every (t,g). `FromNonGoalState` requires that g is not reached
at s_t. `CounterfactualEpisode` requires that g is reached at none of s_0 through s_t;
it includes the transition arriving at the first reached state. Reach means the
objective predicate, which need not be equality. The latter two rules are subsets of
the preceding rule. Original-goal samples from valid collected task episodes pass all
three. `GoalFacts` reports reach at the current state and whether the counterfactual
end has already occurred, including when s_0 reached the new goal.

`relabel_episode` returns indexed samples under one goal and rule. The counterfactual
rule returns exactly the prefix ending on first reach, or no samples for an already
reached start. The end of a non-empty result is `Terminated`, `Truncated` or `Cut`,
read from its last sample. A result without samples ends as `AlreadyReachedAtStart`
under the counterfactual rule and as `NoAdmissibleTransition` under the non-goal rule —
for example under the cell of a HandEye block that never moves — so it cannot be read
as a cut, terminated or truncated episode; every-transition never returns an empty
result. A cut supplies no invented continuation. The less restrictive rules
deliberately retain their different semantics; the non-goal rule can leave gaps, so
indices are kept instead of pretending they form a feasible contiguous episode.

### Drawing, provenance, composition and cost

`SamplerConfiguration` is separate from every existing agent configuration. For each
requested draw, `Sampler` selects a transition uniformly over retained steps, with
replacement, then an independent Bernoulli coin requests relabeling in the configured
proportion. Proportions zero and one skip the coin. `Original` always uses the original
route. Every other strategy draws from its exact restricted distribution. An empty
admissible distribution falls back to this transition's original goal and increments
fallback counters. The transition is not redrawn and the requested number of samples
does not change. Drawing a positive count from empty history returns `EmptyStore`;
zero draws return an empty vector. ACS2ER draws without replacement within its small
batch; both have uniform transition marginals, but their batch covariances differ.

Each `DrawnSample` carries episode identifier, transition index, goal, scored sample,
raw derived outcome, objective cost by purpose and provenance. Provenance names original
versus relabeled route, effective and requested strategy, current-state reach, occurrence
after the counterfactual end, done, membership outside candidates, and fallback.
A successful relabeling remains relabeled even if its goal equals the original goal.
Fallbacks and non-relabeling coin outcomes are original. The future HER agent can use this route to
restrict goal-independent ALP, quality and rule creation to original samples while
updating values on relabeled samples. The HER values variants apply this route-dependent policy through `acs2-agents`.

Cumulative `ReplayCounters` keep one `RouteCounters` per route — draws, the four flags
(reached at s_t, after the counterfactual end, done, outside candidates) and the sum of
rewards — next to fallbacks, strategy counts, failed calls and the cost by purpose.
`pooled()` adds the routes and reproduces the earlier pooled counts exactly;
`given_route()` returns the shares and mean reward among one route's own draws, or none
for a route without draws. At relabel proportion one the original route holds the
fallbacks only. A `draw` call that fails on any of its samples returns the error and
records none of that call's draws, flags, strategies or fallbacks; the objective work it
performed, including its earlier samples and the failed scoring, is added to the cost,
and `failed_draws` grows by one. Callers must abort that invalid replay operation rather
than retry silently.

### Objective cost by purpose

Objective cost counts direct public reward and reach evaluations made by this layer;
internal work inside an arbitrary objective is not instrumented. Every counted call has
exactly one purpose. `CostByPurpose` carries the split in each drawn sample, goal
distribution and relabeled episode and in the cumulative counters; the exact
composition reports the expected calls of a draw for each purpose, and the diagnostics
payload serializes the counters' split:

- *Scoring* computes the drawn sample's reward and termination: one reward call and,
  unless the raw terminal-state flag already ends the step, one reach call.
- *Selection* is the reach calls whose results the admissibility rule reads to decide
  which goals may be drawn. `goal_facts` evaluates a goal at s_t first and then scans
  s_0, s_1, … until the first reach; `FactQueries` counts the two parts. The non-goal
  rule reads the s_t call, the counterfactual rule reads both, every-transition reads
  none (`Admissibility::attribute`). A fallback's empty attempt is attributed the same
  way: its rule read those calls before finding nothing to draw.
- *Provenance* is every other call, made only to fill diagnostic flags: the facts a rule
  does not read, and the original goal's facts on every original-route draw. In a valid
  task episode those cost 1 + t reach calls; they are never selection, so a draw at
  proportion zero — the HER agent's original-only replay control — costs scoring and provenance only. In `relabel_episode`, calls the
  rule does not read count as provenance although that function returns no flags.

For the same draws the three parts sum to the earlier pooled objective-call totals. The distribution
computes facts once per unique proposed goal, sharing them across duplicate state
indices; this also defines the reference sampler's cost. No rejection loop has an
unbounded query cost. Randomness enters only through the supplied `RandomSource`;
thesis callers reserve ChaCha stream **7** for this sampler, leaving streams 1–6
unchanged.

### Exact composition by route

`episode_composition` computes expected shares under uniform transition drawing,
including a supplied relabel proportion, candidate filter and original fallback,
without any RNG. Its pooled values — admissible source mass, empty-distribution
transition share, fallback share, both problematic-goal flags, done, outside-candidate
share, mean reward and expected reward and reach evaluations of a draw — are accumulated
exactly as before the purpose/route split. `original` and `relabeled` are `RouteComposition`s of joint
expectations per draw: the route's share and, for example, P(relabeled ∧ done) and
E[reward · 1{relabeled}]. Joint values add across transitions and episodes, and the two
routes reproduce the pooled values up to rounding. `given_route` divides by the route
share, which gives the composition of that route's draws alone; an aggregate over
episodes therefore adds joint values and divides once instead of averaging per-episode
ratios. Expected evaluations per draw are reported for scoring, selection and provenance
next to the reward/reach totals. The separate `analysis_cost` counts the actual work of
computing this diagnostic — per-goal reach histories and the scoring of each weighted
choice. It is the cost of the measurement rather than of replay and has no purpose split.

`GoalAgent::replay_diagnostics` defaults to absent for ACS2 and ACS2ER. The runner emits
this optional payload at every existing evaluation point, without agent-specific
measurement branches. Rows advance to schema **3**; schema-2 archived baselines remain
unchanged. `acs2-measure::trajectory::replay_diagnostics` serializes the pooled counters
under their original schema field names, `objective_evaluations_by_purpose` (reward, reach and total
calls for scoring, selection and provenance) and `routes` (per route: draws, the four
flags, `reward_sum` and `mean_reward`, null without draws). A recording-only test agent
passes `assert_evaluation_read_only`, including store, policy RNG, sampler RNG and
diagnostics in its snapshot. The later HER adapter consumes the same diagnostics contract.

### Rejected designs and executable evidence

Relabeling at storage time multiplies history entries and confounds goal selection
with the retained environment window. Storing finished samples makes derived reward
and endings authoritative and loses termination/truncation information. Deriving
achieved goals from observations confuses MazeF3's perceptual twins and hidden blocks.
Single-step eviction without retained prefixes loses exact counterfactual semantics;
retaining those prefixes can exceed a step payload bound. Whole episodes trade at most
one episode's occupancy for simple exact semantics. Redrawing an ineligible transition
would bias transition marginals; dropping it would change update volume. Goal lists
from the runner would reveal unobserved real goals to TU. A second learning agent or
checkpoint format is outside the storage and relabeling layer.

`acs2-trajectory/tests/contracts.rs` checks independent admissible enumeration on every
four-bit transition under all goals/strategies/rules/filters, a stationary HandEye
block, and a maze that leaves and returns, including the selection and provenance cost
each distribution attributes. On the same episodes it compares every `relabel_episode`
result — admitted transitions, end status and cost by purpose — with an independent
model, and pins the empty non-goal relabeling under the stationary block's cell. It
checks a non-equality objective with non-1000 reward, MazeF3 twins, cap and cut flags,
non-negative reward enforcement, identical goal suffixes, raw achieved goals,
FIFO/identifier/candidate invariants, weighted duplicates, fixed-seed goal and
transition frequencies, independent relabel share, fallback volume and route
provenance. For every strategy, rule and filter at proportions 0, 0.6 and 1, each drawn
sample's flags, reward and cost by purpose are compared with an independent
enumeration, and the cumulative counters — routes, their conditional means, fallbacks,
strategies and costs — with a recount of the drawn samples. Every field of the exact
composition — pooled values, both routes, their conditional means and the three
purposes — is compared with an independent enumeration on three episodes, every
strategy, rule and filter, proportions 0, 0.3 and 1 and both truncation modes. A draw
call failing on a later sample is checked to leave its draws out of the counters while
counting their objective work, and direct objective costs are compared with counted
calls. `acs2-measure/tests/trajectory.rs` compares samples bit for bit and in order to
actual ACS2ER memory through `TrainingEnvironment`, in both truncation modes on Maze4,
HandEye4 and Taxi, checks research-family rewards, exercises schema-3 diagnostics with
read-only evaluation and maps every counter to its own key in the diagnostics payload.
The probe's own tests map every composition field to its row key and aggregate routes
as step-weighted joint quantities. Baseline regression, mutation results and
random-policy composition are recorded with the composition measurement below.

### Random-policy composition before replay agents

The probe uses all ten registered pilot configurations, seeds 42–61, and the
first complete episode meeting 20,000 environment steps in each run. All 200 runs
completed: **4,003,305 steps, 372,187 episodes**, about **154 seconds** in total on
this laptop. Measurement source is clean `a86b7db2d80f04a9aac0693bd871b7ca8d8f6785`;
the registered pilot plan hash and streams are repeated in every row. Actions are
uniform from stream 1, environment resets use stream 2, and restricted-goal draws
use stream 4. No learning or random diagnostic sampling occurs. The fix round re-ran
the probe from clean `cd950fc805e6f99f86fbfe5dceb1e735955ec4f7` in about **156
seconds**. Its rows advance to schema **2**, which adds the purposes and the routes;
every field the original schema-1 rows carried is bit-identical in all 4,800 rows.

Each completed episode is analyzed before the next reset against desired goals
observed up to that completion. Aggregation weights episodes by their step counts,
then averages the resulting per-seed shares equally over 20 seeds. This describes a
uniform transition draw over the collected first 20k steps, rather than the final
10k buffer alone. Relabel request proportion is one; an empty distribution falls
back to the original goal, so final sample shares include fallback; the route values
separate them. A route-only share is, per seed, the ratio of the step-weighted joint
quantity to the step-weighted route share, never an average of per-episode ratios;
across seeds it is the mean and SE of those per-seed ratios, over the seeds in which
the route has draws. Mean ± SE uses sample variance across seeds, without treating
transitions as independent replicas.

The complete output is 4,800 raw seed rows and 240 configuration/strategy/rule/filter
summary cells. It includes final, future, episode and uniform real, all three rules,
both filter settings, admissible source mass, both problematic-goal shares, done,
outside-candidate share, empty-distribution share, mean reward and expected objective
evaluations per draw, split into scoring, selection and provenance, and the same
composition for each route. The summary script checks that the routes reproduce the
pooled values, the purposes the totals, and that original goals are never already
reached, past their end or outside the candidates. Raw output, summaries and the
report stay outside the checkout.

Key values below are **percentages**, except reward and objective evaluations. These
are `Future`, `EveryTransition`, without the candidate filter; admissible mass is
100% and empty-distribution share is zero for each.

| Configuration | Already reached % ± SE | After counterfactual end % ± SE | Done % ± SE | Outside candidates % ± SE | Mean reward ± SE | Objective evaluations/draw ± SE | Scoring + selection/draw | Provenance/draw ± SE |
|---|---|---|---|---|---|---|---|---|
| maze4_c5_full | 34.780 ± 0.085 | 44.224 ± 0.093 | 65.661 ± 0.050 | 0.630 ± 0.021 | 656.612 ± 0.500 | 5.769 ± 0.007 | 2.000 | 3.769 ± 0.007 |
| maze4_c5_p4 | 35.048 ± 0.062 | 44.414 ± 0.069 | 65.881 ± 0.045 | 85.703 ± 0.103 | 658.813 ± 0.450 | 5.748 ± 0.004 | 2.000 | 3.748 ± 0.004 |
| maze6_c10_full | 30.112 ± 0.076 | 45.610 ± 0.089 | 51.048 ± 0.055 | 1.752 ± 0.056 | 510.477 ± 0.551 | 10.882 ± 0.017 | 2.000 | 8.882 ± 0.017 |
| maze6_c10_p4 | 30.072 ± 0.093 | 45.739 ± 0.090 | 50.971 ± 0.073 | 89.439 ± 0.093 | 509.712 ± 0.733 | 10.881 ± 0.020 | 2.000 | 8.881 ± 0.020 |
| handeye4_c50_full | 86.393 ± 0.205 | 89.155 ± 0.151 | 88.076 ± 0.191 | 3.809 ± 0.221 | 880.757 ± 1.912 | 11.214 ± 0.185 | 2.000 | 9.214 ± 0.185 |
| handeye4_c50_p4 | 85.946 ± 0.232 | 88.903 ± 0.182 | 87.641 ± 0.210 | 78.601 ± 0.441 | 876.406 ± 2.103 | 11.520 ± 0.281 | 2.000 | 9.520 ± 0.281 |
| taxi_c200_full | 89.333 ± 0.379 | 92.641 ± 0.297 | 89.570 ± 0.372 | 19.360 ± 0.740 | 895.703 ± 3.723 | 23.417 ± 0.916 | 2.000 | 21.417 ± 0.916 |
| bitflip8_c8_full | 3.302 ± 0.018 | 10.543 ± 0.065 | 36.477 ± 0.015 | 9.927 ± 0.100 | 364.767 ± 0.147 | 15.427 ± 0.008 | 2.000 | 13.427 ± 0.008 |
| handeye5_c50_full | 88.094 ± 0.256 | 90.553 ± 0.200 | 89.552 ± 0.233 | 6.053 ± 0.319 | 895.520 ± 2.331 | 10.048 ± 0.255 | 2.000 | 8.048 ± 0.255 |
| maze7_c10_full | 31.047 ± 0.096 | 46.940 ± 0.101 | 51.692 ± 0.067 | 1.642 ± 0.065 | 516.915 ± 0.668 | 10.605 ± 0.021 | 2.000 | 8.605 ± 0.021 |

Scoring takes exactly two calls per draw, because no research task sets the raw
terminal-state flag, and the every-transition rule reads no fact. The remaining 65–91%
of these calls only fill provenance flags: the reach checks of each proposed goal at
s_t and in the earlier states, which this rule never reads. Read as the cost of
choosing HER's goals, the total column overstates it by up to an order of magnitude
(23.4 calls against 2 on Taxi). A draw at proportion zero, the HER agent's original-only replay control,
selects nothing: it costs the scoring pair and 1 + t provenance calls for the original
goal's flags.

The non-goal rule zeros already-reached shares, but does not remove all transitions
after an earlier reach. For unfiltered future:

| Configuration | Non-goal: after-end % ± SE | Non-goal: empty/fallback % ± SE | Counterfactual: empty/fallback % ± SE |
|---|---|---|---|
| maze4_c5_full | 11.384 ± 0.067 | 19.463 ± 0.103 | 27.710 ± 0.119 |
| maze4_c5_p4 | 11.293 ± 0.066 | 19.707 ± 0.076 | 27.916 ± 0.078 |
| handeye4_c50_full | 4.070 ± 0.135 | 77.819 ± 0.291 | 80.032 ± 0.254 |
| taxi_c200_full | 7.047 ± 0.401 | 70.678 ± 1.097 | 76.740 ± 0.883 |

Where these rules fall back, the pooled shares mix original-goal fallbacks with
relabeled draws. The composition of the relabeled draws alone, for unfiltered future,
with the pooled mean reward repeated for comparison; the two cost columns are per draw
over all draws, fallbacks included:

| Configuration | Rule | Relabeled % ± SE | After end % ± SE | Done % ± SE | Outside candidates % ± SE | Mean reward ± SE | Pooled mean reward ± SE | Scoring + selection/draw ± SE | Provenance/draw ± SE |
|---|---|---|---|---|---|---|---|---|---|
| maze4_c5_full | non-goal | 80.537 ± 0.103 | 14.135 ± 0.082 | 46.120 ± 0.040 | 0.616 ± 0.028 | 461.199 ± 0.404 | 371.435 ± 0.459 | 3.958 ± 0.002 | 2.579 ± 0.002 |
| maze4_c5_full | counterfactual | 72.290 ± 0.119 | 0 | 42.371 ± 0.037 | 0.602 ± 0.029 | 423.712 ± 0.374 | 306.294 ± 0.389 | 5.769 ± 0.007 | 1.099 ± 0.005 |
| maze4_c5_p4 | non-goal | 80.293 ± 0.076 | 14.064 ± 0.079 | 46.151 ± 0.049 | 82.748 ± 0.121 | 461.514 ± 0.491 | 370.562 ± 0.430 | 3.951 ± 0.001 | 2.571 ± 0.002 |
| maze4_c5_p4 | counterfactual | 72.084 ± 0.078 | 0 | 42.429 ± 0.038 | 82.102 ± 0.125 | 424.286 ± 0.381 | 305.837 ± 0.244 | 5.748 ± 0.004 | 1.103 ± 0.003 |
| handeye4_c50_full | non-goal | 22.181 ± 0.291 | 18.306 ± 0.495 | 12.134 ± 0.132 | 3.304 ± 0.334 | 121.344 ± 1.315 | 26.887 ± 0.371 | 3.581 ± 0.011 | 28.968 ± 0.130 |
| handeye4_c50_full | counterfactual | 19.968 ± 0.254 | 0 | 9.641 ± 0.101 | 3.153 ± 0.352 | 96.413 ± 1.013 | 19.225 ± 0.220 | 11.214 ± 0.185 | 21.859 ± 0.056 |
| taxi_c200_full | non-goal | 29.322 ± 1.097 | 23.857 ± 1.065 | 2.665 ± 0.063 | 69.902 ± 1.235 | 26.652 ± 0.633 | 7.808 ± 0.319 | 3.350 ± 0.013 | 96.751 ± 0.395 |
| taxi_c200_full | counterfactual | 23.260 ± 0.883 | 0 | 1.197 ± 0.023 | 72.784 ± 1.265 | 11.974 ± 0.231 | 2.767 ± 0.089 | 23.417 ± 0.916 | 83.383 ± 0.744 |

Both rules leave no relabeled draw already reached at its start. No fallback draw is
done in any of the 240 cells: the transition reaching the original goal always has
that goal among its admissible choices, so it never falls back. Pooled done
and mean reward are therefore close to the relabeled values times the relabeled share —
about a fifth of them on full HandEye4 under the counterfactual rule. On Taxi 70–73% of
the relabeled goals lie outside the candidates; the passenger in the taxi is the only
achieved goal that is never a desired one. Where fallbacks dominate, provenance
dominates the cost: 76.7% of Taxi's counterfactual draws scan the original goal's
history for its flags, 83.4 of 106.8 calls per draw.

The counterfactual rule zeros both problematic-goal shares for every configuration
and strategy; the candidate filter zeros outside-candidate shares. Neither changes
draw volume because empty distributions produce original fallback. Uniform real has
zero outside-candidate and empty-distribution shares under every rule: the valid
original goal remains a candidate. Under every-transition, its already-reached share
is 3.700 ± 0.001% on full Maze4, 6.203 ± 0.017% on full HandEye4, and
21.395 ± 0.175% on Taxi; restricting goals to real ones alone does not exclude
already-reached goals.

Outside **observed candidates** is different from outside the runner's fixed real
pool. In early episodes, even a full-pool task has achieved goals the agent has not
yet received as desired goals. This explains positive full-pool Maze/HandEye/BitFlipping
entries without leaking the pool into the store. Taxi additionally achieves passenger
location 4, in the taxi, which is never a desired delivery goal.

The earlier goal-conditioning comparison used epsilon-greedy ACS2 trajectories and five seeds, whereas
this probe uses uniformly random actions and twenty seeds. For future, Maze4 full
has 34.780 ± 0.085% already reached and 65.661 ± 0.050% done, versus 27–29% and
about 62% in that comparison; the restricted pool has 85.703 ± 0.103% outside candidates, versus
79–84% outside its fixed pool. Full HandEye4 has 86.393 ± 0.205% already reached
and 88.076 ± 0.191% done, versus 92–94% and 93–95%. Different action mixtures
and revisit/block-motion behavior alter achieved-goal multiplicities, and online
candidate discovery adds an early membership difference. Lower HandEye triviality is
consistent with more transport under random actions; this comparison does not identify
a causal learning mechanism or predict HER performance. The earlier probe's known storage-window
confound remains; it is not repaired retroactively by these composition numbers.

### Mutation and baseline acceptance evidence

Ten mutations were applied individually, compiled, and caused the named contract test
to fail: reward forced to one; done forced false; cap copied directly into truncation;
future including s_t; missing achieved goal at s_0; counterfactual using only current
non-goal state; different goal on next perception; an achieved goal inserted into
candidates; dropping fallback draws; and removing the negative-reward guard. Every
source file was restored and the unmutated tests passed afterwards. The harness and
individual failure logs are outside the checkout.

Reference comparison uses `af3ce49`: the pre-change `c06b92c` source is identical in
acs2-core, acs2-envs and acs2-measure, confirmed by an empty Git diff. Seed 42 on
Maze4 cap 5 with its four goals, full HandEye4 cap 50 and full Taxi cap 200, both
ACS2 and ACS2ER, at points 100/500/1000, produces **18 identical rows in 756 compared
top-level fields**, including complete nested value diagnostics and every population,
replay, update, match and success field. Only schema, optional diagnostics, build/host
metadata and wall times are excluded. No learning path changes.

The fix round applied the review's three mutations one at a time — the already-reached
and after-end shares swapped in `episode_composition`, the attempt probability reported
as the relabeled share, and the two counters swapped where the sampler records a draw
(now `RouteCounters::record`) — and sixteen of its own: the non-goal rule reading the
earlier states, original-route facts charged to selection, fallbacks counted on the
relabeled route, a route's done taken from the outside flag, composition selection
taken from provenance, payload purposes and probe routes swapped, a failed call keeping
its earlier draws or dropping their work, an empty relabeling ending as `Cut`, a route
mean left undivided or off by one, scoring without its reach call, and three checks
removed from the summary script. Each compiled and failed at least one named test; the
sources were restored and checked byte for byte.

Drawing is unchanged by the fix round. On stored random-policy history of five research
configurations (capacity 1,000, with eviction), 240 sampler configurations each — every
strategy, rule and filter, proportions 0, 0.3, 0.8 and 1, both truncation modes —
reproduce `360f6e5` exactly: 97,200 draws with their transitions, goals, samples,
provenance flags and total costs, the random-word position after each of 6,000 calls,
and 1,200 counter totals. 45,000 goal distributions and 26,280 episode compositions are
identical as well; only 38 empty non-goal relabelings change their end from `Cut` to
`NoAdmissibleTransition`. The re-run probe reproduces every original composition field bit for bit.
The baseline regression above, repeated at `cd950fc` against a fresh `360f6e5` build
and against the archived pre-fix measurement rows, gives **18 identical rows in 792 compared
fields**, schema 3 and the null replay diagnostics included.

Acceptance gates at `4f983bf` after the fix round: **243 Rust tests** (237 at the
initial storage/relabeling acceptance, 211 before it), including MPX reach regressions;
tools Python **47**, baseline oracles **5**, unchanged pilot tests **20**, composition
analysis tests **13** (6 at the initial storage/relabeling acceptance). The
benchmark builds and P9's first seven columns match all five
archived rows byte for byte. A release clean includes acs2-trajectory before Clippy;
there are **22 warnings in older files, zero in the trajectory code**. Build provenance also
watches the new trajectory source directory, so editing only that dependency cannot
leave a previously clean build label cached. Protected source,
configuration, checkpoint, multiplexer, archived reports and pilot files remain
unchanged.

Reproduce the composition and its summary without writing results into a checkout:

```bash
cargo run --release -p acs2-measure --example trajectory_composition -- --out <outside-checkout>/composition-raw.jsonl
python3 -B experiments/relabeling-composition/summarize.py <outside-checkout>/composition-raw.jsonl --out <outside-checkout>
```

The summary script reads schema-2 rows only; the original schema-1 output stays archived
outside the checkout.

## Start-goal pair holdout

The main maze protocol is **`maze-pairs-v1`**, registered and pushed before any
learning under it in `eef356cfe050f5e11502e7a1e9a4a229bda29de3`.
[`pair-protocol.json`](../experiments/maze-baselines/pair-protocol.json) contains
every pair; [`register_pairs.py`](../experiments/maze-baselines/register_pairs.py)
reconstructs the registration using only the reference geometry and exact
integer absorbing-state random-policy calculation. The reference geometry is
unmodified pyalcs commit `79d9f862e35385ba6fbad3a75415c96b9b116500`.

For each goal in row/column order, rank every other walkable start by shortest
eight-action distance, row and column. Split ranks into three balanced contiguous
strata, giving remainders to earlier strata. Within a stratum make
`max(1, floor((size+2)/5))` balanced contiguous segments. Segment endpoints enter
validation/test, with their order alternating by goal-index plus stratum-index
plus segment-index parity; interiors enter training. Sort starts within each
goal by row/column. No seed or learning result enters this rule.

Every goal must have at least three starts in each block, including one from
each distance-rank stratum and two distinct distances; every maze block must
have at least forty pairs. The disjoint union is all ordered distinct walkable
pairs, including the original `9` landmark. The rounded proportions are near
60/20/20; exact proportions and counts are registered explicitly.

| Maze | Cap | Training pairs | Validation pairs | Test pairs | Training/validation/test proportions |
|---|---:|---:|---:|---:|---|
| Maze4 | 5 | 378 | 162 | 162 | 7/13, 3/13, 3/13 |
| Maze5 | 8 | 888 | 222 | 222 | 2/3, 1/6, 1/6 |
| Maze6 | 10 | 888 | 222 | 222 | 2/3, 1/6, 1/6 |
| Maze7 | 10 | 828 | 216 | 216 | 23/35, 6/35, 6/35 |
| Woods14 | 18 | 228 | 57 | 57 | 2/3, 1/6, 1/6 |
| MazeMA | 9 | 120 | 45 | 45 | 4/7, 3/14, 3/14 |

Caps are unchanged from the goal registration. Maze4 has two explicit six-step
pairs beyond its legacy cap five: `(6,6)->(1,1)` and `(1,1)->(6,6)`, both in
validation. Its validation ceiling is 160/162; other blocks' ceilings are one.
Each block has its own exact random floor below 0.20. Test floors are respectively
0.07961, 0.07676, 0.10023, 0.09960, 0.13635 and 0.11893.

Training draws a uniform goal from all walkable cells, then a uniform start from
that goal's registered training starts. Every goal has the same number of starts
in a block, so training is also uniform over training pairs. Both agents share
this boundary and random-stream law. The agent receives all goal identifiers
and training resets; the measurement layer retains validation/test pair sets.
A held-out pair is an episode-start holdout: training trajectories may visit its
start and traverse transitions also present in evaluation. It does not hold out
transitions or guarantee generalization to an unseen goal or maze.

Evaluation is read-only and exhaustive, with exact equal pair weights in each
of the three blocks. Schema 5 adds goal-conditioned knowledge; schemas 2–4 remain
readable through [`grid/row_schema.py`](../experiments/grid/row_schema.py). The
full-pool learning regression remains eighteen rows against the archived oracle,
with every original learning/evaluation field preserved. Neither learner nor
core configuration, replay sample, checkpoint identity or codec changes.
The budget rule was registered before pair-protocol learning:
`2M * ceil(eight-action graph diameter / 10)` interactions, equal for both
agents: 2M on five mazes, 4M on Woods14. This is a measurement horizon and
makes no convergence claim.

## Goal holdout diagnostics

The unchanged auxiliary protocol **`maze-goals-v1`** is in
[`protocol.json`](../experiments/maze-baselines/protocol.json), preregistered in
`15ed86132b27cde4658f2581d95ead27f481ad68`. Its SHA-256 remains
`7baac478cbf6f7d4c2ab86d2784ce63c60399817fc22f8d20cee2bddcb9b8eff`.
It partitions goals by their integer distance sum, with every other walkable
cell eligible as a start. Training/validation/test goal counts are 15/6/6,
23/7/7, 23/7/7, 24/6/6, 13/3/3 and 9/3/3 in the six-maze order above.
There will be no twenty-seed grid for this protocol in the current scope.
Its earlier proposal is superseded and the submission tools reject that stage.

The diagnostic runs used clean measurement commit
`4eecd194869680e44797228623dd9983849a4551`: both agents, seeds 42–46,
checkpoints 1k/10k/100k/200k interactions, all six mazes and their registered
caps. At 200k, across-seed mean success was:

| Maze | ACS2 training/validation/test | ACS2ER training/validation/test | Test random floor |
|---|---|---|---:|
| Maze4 | 0.3503 / 0.0641 / 0.0397 | 0.3097 / 0.0410 / 0.1128 | 0.08577 |
| Maze5 | 0.2094 / 0.0317 / 0.0929 | 0.2184 / 0.0437 / 0.0810 | 0.08857 |
| Maze6 | 0.2447 / 0.0254 / 0.0635 | 0.2459 / 0.0675 / 0.0532 | 0.11197 |
| Maze7 | 0.2576 / 0.0352 / 0.0352 | 0.2321 / 0.0800 / 0.0486 | 0.09362 |
| Woods14 | 0.2581 / 0.1778 / 0.1259 | 0.1889 / 0.1741 / 0.0741 | 0.11584 |
| MazeMA | 0.3238 / 0.1000 / 0.1476 | 0.3317 / 0.1476 / 0.2048 | 0.14807 |

Transfer was weak and inconsistent compared with training improvement; these
five-seed point means are not a universal proof of absence of transfer. Some
held-out means exceed their floors, including MazeMA ACS2ER. The separate
seed-42 cluster cost pilot evaluated through 10M interactions at
1k/3k/10k/30k/100k/200k/300k/1M/2M/4M/5M/10M. At 10M its Maze4 ACS2 test
success was 0.0256 against floor 0.08577; ACS2ER was 0.1731. One seed cannot
establish a performance ceiling or an agent ranking.

On the non-learning composition probe, seeds 42–46 and 10k interactions per
maze, unfiltered achieved-goal strategies selected 31–46% of goals outside the
fixed training set. This would give HER access to held-out desired goals while
TU's fixed training-goal candidate set could not select them. Pair holdout
keeps all goals in training and removes that goal-exposure imbalance. Together
with poor transfer diagnostics, this motivated the preapproved alternative.
The original geometry, goal pools, local data and cost-pilot data remain archived.

## Goal-conditioned anticipation knowledge

Every maze measurement row reports `knowledge`, the share of `(transition, goal)`
cases correctly anticipated by at least one classifier with quality strictly
above `theta_r`, using the core's existing `anticipation_fraction`. Enumerate all
walkable transitions of the multi-goal maze, including departures from the
original landmark `9`, exclude wall bumps, join every walkable goal to both
sides, and exclude transitions starting at their own goal. The denominator is
transition-count times `(goal-count - 1)`; this secondary metric is shared across
blocks and is independent of the pair split and episode cap. Rows record it as
`knowledge_transition_goal_pairs`. Other task types and unevaluated control
runs use null until they provide a knowledge universe.

Knowledge runs outside training timers and match counters, consumes no random
stream, and takes immutable population access. Exhaustive populations score one,
empty populations and quality exactly at the threshold score zero, goal-specific
classifiers count only under their matching goal, and an independent geometry
and symbol enumeration checks a fixed mixed population. State comparisons cover
ACS2/ACS2ER populations, marks, numerosity, RNG, replay and training counters.

This is related to the predecessors' knowledge metric, but their reported
multi-goal values cannot be compared numerically without recomputation. Their
pyalcs matching used a truncated `zip` that ignored the goal portion of the
condition. This implementation matches the entire joined condition, including
both coordinate goal symbols, and measures the corrected goal-dependent model.

## Maze baseline execution and analysis

[`experiments/grid/`](../experiments/grid/) is the single shared execution path;
experiment-specific registrations and analysis stay in
[`maze-baselines/`](../experiments/maze-baselines/). Node guards verify clean pinned
source, musl binary hash, exact allocation identity, CPU model, preset and rows;
first-row and process watchdogs stop the phase on failure. Submission checks the
shared grant and test-only scheduling, counting pilot plus spent, reserved and
new hard limits against forty CPU-hours. Jobs use one CPU and 20–40 concurrency,
without host pinning, leases or laptop supervision. Raw data, accounting,
resolved plans, approvals and Polish reports stay outside Git.

The primary score is test success averaged equally over registered nominal
checkpoints within a seed, then across seeds 42–61. Paired ACS2ER-minus-ACS2
differences receive Student t(19) 95% intervals, critical value
2.093024054408263. Training and validation are diagnostic side blocks. The
threshold is the first registered nominal point where across-seed mean test
success reaches half its own ceiling, with no interpolation and null if absent.
If neither final lower 95% test bound exceeds its floor, retain the baseline for
successor methods; otherwise flag a maze as too easy for speed ranking if ACS2
already reaches 90% of its ceiling at the first point. Retain all configurations
and all three blocks regardless of this diagnostic verdict.

The pair-protocol local check used clean measurement commit
`38fcc300eee6210686a297ba3ad93b0a52e25d3d`, both agents and seeds 42–46 at
1k/10k/100k/200k interactions. At 200k all twelve test means exceeded their
own floors. The respective ACS2/ACS2ER test means were 0.2469/0.2235,
0.1775/0.1486, 0.1928/0.1739, 0.2306/0.1991, 0.2246/0.2281 and
0.2533/0.2000. Knowledge means ranged from 0.9885 to 1.0000. These sixty
short runs check measurement usefulness; they do not establish an agent
ranking. High anticipation coverage does not imply high goal-reaching success.

Before these results, `pair-pilot.json` fixed a 25% training/body cost-change
threshold for recalibration and the geometry-based 2M/4M horizons. Five-seed
local ACS2 cost per step increased 35.6% on Maze4 and 32.5% on Maze5 versus
the archived goal diagnostics. These laptop timing comparisons are coarse,
so a new seed-42 Lem pilot recalibrated the actual cluster costs rather than
reusing the old distribution's times. It used the grid checkpoints and the
same pinned measurement source; all twelve allocations completed and all
98 rows, identities and hashes passed checks. Its resolved plan SHA-256 is
`3ed1d2908d027ba1efa235d9dee3e99ff616be0c0274b0bb1dc440bfdd098d05`.
The retained musl binary SHA-256 is
`a9e5b6671100e56ec31e3b5502410f97f027e72acc8d7253c5e2cdab0601e0d7`.

The historical goal pilot spent 4,475 allocated CPU-seconds; the new pair
pilot spent 1,013, so both consume 5,488 seconds (1.5244 CPU-hours). Count
allocation `CPUTimeRAW` once, excluding duplicated batch/extern/step rows.
The new [`grid-plan.json`](../experiments/maze-baselines/grid-plan.json)
proposes 240 runs and 1,960 rows, in six arrays of forty single-CPU jobs,
concurrency forty, chained by `afterok`. Each array reserves its larger
agent limit: 7/8/10/10/5/3 minutes in the six-maze order. Limits use
`ceil((2.5 * seed42 body wall at budget + 90) / 60)`; the single-seed
forecast includes measured nonnegative allocation overhead.

The nominal grid forecast is 5.6294 CPU-hours; doubling it and adding both
pilots gives 12.7833 hours. The grid hard limits reserve 28.6667 hours,
or 30.1911 including both completed pilots, leaving 9.8089 hours below
forty. This is a forecast and allocation bound, not a performance conclusion.
The immutable plan bytes retain `grid_pending_approval`; the external approval
matching their SHA-256 authorized the completed twenty-seed pair grid.
`goal-grid-plan.json` records the superseded goal proposal. The composition
example retains its explicit `maze-goals-v1` diagnostic path and the archived
schema-2 full-pool path.

## Start-goal baseline results

The externally approved plan SHA-256 is
`913ce814484b742265ba88db09d72d4feaf8f981b200811454f9fce11b6b5a78`. All 240 runs and
1,960 schema-5 rows completed on Lem, with seeds 42–61 for both agents. Measurement
source `38fcc300eee6210686a297ba3ad93b0a52e25d3d` and musl binary
`a9e5b6671100e56ec31e3b5502410f97f027e72acc8d7253c5e2cdab0601e0d7` stayed pinned. Every
seed-42 learning field matches the new pair pilot (98 rows, 3,430 fields); all 960 raw
artifact hashes passed verification. No failures or retries occurred.

T/V/Test below denote the registered training, validation and test reset pairs; every
goal occurs in every block. Floors are exact random-policy references, rounded here;
ceilings are geometric reachability within the cap. The pair registration above
specifies the start law and beyond-cap pairs.

| Maze | Cap | Pairs T/V/Test | Floor T/V/Test | Ceiling T/V/Test | Budget |
| --- | --- | --- | --- | --- | --- |
| Maze4 | 5 | 378/162/162 | 0.09543/0.07237/0.07961 | 1.00000/0.98765/1.00000 | 2M |
| Maze5 | 8 | 888/222/222 | 0.08582/0.07695/0.07676 | 1.00000/1.00000/1.00000 | 2M |
| Maze6 | 10 | 888/222/222 | 0.10334/0.10225/0.10023 | 1.00000/1.00000/1.00000 | 2M |
| Maze7 | 10 | 828/216/216 | 0.10522/0.09140/0.09960 | 1.00000/1.00000/1.00000 | 2M |
| Woods14 | 18 | 228/57/57 | 0.12477/0.14139/0.13635 | 1.00000/1.00000/1.00000 | 4M |
| MazeMA | 9 | 120/45/45 | 0.13098/0.12651/0.11893 | 1.00000/1.00000/1.00000 | 2M |

Final test success is mean ± standard error across twenty seeds. The last column is mean
estimated training interactions per CPU-second, ACS2/ACS2ER, obtained separately within
each run from sampled CPU attribution.

| Maze | Final ACS2 test ± SE | Final ACS2ER test ± SE | Registered verdict | Training interactions/CPU-s A/ER |
| --- | --- | --- | --- | --- |
| Maze4 | 0.3253 ± 0.0094 | 0.2272 ± 0.0119 | compare | 81 315/21 670 |
| Maze5 | 0.2356 ± 0.0084 | 0.1604 ± 0.0096 | compare | 46 436/11 987 |
| Maze6 | 0.2849 ± 0.0105 | 0.1664 ± 0.0100 | compare | 44 998/10 985 |
| Maze7 | 0.2664 ± 0.0119 | 0.1799 ± 0.0103 | compare | 42 657/9 897 |
| Woods14 | 0.1974 ± 0.0143 | 0.1833 ± 0.0165 | compare | 311 886/71 098 |
| MazeMA | 0.2722 ± 0.0177 | 0.2144 ± 0.0181 | compare | 309 347/73 058 |

Allocation-only `sacct` CPUTimeRAW totals 19247 seconds (5.3464 CPU-hours) for the grid
and 24735 seconds (6.8708 hours) including both pilots; remaining reservations are zero.
The approved pre-submission bound was 30.1911 hours within forty. Process CPU (18521.01
seconds) and attributed training CPU (18357.43 seconds, estimated) are reported
separately, without adding them to allocation cost.

All final lower 95% test bounds exceed their own random floors; no configuration meets
the registered too-easy condition. No mean test curve reaches half its ceiling at a
registered point, so every threshold count is null. Individual primary paired intervals favor
ACS2 on Maze4–7 and contain zero on Woods14 and MazeMA. ACS2ER has higher
checkpoint-mean knowledge on every maze; final knowledge means are at least 0.9998 for
both agents. This separates anticipation coverage from goal-reaching policy success.

The registered primary comparison remains the equal-checkpoint mean within each seed on
Test, with paired ACS2ER-minus-ACS2 t(19) intervals. Final success is a separate
diagnostic for the registered verdict. Knowledge remains secondary. Full seed scores,
block curves, primary intervals, knowledge, costs and the Polish report remain outside
the checkout under the resolved `pair-grid/` result root; `maze-baselines/analyze.py`
reconstructs them with terminal accounting and approval checks. This finite 2M/4M
horizon makes no convergence claim.

## Hindsight copy updates and held-pair replay results

The approved registration is [`experiments/her-variants/plan.json`](../experiments/her-variants/plan.json), SHA-256
`e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444`.
The measurement clone stays at `a592a5bad8bbc13d62553cc09a72748182809879`;
the unchanged musl binary has SHA-256
`83ebf9a69d3d42cd8f275bab9f77f306fd25e5bcb7a665cf4771d169ab2ac3f2`.
Execution and analysis tools may evolve without rebuilding this measurement.

`acs2-agents` depends on `acs2-core` and `acs2-trajectory`; measurement constructs
its learners and implements their raw-episode environment port. Agents never
depend on measurement. `acs2er/episodic.rs` contains the original-only episodic
control, `acs2her/` contains the full, values and shortened values variants, and
`acs2er/one_update.rs` owns the diagnostic replay parameters beside ACS2ER's FIFO implementation.
Shared collection and update dispatch live in `episodic_replay.rs`; the full learning
step and replay sample live in `acs2-core::learning::replay`. Checkpoint identity and
codec, `Configuration`, `ReplaySample`, and multiplexer dynamics remain unchanged.

All episodic arms act epsilon-greedily with epsilon 0.8 on the episode's real
state joined with its real goal. They collect without learning, insert one
complete episode into a 10,000-step FIFO store, and then draw three times that
episode's length. Warmup requires at least one retained step after insertion;
skipped episodes are never repaid. Draws are uniform over retained transitions,
with replacement, so a one-step store supplies three updates. ALP/GA receive the
end-of-episode environment clock; replay never advances this physical clock.
Evaluation uses BestAction with its separate RNG and changes no learning state,
store, sampler stream, sidecar or counters.

The exact update rules are:

`acs2her::HindsightConfiguration` describes three independent choices with the fields
`copy_update`, `admissibility` and `relabeled_proportion`. `Preset::configuration()`
expands each named preset into those choices before constructing the shared episodic
learner. Full updates retain ALP, RL and optional GA; values-only updates retain only
the action-set RL application. Admissibility uses the existing trajectory enum.

The parameter type can express other combinations, which keeps the two choices usable
for trajectory utility. `acs2her::Agent::new` accepts only a named `Preset`, and the
measurement factory/CLI exposes only the three measured combinations below. Other
parameter combinations have no run name and are not measured. Keeping that boundary
prevents an unregistered combination from being written under a measured arm's name.

| Arm | Relabeling | Update of an original or fallback | Update of a relabeled copy |
|---|---|---|---|
| `acs2er_episodic` | p=0, same episodic path | Core ACS2ER replay step: ALP, RL, optional GA | None |
| `acs2her` | Future, every transition, fixed p=2/3 | Full step | Full step |
| `acs2her_values` | Future, every transition, selected p | Full step | Action-set RL only, changing r and ir |
| `acs2her_values_shortened` | Future, counterfactual episode admissibility, selected p | Full step | Action-set RL only |
| `her_values_p2of3` | Future, every transition, fixed p=2/3 | Full step | Action-set RL only |
| `acs2er_one_update` | Unchanged ACS2ER FIFO replay, one application per environment step | Full step | None |

Values copies bootstrap from the next state joined with the same copied goal.
They change no conditions, effects, marks, quality, numerosity, experience,
timestamps or population size. Copies equal to the original goal still belong
to the relabeled route. All research rewards and endings come from the task
objective: 1000/0 here, terminal goal reaches bootstrap zero, and time-limit
truncations bootstrap under the thesis preset. Sampler and learner truncation
modes must agree. The one-update observer forwards every random draw
unchanged and records selected original reward/done using a FIFO outcome
sidecar. It makes no additional random or objective calls, and its sidecar
logical bytes belong to replay memory. It has no new checkpoint interface.

Rows retain schema 5. The new arms add full and values-only application counts,
realized relabeled share, original/relabeled route composition, fallbacks,
objective cost by scoring/selection/provenance purpose, raw trajectory logical
bytes and replay memory. Counts are learning-step applications, including an
RL application whose action set may be empty. Full steps divided by actual
environment steps give observed model update volume. Every-transition values
copies have expected volume 3p per environment step, and full steps 3(1 - p);
shortening fallbacks increase the full-step volume. Actual
route counts, rather than requested p, determine the reported volume.

The private test-compiled HER oracle reproduces pyalcs copy rewards 0/1 and
`done=False` and invokes the full replay step. It has no research CLI mode.
Its 1000 single-step cases cover original and copied rewards/endings,
future/final/episode selection, expected/unexpected ALP, covering, marks,
reinforcement and bootstrap, comparing complete populations with P11's 1e-9
tolerance. No case increments numerosity or removes a classifier; all 3,406
post-step classifiers have num=1. Core subsumption is covered by the P3/P11 tests.
GA/PEE and overspecification are disabled; theta_i=0 excludes
the reference deletion-skip hazard. It cannot validate the reference's
exploration that ignored goals, exploitation learning, replay timing, retention
distribution or long-run growth. Defect contracts, actual RNG-stream checks,
independent match-formation counts, and the 60-row bit regression on all five
new arms and six mazes supplement it. The 98-row seed-42 baseline regression
preserves every learning, evaluation and knowledge field of the archived pair
grid. Those twenty-seed baselines are reused, never rerun for this study.

Tuning uses validation only, seeds 42--46 and the registered 2M/4M budgets and
points. For each values variant and maze separately, maximize checkpoint-mean
validation success within a seed, then the mean over five seeds. Scores within
1e-12 tie in favor of smaller p. The candidates are 1/3 and 2/3; 0.9 was removed
before tuning by the registered cost rule. The full variant is fixed at 2/3.
The main grid contains five arms, six mazes and seeds 42--61. If the selected
values p differs from 2/3, twenty additional fixed-p values runs provide the
update-rule ablation. Otherwise that ablation reuses twenty-seed, three-block
final-grid values rows. Validation-only tuning rows never enter test analysis.

Primary success is the test-block mean over all registered points within each
seed, then mean and SE over twenty seeds. Every registered difference pairs
those seed scores and uses a marginal 95% t(19) interval, critical value
2.093024054408263, without multiplicity correction. The tuned values-minus-full
comparison and the fixed-p values-minus-full update ablation have separate
interpretations. Knowledge curves and full applications per environment step
appear beside every values comparison in JSON and scientific SVG/PNG figures.

Both interaction thresholds use exact rational arithmetic: archived ACS2 final
mean test success and archived ACS2ER final mean test success. Success is an
integer count over each block's fixed pair count; means and >= comparisons use
Fraction. A crossing is the first registered nominal point, without
interpolation, or null. All 24 archived baseline crossings are independently
reproduced before analysis. This finite 2M/4M horizon makes no convergence claim.

Shared grid execution uses immutable operation snapshots outside the frozen
measurement clone, a shared STOP, node guards and one accounting scope over
all tuning, main-grid and control roots. Every submission checks the grant and
sbatch test-only result. Pilots, allocated CPUTimeRAW once per attempt, outstanding
hard reservations and every new limit must total at most sixty CPU-hours.
No lease, host pinning, laptop automation or new monitor is used. Fixed arms
reproduce seed-42 pilot rows; selected values reproduce five tuning seeds on
shared points, and p=2/3 additionally reproduces its pilot. Node identity
records are recalculated independently after pulling, before analysis. Raw
artifacts and the Polish report live outside Git under the approved study roots
in `~/tu_runs/5/` and `~/Desktop/tu-runs/5/`.

### Validation selections and control admission

The two candidates were evaluated on five validation seeds at full registered budgets.
All shortened cells selected 1/3. Every-transition values selected 1/3 except Woods14,
which selected 2/3. The following scores are means over points within each seed,
then over five seeds. No test rows participated in selection.

| Maze | Values scores: 1/3, 2/3 | Selected p | Shortened scores: 1/3, 2/3 | Selected p |
|---|---|---|---|---|
| maze4 | 0.148148, 0.119136 | 1/3 | 0.141821, 0.130864 | 1/3 |
| maze5 | 0.111824, 0.094820 | 1/3 | 0.126239, 0.110360 | 1/3 |
| maze6 | 0.115878, 0.088401 | 1/3 | 0.143694, 0.120946 | 1/3 |
| maze7 | 0.109375, 0.091551 | 1/3 | 0.123495, 0.113542 | 1/3 |
| woods14 | 0.128655, 0.138012 | 2/3 | 0.201170, 0.191033 | 1/3 |
| mazema | 0.199444, 0.166667 | 1/3 | 0.237222, 0.197222 | 1/3 |

Five fixed-p control cells were admitted immediately after selection, under the shared
budget. Woods14 reused its twenty final-grid values seeds and all three blocks; it
allocated no extra runs. Admission decisions are durable in `control-admission.jsonl`.

| Control cell | New hard limit, CPU-s | Spent before admission | Reserved before admission | Bound including pilots and new limit, CPU-h |
|---|---:|---:|---:|---:|
| maze4 | 4800 | 18359 | 0 | 7.9456 |
| maze5 | 7200 | 18359 | 4800 | 9.9456 |
| maze6 | 8400 | 18359 | 12000 | 12.2789 |
| maze7 | 8400 | 18359 | 20400 | 14.6122 |
| mazema | 2400 | 18359 | 28800 | 15.2789 |

### Test success and paired differences

Each cell below is primary mean ± SE over twenty seeds, 42--61. The mean uses all
eight registered points, or nine on Woods14; it is not final-point success. Budgets
remain 2M interactions, 4M on Woods14. Caps are 5, 8, 10, 10, 18 and 9 respectively.
Training, validation and test pair blocks retain the archived protocol and hashes.

| Maze | ACS2 | ACS2ER | ACS2ER episodic | ACS2HER | ACS2HER values | ACS2HER values shortened | ACS2ER one update | Values p=2/3 |
|---|---|---|---|---|---|---|---|---|
| maze4 | 0.1934 ± 0.0038 | 0.1665 ± 0.0050 | 0.1726 ± 0.0045 | 0.1847 ± 0.0041 | 0.1576 ± 0.0036 | 0.1672 ± 0.0049 | 0.1694 ± 0.0045 | 0.1390 ± 0.0043 |
| maze5 | 0.1313 ± 0.0028 | 0.1161 ± 0.0033 | 0.1110 ± 0.0035 | 0.1405 ± 0.0029 | 0.1111 ± 0.0020 | 0.1325 ± 0.0026 | 0.1145 ± 0.0031 | 0.0981 ± 0.0021 |
| maze6 | 0.1570 ± 0.0035 | 0.1288 ± 0.0037 | 0.1336 ± 0.0037 | 0.1565 ± 0.0024 | 0.1213 ± 0.0024 | 0.1365 ± 0.0027 | 0.1346 ± 0.0028 | 0.1018 ± 0.0029 |
| maze7 | 0.1477 ± 0.0036 | 0.1371 ± 0.0033 | 0.1380 ± 0.0035 | 0.1575 ± 0.0031 | 0.1244 ± 0.0030 | 0.1450 ± 0.0023 | 0.1294 ± 0.0036 | 0.0974 ± 0.0024 |
| woods14 | 0.1848 ± 0.0063 | 0.1729 ± 0.0081 | 0.1669 ± 0.0053 | 0.1649 ± 0.0053 | 0.1360 ± 0.0036 | 0.1871 ± 0.0054 | 0.1735 ± 0.0055 | 0.1360 ± 0.0036 |
| mazema | 0.1874 ± 0.0065 | 0.1910 ± 0.0071 | 0.1974 ± 0.0071 | 0.1990 ± 0.0083 | 0.1757 ± 0.0076 | 0.1958 ± 0.0090 | 0.1835 ± 0.0061 | 0.1611 ± 0.0029 |

All ten registered paired contrasts follow, including the separate fixed-p ablation.
Intervals are marginal and carry no multiplicity correction. Each values contrast in
`analysis/results.json` also contains both arms' full-update and knowledge summaries
and their complete point curves. The six `*-values-comparisons.svg` figures put its
primary interval, knowledge curve and observed update-volume curve in the same row.

**maze4**, twenty paired seeds:

| Left − right | Difference | SE | Marginal 95% t(19) interval |
|---|---:|---:|---|
| ACS2HER − ACS2ER episodic | +0.0121 | 0.0059 | [-0.0002, +0.0245] |
| ACS2HER values − ACS2ER episodic | -0.0150 | 0.0049 | [-0.0253, -0.0047] |
| ACS2HER values shortened − ACS2ER episodic | -0.0054 | 0.0064 | [-0.0189, +0.0081] |
| ACS2HER values − ACS2HER | -0.0271 | 0.0059 | [-0.0393, -0.0148] |
| ACS2HER values shortened − ACS2HER values | +0.0095 | 0.0042 | [+0.0008, +0.0183] |
| ACS2ER episodic − ACS2ER | +0.0061 | 0.0059 | [-0.0063, +0.0185] |
| ACS2ER episodic − ACS2 | -0.0208 | 0.0049 | [-0.0311, -0.0105] |
| ACS2ER one update − ACS2ER | +0.0029 | 0.0068 | [-0.0114, +0.0172] |
| ACS2ER one update − ACS2 | -0.0240 | 0.0068 | [-0.0382, -0.0098] |
| Values p=2/3 − ACS2HER | -0.0458 | 0.0067 | [-0.0598, -0.0317] |

**maze5**, twenty paired seeds:

| Left − right | Difference | SE | Marginal 95% t(19) interval |
|---|---:|---:|---|
| ACS2HER − ACS2ER episodic | +0.0295 | 0.0047 | [+0.0197, +0.0394] |
| ACS2HER values − ACS2ER episodic | +0.0001 | 0.0049 | [-0.0101, +0.0103] |
| ACS2HER values shortened − ACS2ER episodic | +0.0215 | 0.0041 | [+0.0130, +0.0300] |
| ACS2HER values − ACS2HER | -0.0294 | 0.0036 | [-0.0369, -0.0220] |
| ACS2HER values shortened − ACS2HER values | +0.0214 | 0.0036 | [+0.0138, +0.0290] |
| ACS2ER episodic − ACS2ER | -0.0051 | 0.0052 | [-0.0160, +0.0058] |
| ACS2ER episodic − ACS2 | -0.0203 | 0.0049 | [-0.0306, -0.0100] |
| ACS2ER one update − ACS2ER | -0.0016 | 0.0051 | [-0.0122, +0.0090] |
| ACS2ER one update − ACS2 | -0.0168 | 0.0039 | [-0.0250, -0.0085] |
| Values p=2/3 − ACS2HER | -0.0425 | 0.0036 | [-0.0501, -0.0349] |

**maze6**, twenty paired seeds:

| Left − right | Difference | SE | Marginal 95% t(19) interval |
|---|---:|---:|---|
| ACS2HER − ACS2ER episodic | +0.0229 | 0.0048 | [+0.0127, +0.0330] |
| ACS2HER values − ACS2ER episodic | -0.0124 | 0.0042 | [-0.0211, -0.0036] |
| ACS2HER values shortened − ACS2ER episodic | +0.0029 | 0.0044 | [-0.0062, +0.0121] |
| ACS2HER values − ACS2HER | -0.0352 | 0.0033 | [-0.0423, -0.0282] |
| ACS2HER values shortened − ACS2HER values | +0.0153 | 0.0040 | [+0.0070, +0.0236] |
| ACS2ER episodic − ACS2ER | +0.0048 | 0.0060 | [-0.0077, +0.0174] |
| ACS2ER episodic − ACS2 | -0.0233 | 0.0047 | [-0.0331, -0.0136] |
| ACS2ER one update − ACS2ER | +0.0058 | 0.0047 | [-0.0040, +0.0155] |
| ACS2ER one update − ACS2 | -0.0224 | 0.0044 | [-0.0316, -0.0132] |
| Values p=2/3 − ACS2HER | -0.0547 | 0.0042 | [-0.0635, -0.0459] |

**maze7**, twenty paired seeds:

| Left − right | Difference | SE | Marginal 95% t(19) interval |
|---|---:|---:|---|
| ACS2HER − ACS2ER episodic | +0.0195 | 0.0051 | [+0.0088, +0.0302] |
| ACS2HER values − ACS2ER episodic | -0.0137 | 0.0049 | [-0.0240, -0.0034] |
| ACS2HER values shortened − ACS2ER episodic | +0.0070 | 0.0048 | [-0.0030, +0.0169] |
| ACS2HER values − ACS2HER | -0.0332 | 0.0037 | [-0.0409, -0.0254] |
| ACS2HER values shortened − ACS2HER values | +0.0206 | 0.0039 | [+0.0126, +0.0287] |
| ACS2ER episodic − ACS2ER | +0.0009 | 0.0035 | [-0.0064, +0.0082] |
| ACS2ER episodic − ACS2 | -0.0096 | 0.0043 | [-0.0186, -0.0007] |
| ACS2ER one update − ACS2ER | -0.0077 | 0.0040 | [-0.0161, +0.0007] |
| ACS2ER one update − ACS2 | -0.0183 | 0.0043 | [-0.0273, -0.0092] |
| Values p=2/3 − ACS2HER | -0.0601 | 0.0030 | [-0.0663, -0.0539] |

**woods14**, twenty paired seeds:

| Left − right | Difference | SE | Marginal 95% t(19) interval |
|---|---:|---:|---|
| ACS2HER − ACS2ER episodic | -0.0019 | 0.0064 | [-0.0154, +0.0115] |
| ACS2HER values − ACS2ER episodic | -0.0309 | 0.0065 | [-0.0444, -0.0174] |
| ACS2HER values shortened − ACS2ER episodic | +0.0203 | 0.0070 | [+0.0056, +0.0350] |
| ACS2HER values − ACS2HER | -0.0289 | 0.0067 | [-0.0430, -0.0149] |
| ACS2HER values shortened − ACS2HER values | +0.0512 | 0.0064 | [+0.0378, +0.0646] |
| ACS2ER episodic − ACS2ER | -0.0060 | 0.0089 | [-0.0247, +0.0126] |
| ACS2ER episodic − ACS2 | -0.0179 | 0.0080 | [-0.0346, -0.0013] |
| ACS2ER one update − ACS2ER | +0.0006 | 0.0093 | [-0.0189, +0.0201] |
| ACS2ER one update − ACS2 | -0.0113 | 0.0091 | [-0.0304, +0.0078] |
| Values p=2/3 − ACS2HER | -0.0289 | 0.0067 | [-0.0430, -0.0149] |

**mazema**, twenty paired seeds:

| Left − right | Difference | SE | Marginal 95% t(19) interval |
|---|---:|---:|---|
| ACS2HER − ACS2ER episodic | +0.0017 | 0.0117 | [-0.0228, +0.0261] |
| ACS2HER values − ACS2ER episodic | -0.0217 | 0.0106 | [-0.0438, +0.0005] |
| ACS2HER values shortened − ACS2ER episodic | -0.0015 | 0.0120 | [-0.0267, +0.0236] |
| ACS2HER values − ACS2HER | -0.0233 | 0.0120 | [-0.0485, +0.0019] |
| ACS2HER values shortened − ACS2HER values | +0.0201 | 0.0132 | [-0.0075, +0.0478] |
| ACS2ER episodic − ACS2ER | +0.0064 | 0.0111 | [-0.0169, +0.0297] |
| ACS2ER episodic − ACS2 | +0.0100 | 0.0096 | [-0.0100, +0.0300] |
| ACS2ER one update − ACS2ER | -0.0075 | 0.0069 | [-0.0219, +0.0069] |
| ACS2ER one update − ACS2 | -0.0039 | 0.0092 | [-0.0231, +0.0153] |
| Values p=2/3 − ACS2HER | -0.0379 | 0.0088 | [-0.0564, -0.0195] |

ACS2HER improves primary success over ACS2ER episodic on Maze5--7 with intervals above
zero; Maze4, Woods14 and MazeMA remain uncertain. Fixed-p values is below full on all
six mazes. The comparison changes model update volume from three to approximately
one full application per interaction, with slower early knowledge acquisition; it
does not identify the value estimator's quality separately from model learning dose.
Shortening improves the tuned values result on Maze4--7 and Woods14; MazeMA is
uncertain. The Woods14 shortening contrast also changes selected p from 2/3 to 1/3.
ACS2HER values shortened improves over ACS2ER episodic on Maze5 and Woods14. Neither episodic
ER nor ACS2ER one update has a positive interval excluding zero against archived ACS2ER on any maze.

### Exact interaction thresholds

Registered thresholds are fractions reconstructed from archived final test counts.
Both archived baselines and all six measured arms are evaluated against both thresholds.

| Maze | ACS2 final mean threshold | ACS2ER final mean threshold |
|---|---|---|
| maze4 | 527/1620 | 92/405 |
| maze5 | 523/2220 | 89/555 |
| maze6 | 253/888 | 739/4440 |
| maze7 | 1151/4320 | 259/1440 |
| woods14 | 15/76 | 11/60 |
| mazema | 49/180 | 193/900 |

Each cell below gives the first nominal point for **ACS2 threshold / ACS2ER threshold**,
as exact integer interactions. `null` is no crossing at a registered point; it is not
a claim that a threshold is unreachable. The 24 archived-baseline crossings match the
registration exactly. No interpolation or rounded-float comparison is used.

| Maze | ACS2 | ACS2ER | ACS2ER episodic | ACS2HER | ACS2HER values | ACS2HER values shortened | ACS2ER one update | Values p=2/3 |
|---|---|---|---|---|---|---|---|---|
| maze4 | 2000000 / 300000 | null / 2000000 | null / 300000 | null / 30000 | null / null | null / null | null / 300000 | null / null |
| maze5 | 1000000 / 300000 | null / 1000000 | null / null | null / 30000 | null / null | null / 300000 | null / 1000000 | null / null |
| maze6 | 1000000 / 300000 | null / 100000 | null / 300000 | null / 30000 | null / null | null / 300000 | null / 300000 | null / null |
| maze7 | 2000000 / 300000 | null / 100000 | null / 300000 | null / 30000 | null / 1000000 | null / 100000 | null / 1000000 | null / null |
| woods14 | 100000 / 100000 | 30000 / 30000 | 4000000 / 100000 | null / 300000 | null / null | 10000 / 10000 | 30000 / 30000 | null / null |
| mazema | 2000000 / 100000 | null / 30000 | null / 30000 | null / 10000 | null / 300000 | null / 30000 | null / 100000 | null / null |

### Realized copies, model updates and knowledge

The following means ± SE use the same checkpoint averaging and twenty seeds. Full
applications divide cumulative full-step counts by actual physical interactions,
including episode overshoot. Values applications fill the remaining replay budget.
Original and fallback routes receive the full step. Total replay volume is three for
episodic arms, one for ACS2ER one update; archived online ACS2 has one full application per step.

| Maze | Arm | Realized copy share | Full applications / interaction | Knowledge |
|---|---|---|---|---|
| maze4 | ACS2 | — | 1.0000 ± 0.0000 | 0.7619 ± 0.0013 |
| maze4 | ACS2ER | — | 2.9994 ± 0.0000 | 0.8878 ± 0.0022 |
| maze4 | ACS2ER episodic | 0.0000 ± 0.0000 | 3.0000 ± 0.0000 | 0.8892 ± 0.0013 |
| maze4 | ACS2HER | 0.6672 ± 0.0003 | 3.0000 ± 0.0000 | 0.8818 ± 0.0013 |
| maze4 | ACS2HER values | 0.3333 ± 0.0003 | 2.0001 ± 0.0010 | 0.8465 ± 0.0018 |
| maze4 | ACS2HER values shortened | 0.2587 ± 0.0007 | 2.2240 ± 0.0021 | 0.8537 ± 0.0023 |
| maze4 | ACS2ER one update | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7674 ± 0.0017 |
| maze4 | Values p=2/3 | 0.6673 ± 0.0004 | 0.9980 ± 0.0011 | 0.7674 ± 0.0017 |
| maze5 | ACS2 | — | 1.0000 ± 0.0000 | 0.7225 ± 0.0013 |
| maze5 | ACS2ER | — | 2.9994 ± 0.0000 | 0.8486 ± 0.0019 |
| maze5 | ACS2ER episodic | 0.0000 ± 0.0000 | 3.0000 ± 0.0000 | 0.8458 ± 0.0013 |
| maze5 | ACS2HER | 0.6665 ± 0.0004 | 3.0000 ± 0.0000 | 0.8441 ± 0.0012 |
| maze5 | ACS2HER values | 0.3343 ± 0.0003 | 1.9970 ± 0.0010 | 0.7997 ± 0.0018 |
| maze5 | ACS2HER values shortened | 0.2602 ± 0.0009 | 2.2195 ± 0.0026 | 0.8115 ± 0.0022 |
| maze5 | ACS2ER one update | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7303 ± 0.0018 |
| maze5 | Values p=2/3 | 0.6673 ± 0.0004 | 0.9980 ± 0.0012 | 0.7208 ± 0.0020 |
| maze6 | ACS2 | — | 1.0000 ± 0.0000 | 0.7194 ± 0.0019 |
| maze6 | ACS2ER | — | 2.9994 ± 0.0000 | 0.8453 ± 0.0017 |
| maze6 | ACS2ER episodic | 0.0000 ± 0.0000 | 3.0000 ± 0.0000 | 0.8439 ± 0.0019 |
| maze6 | ACS2HER | 0.6674 ± 0.0004 | 3.0000 ± 0.0000 | 0.8455 ± 0.0021 |
| maze6 | ACS2HER values | 0.3333 ± 0.0004 | 2.0000 ± 0.0013 | 0.7998 ± 0.0022 |
| maze6 | ACS2HER values shortened | 0.2648 ± 0.0010 | 2.2055 ± 0.0031 | 0.8093 ± 0.0021 |
| maze6 | ACS2ER one update | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7235 ± 0.0022 |
| maze6 | Values p=2/3 | 0.6668 ± 0.0003 | 0.9995 ± 0.0010 | 0.7181 ± 0.0023 |
| maze7 | ACS2 | — | 1.0000 ± 0.0000 | 0.7160 ± 0.0015 |
| maze7 | ACS2ER | — | 2.9994 ± 0.0000 | 0.8445 ± 0.0018 |
| maze7 | ACS2ER episodic | 0.0000 ± 0.0000 | 3.0000 ± 0.0000 | 0.8449 ± 0.0018 |
| maze7 | ACS2HER | 0.6672 ± 0.0004 | 3.0000 ± 0.0000 | 0.8426 ± 0.0021 |
| maze7 | ACS2HER values | 0.3340 ± 0.0005 | 1.9979 ± 0.0016 | 0.7937 ± 0.0021 |
| maze7 | ACS2HER values shortened | 0.2625 ± 0.0008 | 2.2124 ± 0.0025 | 0.8055 ± 0.0017 |
| maze7 | ACS2ER one update | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7254 ± 0.0020 |
| maze7 | Values p=2/3 | 0.6669 ± 0.0005 | 0.9994 ± 0.0014 | 0.7162 ± 0.0019 |
| woods14 | ACS2 | — | 1.0000 ± 0.0000 | 0.8577 ± 0.0018 |
| woods14 | ACS2ER | — | 2.9995 ± 0.0000 | 0.9548 ± 0.0018 |
| woods14 | ACS2ER episodic | 0.0000 ± 0.0000 | 3.0000 ± 0.0000 | 0.9574 ± 0.0021 |
| woods14 | ACS2HER | 0.6679 ± 0.0004 | 3.0000 ± 0.0000 | 0.9497 ± 0.0022 |
| woods14 | ACS2HER values | 0.6666 ± 0.0004 | 1.0001 ± 0.0011 | 0.8543 ± 0.0019 |
| woods14 | ACS2HER values shortened | 0.2076 ± 0.0011 | 2.3772 ± 0.0032 | 0.9327 ± 0.0027 |
| woods14 | ACS2ER one update | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8636 ± 0.0017 |
| woods14 | Values p=2/3 | 0.6666 ± 0.0004 | 1.0001 ± 0.0011 | 0.8543 ± 0.0019 |
| mazema | ACS2 | — | 1.0000 ± 0.0000 | 0.8707 ± 0.0028 |
| mazema | ACS2ER | — | 2.9994 ± 0.0000 | 0.9704 ± 0.0023 |
| mazema | ACS2ER episodic | 0.0000 ± 0.0000 | 3.0000 ± 0.0000 | 0.9681 ± 0.0021 |
| mazema | ACS2HER | 0.6672 ± 0.0004 | 3.0000 ± 0.0000 | 0.9637 ± 0.0020 |
| mazema | ACS2HER values | 0.3339 ± 0.0004 | 1.9984 ± 0.0011 | 0.9361 ± 0.0027 |
| mazema | ACS2HER values shortened | 0.2194 ± 0.0012 | 2.3419 ± 0.0037 | 0.9497 ± 0.0016 |
| mazema | ACS2ER one update | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8741 ± 0.0026 |
| mazema | Values p=2/3 | 0.6673 ± 0.0004 | 0.9980 ± 0.0013 | 0.8705 ± 0.0024 |

Counterfactual admissibility removes copies whose goals were already reached at the
transition start or whose transitions follow an earlier goal reach. Consequently its
realized share is 0.208--0.265 despite requested p=1/3, and fallback originals raise
full-step volume. Every-transition copies have nonzero already-reached and
after-counterfactual-end composition; the sampler reports them rather than silently
changing semantics. Route counters, done and reward composition and their original
counterparts are retained at every point in the raw rows and analysis JSON.

### Objective cost, population, memory and time

Purpose costs below are objective-query counts per replay draw, averaged over points
and twenty seeds. Scoring includes one reward and one reach query. Selection and
provenance are reach queries; ACS2ER one update forwards recorded outcomes and makes none. Logical
memory is a lower bound independent of process peak RSS. Population is the final
point mean ± SE; CPU is mean process CPU seconds over the twenty seeds. Woods14 fixed-p
cost displays reused main-grid work, not an additional allocation.

| Maze | Arm | Scoring / selection / provenance per draw | Final population | Final logical MiB | Mean process CPU-s | Mean peak RSS MiB |
|---|---|---|---|---:|---:|---:|
| maze4 | ACS2ER episodic | 2.000 / 0.000 / 2.950 | 925.6 ± 15.5 | 0.743 | 107.2 | 8.750 |
| maze4 | ACS2HER | 2.000 / 0.000 / 3.752 | 924.3 ± 19.6 | 0.744 | 109.6 | 8.758 |
| maze4 | ACS2HER values | 2.000 / 0.000 / 3.359 | 1013.5 ± 17.8 | 0.775 | 115.1 | 8.706 |
| maze4 | ACS2HER values shortened | 2.000 / 1.399 / 2.275 | 1030.2 ± 22.6 | 0.781 | 116.2 | 8.736 |
| maze4 | ACS2ER one update | 0.000 / 0.000 / 0.000 | 1126.5 ± 22.8 | 1.159 | 55.5 | 8.750 |
| maze4 | Values p=2/3 | 2.000 / 0.000 / 3.745 | 1116.8 ± 22.9 | 0.811 | 121.3 | 8.727 |
| maze5 | ACS2ER episodic | 2.000 / 0.000 / 4.427 | 1376.3 ± 24.1 | 0.859 | 183.3 | 8.805 |
| maze5 | ACS2HER | 2.000 / 0.000 / 6.083 | 1478.0 ± 27.9 | 0.896 | 218.1 | 8.832 |
| maze5 | ACS2HER values | 2.000 / 0.000 / 5.299 | 1538.2 ± 27.2 | 0.916 | 200.0 | 8.799 |
| maze5 | ACS2HER values shortened | 2.000 / 2.365 / 3.428 | 1495.8 ± 24.7 | 0.901 | 196.8 | 8.820 |
| maze5 | ACS2ER one update | 0.000 / 0.000 / 0.000 | 1769.3 ± 34.4 | 1.384 | 106.0 | 8.816 |
| maze5 | Values p=2/3 | 2.000 / 0.000 / 6.102 | 1759.7 ± 32.9 | 0.994 | 224.7 | 8.810 |
| maze6 | ACS2ER episodic | 2.000 / 0.000 / 5.390 | 1512.0 ± 30.1 | 0.893 | 204.1 | 8.812 |
| maze6 | ACS2HER | 2.000 / 0.000 / 8.235 | 1625.1 ± 31.1 | 0.934 | 240.6 | 8.828 |
| maze6 | ACS2HER values | 2.000 / 0.000 / 6.853 | 1669.2 ± 26.0 | 0.949 | 220.9 | 8.819 |
| maze6 | ACS2HER values shortened | 2.000 / 3.279 / 4.155 | 1653.8 ± 33.1 | 0.943 | 226.5 | 8.809 |
| maze6 | ACS2ER one update | 0.000 / 0.000 / 0.000 | 1845.2 ± 31.0 | 1.411 | 112.3 | 8.810 |
| maze6 | Values p=2/3 | 2.000 / 0.000 / 8.144 | 1843.0 ± 35.0 | 1.010 | 232.2 | 8.803 |
| maze7 | ACS2ER episodic | 2.000 / 0.000 / 5.392 | 1607.5 ± 34.8 | 0.927 | 224.1 | 8.815 |
| maze7 | ACS2HER | 2.000 / 0.000 / 8.040 | 1790.3 ± 35.2 | 0.992 | 272.3 | 8.788 |
| maze7 | ACS2HER values | 2.000 / 0.000 / 6.723 | 1905.2 ± 26.4 | 1.032 | 259.8 | 8.802 |
| maze7 | ACS2HER values shortened | 2.000 / 3.213 / 4.165 | 1721.8 ± 27.0 | 0.968 | 232.8 | 8.774 |
| maze7 | ACS2ER one update | 0.000 / 0.000 / 0.000 | 2014.8 ± 35.0 | 1.470 | 123.4 | 8.800 |
| maze7 | Values p=2/3 | 2.000 / 0.000 / 7.946 | 1949.8 ± 31.2 | 1.048 | 246.3 | 8.800 |
| woods14 | ACS2ER episodic | 2.000 / 0.000 / 9.211 | 255.4 ± 8.3 | 0.428 | 62.5 | 8.694 |
| woods14 | ACS2HER | 2.000 / 0.000 / 9.869 | 276.8 ± 5.7 | 0.436 | 73.0 | 8.690 |
| woods14 | ACS2HER values | 2.000 / 0.000 / 9.730 | 327.9 ± 12.4 | 0.453 | 70.0 | 8.701 |
| woods14 | ACS2HER values shortened | 2.000 / 3.632 / 7.769 | 271.4 ± 10.5 | 0.434 | 65.7 | 8.688 |
| woods14 | ACS2ER one update | 0.000 / 0.000 / 0.000 | 305.4 ± 8.9 | 0.870 | 29.2 | 8.681 |
| woods14 | Values p=2/3 | 2.000 / 0.000 / 9.730 | 327.9 ± 12.4 | 0.453 | 70.0 | 8.701 |
| mazema | ACS2ER episodic | 2.000 / 0.000 / 4.843 | 243.1 ± 10.3 | 0.457 | 33.4 | 8.650 |
| mazema | ACS2HER | 2.000 / 0.000 / 5.258 | 255.6 ± 9.6 | 0.461 | 37.4 | 8.654 |
| mazema | ACS2HER values | 2.000 / 0.000 / 5.087 | 271.4 ± 6.3 | 0.467 | 34.9 | 8.671 |
| mazema | ACS2HER values shortened | 2.000 / 1.883 / 3.996 | 266.2 ± 6.3 | 0.465 | 35.6 | 8.667 |
| mazema | ACS2ER one update | 0.000 / 0.000 / 0.000 | 286.6 ± 6.2 | 0.864 | 14.3 | 8.687 |
| mazema | Values p=2/3 | 2.000 / 0.000 / 5.280 | 279.4 ± 7.3 | 0.470 | 33.7 | 8.671 |

The largest population observed at registered points was 4164 (ACS2ER one update, Maze7), and
ACS2HER reached 4163. Maximum process peak RSS was 9048 KiB, below the 512 MiB node limit.
These are observations at finite budgets, not limits on future population growth.
Every point also reports train/evaluation wall time. Process CPU and training CPU
estimates are separate from allocated CPUTimeRAW used for the budget.

### Completed execution and artifact verification

| Batch | Runs | Rows | Allocated CPU-s | CPU-h |
|---|---:|---:|---:|---:|
| tuning | 120 | 980 | 18359 | 5.0997 |
| main-grid | 600 | 4900 | 82720 | 22.9778 |
| control-maze4_c5_pairs | 20 | 160 | 2887 | 0.8019 |
| control-maze5_c8_pairs | 20 | 160 | 4627 | 1.2853 |
| control-maze6_c10_pairs | 20 | 160 | 4772 | 1.3256 |
| control-maze7_c10_pairs | 20 | 160 | 5066 | 1.4072 |
| control-mazema_c9_pairs | 20 | 160 | 729 | 0.2025 |
| Earlier pilots and baseline identity regression | 48 | 392 | 5445 | 1.5125 |

All 820 new attempts across 41 arrays completed with exit code zero: 120 tuning,
600 main-grid and 100 matched controls, with 6680 physical rows. There were no
computational failures, retries, identity mismatches or STOP. Two test-only refusals
allocated no CPU and paused submission of the remaining suffix. The first refusal
did not preserve stderr; the later event recorded `AssocMaxSubmitJobLimit`. Durable
resume checks submitted only previously unsubmitted entries after queue space returned.
The initial combined main-grid admission bound was 59.2789 CPU-hours, including all
pilots, tuning spent, matched-control reservations and new main-grid hard limits.
Final allocation cost including earlier pilots is **124605 s = 34.6125 CPU-hours**,
with zero outstanding reservations. Accounting counts each allocation once, not its
batch/extern steps again. The inherited baselines are neither reallocated nor charged.

The two raw copies passed all 10040 SHA-256 checks, and terminal accounting plus every
required identity record was independently checked on the cluster and after pulling.
The analysis entry point reproduced selections, derived manifests, matched-control
reuse and exact archived crossings before accepting results. Final study roots are
`~/tu_runs/5/approved-her-study/` and `~/Desktop/tu-runs/5/approved-her-study/`; numerical
results, primary/paired CSV tables and scientific figures live in `analysis/`. The
Polish execution report stays there outside Git. `raw-files.sha256.json` records the
verified raw set; the final copy manifest also covers analysis and gate evidence.

Execution and analysis are tested in `experiments/grid/` and `experiments/her-variants/`.
Acceptance has 282 Rust tests; Python gate counts are 47/7/20/13/23/13/33 for tools,
random policy, baseline pilot, relabeling composition, grid, maze baselines and HER
variants. P9 columns 1--7 match all five committed benchmark rows. All six workspace
packages were cleaned before Clippy: the same 22 older warnings, none in new code.
At study acceptance, all workspace `src` trees, Cargo manifests and the lockfile were
byte-identical to the frozen measurement commit; later validation tests and tooling did
not rebuild that executable. The subsequent agent-layout move changes source locations,
module declarations, dependencies and imports while preserving the measured behaviour.
The fixed research horizon and marginal intervals support the stated comparisons;
they do not establish convergence, a learning ceiling or performance on other tasks.

## Value consistency prioritization with independent hindsight heads

`acs2-agents/src/acs2vcp/` owns ACS2VCP, its separate
`ConsistencyConfiguration`, exact priority calculations and test-only reference adapter.
The configuration contains head count, integer temperature, reward scale and the existing
`acs2her::HindsightConfiguration` (copy update, admissibility, proportion). It adds no core
configuration, replay-sample or checkpoint field. Measurement exposes only four named
presets: `acs2vcp_k2`, `acs2vcp_k4` (positive T, default 1) and
`acs2vcp_k2_uniform`, `acs2vcp_k4_uniform` (T=0). All four use full copies,
future goals, every-transition admissibility and p=2/3. Values-only and shortened copies
are implemented as independent hindsight choices and checked against HER, but have no
registered measurement name. K=1 is an identity check, without a research run name.

Each head owns a separate population, random source, environment/update counts and
episode count. Episode e is played epsilon-greedily by head e modulo K on s joined with
the real goal. One shared raw trajectory store receives the episode; capacity, warmup
and three replay draws per interaction are the unchanged thesis preset. After insertion,
the acting head alone learns 3L samples. Every sample is drawn before any of these learning
steps. Originals get the shared full replay step; copies follow the configured full or
values-only HER step, with no sample-dependent learning rate. Head zero uses stream 1,
additional heads use 8/9/10, the existing sampler uses 7 and acceptance uses 12.

For a fixed end-of-episode ensemble, let q(x) denote the existing hindsight proposal
distribution over retained transition, route and selected goal. The value of each head is
`MaxFitnessBootstrap` on that sample's action set: the maximum fitness of change-anticipating
classifiers, or zero. The priority uses population variance v of these K values. Define
B = reward_on_reach squared / 4 = 250000, epsilon=1e-6 and w(x)=(B-v(x)+epsilon)^T.
Rejection sampling proposes x from q and accepts with a(x)=w(x)/(B+epsilon)^T.
The accepted law is exactly q(x)w(x)/sum_y q(y)w(y) in real arithmetic: summing over the geometric number
of preceding rejections cancels the common envelope. There is no finite materialized HER
buffer, resampling approximation, stale priority or importance correction. Both routes
are prioritized. The expected proposals per accepted draw are 1/E_q[a] at each frozen
ensemble; cumulative realized proposals/draw and acceptance are reported separately.

The machine implementation has two precision limits. Locked rand 0.8.6 implements a
Bernoulli coin with an integer threshold `floor(a * 2^64)`. A positive acceptance below
2^-64 therefore never accepts; at T=24 this starts above variance 210627.4672. At the
other end, a positive variance no greater than half an ulp of B, 2^-36, can make `B-v`
round to B. Acceptance then becomes exactly one and skips the coin. K=1 and T=0 are
unaffected: their acceptance is analytically one, so their pinned identities remain exact.

The final cumulative summaries of the 240 selected prioritized runs contain 1858105357
proposals and 989836822 acceptance coins. Their measured breakdown is:

| Proposal category | Count | Share of all proposals |
| --- | ---: | ---: |
| Exact zero variance | 863256317 | 46.458954211% |
| Positive variance with acceptance rounded to one | 5012218 | 0.269748859% |
| Variance at least 218750, certainly below the coin's probability resolution | at least 34223306 | at least 1.841838832% |

Of the rounded-to-one proposals, 4573564 occupy the positive logarithmic bin through
1e-12 and 438654 the bin through 1e-9. No accepted proposal occupies an equal-width
bin above the eighth. Bin-count upper bounds on expected acceptance-decision
discrepancies from real arithmetic are below 1.9e-12 for the probability floor,
4.3e-8 for rounding to one and 5.4e-11 for integer-threshold truncation of the coins.
The floor bound uses the three highest equal-width bins and, for each, the smaller
of 2^-64 and acceptance at its lower edge; this gives the tighter bound 5.509e-13.
The rounding bound is `24 * (4573564 * 1e-12 + 438654 * 1e-9) / (B+epsilon)`;
the coin-truncation bound is `989836822 / 2^64`. These are bounds on machine effects,
not evidence for a cause of the measured success or disagreement differences.

The fixed B avoids a running maximum discovered only after earlier samples have already
been accepted. Popoviciu's bound holds because qualities and value estimates stay in
[0,1] and [0,1000], respectively: the constructor validates beta/gamma, initial values
and reward scale, and proposals require zero nonterminal reward and terminal positive
reward within that scale. Every queried head value is checked against it. Research
temperatures are integers 0..24, computed by repeated multiplication. This range keeps
both raw priorities and their ratio finite and positive even at v=B; no logarithmic or
platform-dependent power routine is used. Underflow, invalid scales and a million
consecutive rejected proposals abort loudly without yielding an unprioritized fallback.
This explicit operational guard stops a pathological run; it does not cap a completed
draw or silently change its distribution. Acceptance one consumes no random coin. For
K=1, variance is analytically zero without value queries; at T=0 all proposals are accepted
in sampler order. T=0 makes no training value query, records no proposal/accepted
variance or candidate-temperature curve, and consumes no acceptance randomness. Its
learning remains identical; unused priority queries are removed from both independent
training counters. The measurement adapter counts the avoided classifier tests arithmetically
from the pre-episode head sizes and accepted volume, without forming a match set.

Priority calculation is immutable over every head, store and sampler. Proposal/acceptance
statistics record the work separately. Accepted routes describe actual learning samples;
objective calls include all proposals and retain scoring/selection/provenance attribution.
The proposal routes are emitted beside accepted routes. Rows add variance moments,
minimum, maximum, a sixteen-bin histogram on [0,B], B, acceptance probabilities, counts
and candidate-temperature acceptance sums, without altering existing arms' rows.
Prioritized multi-head arms also add a positive logarithmic histogram beside each existing
sixteen-bin summary. Exact zeros have their own count; positive upper bounds are
1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 1, then powers of two through 131072, and 250000.
Bins exclude their lower bound and include their upper bound. This resolves the pilot
range while retaining every existing prioritized floating field and bin.
The independent core match counter must satisfy
`formations = environment_steps + 3*full + 2*values_only + priority_value_queries`.
Each priority value query forms exactly one match set. Tests check this identity and
the head-specific interaction/update totals rather than inferring them from preset U.

Evaluation resets the evaluation RNG separately for each head, uses its own BestAction
with thesis epsilon still 0.8, and reports mean success over all heads. Every evaluation
block also contains the complete per-head evaluation fields; knowledge is averaged over
heads and recorded per head. Existing population and population-memory fields sum every
head. The additive ensemble payload contains each head's population, numerosity, marks,
logical bytes, knowledge, updates and stream, plus mean populations and total known bytes.
The raw store is counted once. Unused capacities and allocator overhead remain excluded;
the largest transient accepted batch is reported separately. Environment interactions,
update applications, priority value queries, objective calls, memory and train/evaluation
wall time are separate measurements. Read-only checks snapshot all heads, clocks, raw
history, sampler and acceptance streams, routes, counters and the measurement sidecar.

Every evaluated block adds immutable head disagreement for every multi-head arm,
including T=0. The fixed reference set contains the existing non-identity knowledge
transitions whose source/goal pair belongs to that evaluation block, with equal weight
per transition. Knowledge itself keeps its previously registered universe. The diagnostic
reports population-variance moments, the existing sixteen equal bins, the logarithmic
bins and exact-zero share, nearest-rank p25/p50/p75/p90/p99, and maximum. It makes
K times the reference-count value queries with the training counters paused; attempting
it in an active training counter scope fails. Neither learner nor evaluation random
streams are read or advanced. Queries, reference/variance temporary payload and elapsed
evaluation time remain outside training cost. Additional fixed histograms live in the
measurement adapter, preserving the old agent/header memory fields; additive fields report
the adapter bytes and the combined agent/measurement lower bound.

No importance weights are applied. The predecessor computes inverse-probability weights
and discards them; Xu's normalized-priority loss weight adds selection bias rather than
inverting it. ACS2 also shares beta between quality and reward learning. Preference for
consistent experiences is the mechanism measured here, with its bias a stated limitation.
This implementation differs from Xu's current materialized-buffer maximum and Zemlo's
running maximum, independently represented in the differential below.

The private `pyalcs_differential.rs` mode alone implements the reference whole-action-set
maximum, population variance, running maximum, raw priority, normalized buffer probability
and one V10 learning round followed by unique-sample priority updates. The fixture dumper
in `baseline/dump_vcp_differential.py` instruments the unmodified locked reference. Its
1000 cases cover K=1/2/4, T=0/1/2/4, 6996 full applications, 3474 priority refreshes,
duplicates, undrawn stale entries, covering, expected/unexpected ALP and random mark choices.
Raw priority and complete population fields use the P11 absolute 1e-9 tolerance. Cases
with T>1 use rewards 0/1 to keep raw powered values within this absolute tolerance;
T=0/1 also cover reward 1000. GA/PEE are disabled and theta_i=0 excludes the known deletion
skip hazard; no classifier deletion occurs. Selection indices are supplied to isolate the
learning/priority round from NumPy's random generator. It does not validate NumPy sampling,
unused weights, goal-blind exploration, K episodes per trial, head-zero metrics or
exploitation learning. Research constructors and CLI cannot select this mode.

Independent small-store enumeration checks future proposal masses, duplicate-goal weights,
original fallback, both used admissibility rules and proportions 0, 1/3, 2/3 and 1,
against accepted frequencies and the inverse-acceptance cost. K=1 at T=0 and T=24 reproduces
96 pinned HER/control rows across all six mazes, including all learning, evaluation and
knowledge fields. The only excluded fields are provenance/timing and VCP-only identity
or additional diagnostics. Separate defect tests cover joined goals, change anticipation,
independent heads/streams, acting rotation, joint budgets, greedy ensemble evaluation,
frozen priorities, unweighted full/values steps and the match-count identities.

### Validation checks, pilot costs and approved comparison protocol

Clean measurement `79648424076144c77a3a87acf87418bb61aa7c42`, musl binary SHA-256
`c2478eb60caad34a5bdcc949b77018c329c4fbcdbd8629b60ca7dcb87c2fb6fa`, supplies the
initial checks. Sequential validation-only checks completed all 180 runs: six mazes,
seeds 42--46, K=2/4 at T=1/24 and both T=0 controls, with points 1k/10k/20k.
All 540 rows pass the joint interaction/update/match identities. This is an operational
check, without a performance ranking or test-block analysis. Hindsight settings remain
full copies, every transition and p=2/3.

The first cluster allocation group measured Maze4, seed 42. Both K=2 arms completed
2M interactions; both K=4 arms reached the 1M point but hit the 449-second process
watchdog before the 2M point. The four allocation CPUTimeRAW values are 283, 451, 295
and 451 seconds: 1480 seconds (0.411111 CPU-hours), including 902 seconds in failed
attempts. After explicit user authorization, the twenty dependent pending jobs were
cancelled with zero allocation cost. STOP is retained. This was an undersized process
limit, not evidence of a learning ceiling.

`prepare_recovery.py` freezes separate operational manifests with one maze and one K
per part, at most two jobs, and longer per-maze/head limits. Its shared accounting
scope includes the immutable initial ledger and every potential recovery part, so
failed allocations remain charged and completed K=2 Maze4 runs are reused. A separate
STOP preserves the original failure record. The second temperature is admitted by
complete maze pairs in a fixed cost-pilot order, within the same five-hour bound.
The archive verifier charges every terminal allocation, including failures and
cancellations, verifies all raw hashes, and requires interrupted prefixes to agree
with complete retries. `pin_pilot.py` derives the portable regression fixture only
from this independently audited inventory.

The completed pilot has **34 successful trajectories and 278 validation rows**:
all six mazes at T=1 and both T=0 controls, and both K values at T=24 on Maze4,
Maze5, Maze7, Woods14 and MazeMA. Woods14 has nine registered points; the others have
eight. The second temperature on Maze6 was not admitted: already used allocations
plus its two registered limits (1080 and 1680 seconds) exceeded the 18000-second
authorization even after releasing all outstanding reservations. The immutable
decision records that omitted scope without reducing any interaction budget.

Allocation CPUTimeRAW, seconds, for successful selected trajectories:

| Task | K=2, T=1 | K=4, T=1 | K=2, T=0 | K=4, T=0 | K=2, T=24 | K=4, T=24 |
|---|---:|---:|---:|---:|---:|---:|
| Maze4 | 283 | 446 | 295 | 460 | 203 | 431 |
| Maze5 | 479 | 833 | 453 | 905 | 458 | 871 |
| Maze6 | 469 | 916 | 477 | 899 | not admitted | not admitted |
| Maze7 | 548 | 1064 | 521 | 979 | 484 | 1038 |
| Woods14 | 125 | 195 | 132 | 199 | 137 | 296 |
| MazeMA | 58 | 117 | 58 | 110 | 68 | 132 |

All **56 allocations** are accounted for: 34 completed, two failed and twenty
cancelled. Including the failed attempts, the pilot used **16041 seconds =
4.455833 CPU-hours**, with zero remaining reservations. The largest admission bound
was 17992 seconds. Every submission checked the shared grant and test-only schedule.
All 874 source files were independently checksum-verified after transfer; the full
raw archive and derived summaries are retained on the cluster and locally.

At final points, cumulative mean proposal variance spans 69.743--759.568 at T=1
and 5036.500--23874.746 at T=24. Realized acceptance spans 0.996934--0.999723 and
0.827010--0.962502 respectively; proposals per draw span 1.000277--1.003075 and
1.038959--1.209175. The bound stays 250000 at every point. The archived T=0 pilot queried values for diagnostics despite accepting every
proposal. The revised control removes that cost and measures disagreement only during
evaluation; the immutable pilot retains its original counters. Full per-point moments, histograms,
accepted/proposal routes, objective costs, per-head success/knowledge/population,
memory and times remain in the verified archive. These one-seed operational
measurements do not rank policies or establish convergence.

`fixtures/vcp_pair_regression.json` pins the two earliest points of all 34 selected
pilot trajectories (68 validation rows). Coverage guards require all six first-temperature
mazes and controls and both K values for every declared second-temperature maze.
Every portable floating field
uses binary64 bits. Inline type headers have two measured native layouts: head/agent
without store headers of 376/1832 bytes on the laptop and 384/1856 on the musl build.
The fixture factors those separately, accepts only these measured layouts, and checks
the remaining row plus reconstructed native memory fields bit for bit. The revised
comparison names exactly six changed control paths: training match formations, training
classifier-perception tests, priority value queries, proposal variance, accepted variance
and candidate acceptance sums. The last three become null; the first three remove
exactly the unused queries. All other existing control fields, all existing prioritized
fields and the 96 one-head HER identity rows remain bit-identical. The fixture bytes
are unchanged. Direct differing-head tests now require first-action values 100/300
to average to 200 and knowledge 1/0.5 to average to 0.75. It does not
pretend that compiler/target-dependent inline storage is identical across targets.
Classifiers, marks, raw storage and all learning/evaluation/knowledge fields agree in
the shared 1k/10k cross-platform check.

The test-only K=1 adapter declares the same greedy policy identity as HER, in addition
to matching all numeric fields. Mutation checks found that structural `Classifier`
equality alone omitted reward/quality statistics in the initial unweighted-step test.
Those checks now compare the complete classifier state. Nine Rust mutations are caught:
whole-action-set value, sample variance, a probability-one coin, an extra exponent,
head-zero acting, weighted learning, learning between priority draws, head-multiplied
draw volume and head-zero evaluation. Both shared-tooling gaps also have failing
mutation checks: removing the committed baseline-registration comparison and allowing
substituted measurement identities. All mutated sources are restored afterwards.

The experiment-specific code in `experiments/value-consistency/` supplies exact
head-averaged rational success/threshold calculations, paired seed summaries in both
registered views, the priority-only pilot acceptance curves, a deterministic control-work
calibration, a doubled forecast with actual batch reservations, and the draft builder.
The historical pilot identity is frozen separately in `pilot-measurement.json`; the
current clean source/binary identity belongs to `measurement.json`. Archive validation
continues to audit the original pilot without rewriting its registrations or counters.
`pilot_equivalence.py` checks full pilot-overlapping rows with only the documented
control-cost changes, additive diagnostics, and measured native header reconstruction.
The execution adapter accepts the fixed grid only for approved registration SHA-256
`9428773fa7861f6102fdccb165ae711b597d1fd61a9e644f8becb17399ebf4eb`
and its explicit 110 allocated CPU-hour approval, including every pilot attempt.

The immutable approved artifact retains the filename `plan-draft.json`. It includes all four arms, six mazes and seeds
42--61: prioritized K=2 and K=4 at fixed T=24, and both uniform controls at T=0.
There is no tuning and no automatic sixty-hour cost trigger. Across 1..24 the
pooled pilot acceptance curves remain at least 0.90 and cannot reach the original
3/4, 1/2 or 1/4 targets. T=1 accepted 99.69--99.97% of proposals, effectively repeating
the uniform control; T=24 is the strongest prioritization in the registered range. This choice
precedes the final results. The user confirmed full copies, every transition, future
and p=2/3 on 2026-10-16; the existing approved full-update HER arm is the reference.

The fixed grid has 480 runs, all three evaluation blocks, all registered budgets and
points, and nine comparisons in both views: VCP minus control at each K; VCP and
control minus HER at each K; VCP minus episodic replay at each K; and VCP K=4 minus
K=2. It reuses the approved HER and pair-baseline references, exact thresholds, and their
threshold-crossing checks. Seed-42 grid rows must reproduce every overlapping pilot
learning/evaluation/knowledge field; controls follow the zero-query identity. Maze6
T=24 has no pilot identity oracle and is explicitly excluded from that claim.

`calibrate_controls.py` runs the two revised controls sequentially for a short local
seed-42 check on all six mazes, at the registered 1k/3k/30k points. Its classifier-test
ratio to the same pilot point scales each control's expected cost, while every control
limit stays at least its original pilot-derived size. Maze6 T=24 uses the T=1 cost
scaled by the largest observed T=24/T=1 allocation-CPUTimeRAW ratio on the other five
mazes, separately for each K. Additional evaluation work for all blocks is budgeted
from their fixed transition/pair counts and pilot classifier-test cost. Hard reservations
charge every actual twenty-job batch at its shared maximum limit. The requested limit
is the larger of the doubled forecast including every pilot attempt and that hard bound,
rounded up to five CPU-hours. The user approved 110 hours for the fixed grid and
all pilot attempts; there is no tuning allocation.

The revised clean measurement is `82c08b5a89a63878fddfb76be8d5e25d0ae17a41`, with
locally cross-built musl SHA-256
`0c8dcb9c539462f9b84800519237a24e34c49e4d81b70a79a17cae0d7a98816d`.
The short control calibration completed twelve sequential runs and 36 rows in 85.65
seconds. All 1k/3k/30k pilot-overlapping learning/evaluation/knowledge fields agree.
Classifier-test cost ratios span 0.625034--0.625102 for K=2 and 0.454524--0.454707
for K=4. `control-work-calibration.json` pins these deterministic ratios and the raw
checksums; none is a timing ratio or a policy ranking.

Maze6 T=24 is estimated from the T=1 pilot with same-K maximum allocation ratios
68/58 = 1.172413793 (MazeMA, K=2) and 296/195 = 1.517948718 (Woods14, K=4).
Including separately budgeted additional evaluation, the expected body/allocation
seconds are 554.067/561.476 and 1404.236/1418.721. Their hard run limits are 15 and
36 minutes. Every control keeps its previous pilot-derived grid limit.

| Arm | Expected grid CPU-hours | Hard first-attempt reservations, hours |
|---|---:|---:|
| K=2, T=24 | 10.774141 | 18.000000 |
| K=4, T=24 | 23.622933 | 36.666667 |
| K=2, T=0 | 6.974226 | 17.666667 |
| K=4, T=0 | 9.458912 | 31.333333 |

The expected 480-run grid cost is **50.830212 hours**. Twice that forecast plus all
16041 pilot allocation seconds is **106.116258 hours**. Actual twenty-job batches
reserve 373200 seconds; with the pilot the hard bound is **108.122500 hours**.
The approved ceiling is **110 cumulative CPU-hours**.
This is a forecast from one seed, a short work calibration and an unmeasured Maze6
second temperature; the new training histogram overhead has not been timed separately.
The unchanged control limits, 1.5-body-plus-30-second run margins, doubled estimate,
STOP and cumulative hard admission accounting remain the operational bounds.

Eight further mutations are caught directly: restored T=0 value queries, a summed
first-action estimate, head-zero knowledge, merging exact zeros into positive bins,
an off-by-one quantile rank, an unscaled Maze6 estimate, an unscaled control cost and
changing the fixed temperature. Every mutation is restored before the gates. Shared
collection accepts the original pilot schema and requires the revised zero-query and
per-block disagreement schema for the new grid. Artifact tests check the actual draft,
its source/calibration hashes, all nine contrasts and the real shared-tooling batches.
The registered source and Cargo trees are unchanged by the pre-submission tests and
execution/analysis adapters. The exact approved musl file is copied into the dedicated
clean clone; no cluster rebuild is used.

The pre-submission row test independently counts non-identity transitions from each
evaluated block's start-goal pairs on every maze and all four arms. Substituting the
full knowledge universe in the runner makes this test fail. A negative row-schema test
rejects revised controls with nonzero priority queries, even if their match counter is
adjusted to hide them, or with non-null training variance. Removing that query guard
makes the negative test fail. Both mutations are restored before measurement.

`prepare_study.py` freezes the approved plan, pilot rows and operations. Each normal
`execute_batch.py` invocation derives twenty runs, includes all twenty-four roots in
the shared budget scope, and checks the grant and test-only schedule. Node-side
guards compare clean source, binary, hardware, preset, rows and pilot identities;
a real failure creates a shared STOP. No lease, host pinning, laptop automation or
additional monitor is used.
Shared submission and resumption also load the scope by its registered SHA-256 and
require the resolved batch root to belong to its resolved `batch_roots`, before any
budget query, directory creation or scheduler call. Plans without a shared scope keep
their existing execution path.

The first Maze4/K=2 array, 6109875, hit its registered process deadline at seed 44.
Thirteen runs completed; shared STOP interrupted six others. Its twenty allocations
used 5634 CPU-seconds. The separately approved `recovery.json` (SHA-256
`afae09a7e1e96ba48c35169cf7bd45ca6452c2a6dc303e2bb8db1d377e989615`)
permits only seeds 44/46/49/52/53/59/61 to retry with ten-minute limits in one
seven-job recovery array. Array 6109917 completed all seven, costing 2247 seconds.
The first STOP and all failed prefixes remain archived; all 49 repeated prefix
rows agree in binary64 bits. The combined twenty selected trajectories cost 5815
seconds; all twenty-seven attempts cost 7881. The whole remaining hard bound at
recovery admission, including the pilot, was 391875 seconds, below 110 hours.

`verify.py` checks every raw artifact and allocation, exact attempt scopes, immutable
operations and launch identities, and independently reproduces node-side pilot
checks. Analysis requires all 480 completed cells and all 180 pilot-overlapping rows.
It also requires exactly one admission event per declared cluster batch root, with no
foreign or duplicate root. The only additional submission is the approved recovery:
its resolution and submission receipt must match the second Maze4/K=2 ledger entry.
The seven interrupted prefixes are also compared with their complete retries.
`references.py` verifies the approved HER and pair-baseline archives without rerunning
them. `analyze.py` exports both test-success views, all nine paired comparisons,
exact rational thresholds, block curves, per-head diagnostics, distributions and
separate process/allocation/wall costs. Seeds remain the statistical units. Evaluation
quantiles are summarized across seeds; pooled histograms instead weight raw counts.
An unmeasured control training variance stays absent.
The exported integer head-success counts and exact checkpoint fractions support an
independent rational reconstruction in `validate_tables.py`: seed scores, both sets of
paired intervals, exact threshold crossings and allocation scope are checked without
calling the analysis arithmetic. Synthetic export mutations using unpaired intervals,
strict threshold crossing or charging historical references are all rejected.

Complete episodes retain the registered overshoot below cap: nominally equal budgets
can have slightly different actual_steps. Actual interactions are counted once and
updates equal three times that actual count; K never multiplies the budget.

Implemented-contract gates pass 309 Rust tests; Python suites pass 47/7/20/40/37/14/13/41
for tools, random policy, baseline pilot, grid, HER variants, maze baselines, composition
and value consistency. P9 columns 1--7 are unchanged. Cleaning all six workspace packages
before Clippy reproduces the same 22 older warnings, with none in the additions.

### Held-pair success with fixed-temperature consistency selection

The approved four-arm grid is complete: **480 selected runs, 487 attempts and 3920
selected checkpoint rows**, with training, validation and test blocks in each row.
All 180 overlapping seed-42 validation rows reproduce the pilot, subject only to the
documented control-counter changes and additive diagnostics; Maze6 T=24 has no pilot
oracle. All 49 interrupted-prefix rows reproduce their approved retries. No tuning or
additional experimental arm was run. The analysis and independent table reconstruction
use `6f9a9c3691dd96dd2fb4faf6d480697b989f929e`; the measurement source remains
`82c08b5a89a63878fddfb76be8d5e25d0ae17a41`, with an empty source/Cargo diff.

In the tables, V2/V4 mean `acs2vcp_k2`/`acs2vcp_k4` at T=24, and U2/U4 mean their
`_uniform` controls at T=0. HER is `acs2her`; ER-E is `acs2er_episodic`.
The approved historical HER and pair-baseline runs supply all four reference arms;
their costs are not charged to this execution. Every cell uses seeds 42--61 (n=20).
Success is the head mean within a seed. The primary view weights registered points
equally within each seed, then reports the seed mean and SE; the other view uses the
last point. These are proportions, not percentages or an area under the curve.
The horizon is 2M interactions, except Woods14 at 4M. Nominal points retain the
registered complete-episode overshoot; actual interactions are counted once.

| Maze | Arm | Checkpoint mean ± SE | Final point ± SE |
| --- | --- | --- | --- |
| Maze4 | V2 | 0.166319 ± 0.002271 | 0.211420 ± 0.006386 |
| Maze4 | V4 | 0.145747 ± 0.003231 | 0.193750 ± 0.005247 |
| Maze4 | U2 | 0.181964 ± 0.003941 | 0.224228 ± 0.005697 |
| Maze4 | U4 | 0.168113 ± 0.002532 | 0.234568 ± 0.004310 |
| Maze4 | HER | 0.184722 ± 0.004090 | 0.225000 ± 0.009620 |
| Maze4 | ER-E | 0.172608 ± 0.004452 | 0.227778 ± 0.008220 |
| Maze4 | ACS2 | 0.193403 ± 0.003761 | 0.325309 ± 0.009405 |
| Maze4 | ACS2ER | 0.166512 ± 0.004987 | 0.227160 ± 0.011862 |
| Maze5 | V2 | 0.122579 ± 0.002429 | 0.165203 ± 0.005105 |
| Maze5 | V4 | 0.103836 ± 0.001879 | 0.146171 ± 0.002908 |
| Maze5 | U2 | 0.130152 ± 0.001622 | 0.183671 ± 0.003574 |
| Maze5 | U4 | 0.124859 ± 0.001616 | 0.184516 ± 0.003482 |
| Maze5 | HER | 0.140541 ± 0.002868 | 0.191892 ± 0.006951 |
| Maze5 | ER-E | 0.111008 ± 0.003451 | 0.145721 ± 0.006571 |
| Maze5 | ACS2 | 0.131306 ± 0.002771 | 0.235586 ± 0.008371 |
| Maze5 | ACS2ER | 0.116104 ± 0.003337 | 0.160360 ± 0.009625 |
| Maze6 | V2 | 0.134516 ± 0.002544 | 0.198761 ± 0.004942 |
| Maze6 | V4 | 0.117244 ± 0.001747 | 0.177421 ± 0.004237 |
| Maze6 | U2 | 0.146664 ± 0.002226 | 0.205856 ± 0.004337 |
| Maze6 | U4 | 0.138725 ± 0.001911 | 0.223649 ± 0.003897 |
| Maze6 | HER | 0.156503 ± 0.002389 | 0.209234 ± 0.004852 |
| Maze6 | ER-E | 0.133615 ± 0.003676 | 0.181081 ± 0.011403 |
| Maze6 | ACS2 | 0.156954 ± 0.003462 | 0.284910 ± 0.010457 |
| Maze6 | ACS2ER | 0.128801 ± 0.003729 | 0.166441 ± 0.010022 |
| Maze7 | V2 | 0.126591 ± 0.001814 | 0.175579 ± 0.003451 |
| Maze7 | V4 | 0.116884 ± 0.001368 | 0.178646 ± 0.002054 |
| Maze7 | U2 | 0.140177 ± 0.001742 | 0.197454 ± 0.003434 |
| Maze7 | U4 | 0.128610 ± 0.002223 | 0.201678 ± 0.002658 |
| Maze7 | HER | 0.157523 ± 0.003102 | 0.207639 ± 0.005377 |
| Maze7 | ER-E | 0.138021 ± 0.003490 | 0.201157 ± 0.011896 |
| Maze7 | ACS2 | 0.147656 ± 0.003611 | 0.266435 ± 0.011896 |
| Maze7 | ACS2ER | 0.137124 ± 0.003336 | 0.179861 ± 0.010283 |
| Woods14 | V2 | 0.154971 ± 0.002798 | 0.180702 ± 0.006313 |
| Woods14 | V4 | 0.142325 ± 0.003824 | 0.169079 ± 0.005125 |
| Woods14 | U2 | 0.150195 ± 0.003606 | 0.171053 ± 0.005815 |
| Woods14 | U4 | 0.150146 ± 0.003098 | 0.171272 ± 0.003607 |
| Woods14 | HER | 0.164912 ± 0.005264 | 0.183333 ± 0.008197 |
| Woods14 | ER-E | 0.166862 ± 0.005303 | 0.200000 ± 0.018063 |
| Woods14 | ACS2 | 0.184795 ± 0.006324 | 0.197368 ± 0.014336 |
| Woods14 | ACS2ER | 0.172904 ± 0.008103 | 0.183333 ± 0.016471 |
| MazeMA | V2 | 0.172431 ± 0.004844 | 0.203889 ± 0.008112 |
| MazeMA | V4 | 0.178090 ± 0.003504 | 0.233611 ± 0.005735 |
| MazeMA | U2 | 0.188125 ± 0.006367 | 0.223333 ± 0.010660 |
| MazeMA | U4 | 0.183507 ± 0.004801 | 0.238889 ± 0.006474 |
| MazeMA | HER | 0.199028 ± 0.008341 | 0.213333 ± 0.013527 |
| MazeMA | ER-E | 0.197361 ± 0.007135 | 0.234444 ± 0.014552 |
| MazeMA | ACS2 | 0.187361 ± 0.006450 | 0.272222 ± 0.017725 |
| MazeMA | ACS2ER | 0.190972 ± 0.007079 | 0.214444 ± 0.018116 |

Every registered contrast follows, as mean paired difference ± SE and marginal 95%
t(19) interval (critical value 2.093024054408263). Heads are not extra statistical
replicates. There is no multiplicity correction or pooled test across mazes.

| Maze | Contrast | Checkpoint Δ ± SE [95% CI] | Final Δ ± SE [95% CI] |
| --- | --- | --- | --- |
| Maze4 | V2 − U2 | -0.015644 ± 0.003755 [-0.023503, -0.007785] | -0.012809 ± 0.007050 [-0.027564, +0.001947] |
| Maze4 | V4 − U4 | -0.022367 ± 0.003744 [-0.030203, -0.014530] | -0.040818 ± 0.005642 [-0.052626, -0.029009] |
| Maze4 | V2 − HER | -0.018403 ± 0.004229 [-0.027254, -0.009552] | -0.013580 ± 0.010772 [-0.036126, +0.008966] |
| Maze4 | V4 − HER | -0.038976 ± 0.004553 [-0.048506, -0.029445] | -0.031250 ± 0.011765 [-0.055875, -0.006625] |
| Maze4 | U2 − HER | -0.002758 ± 0.005657 [-0.014598, +0.009081] | -0.000772 ± 0.010404 [-0.022547, +0.021004] |
| Maze4 | U4 − HER | -0.016609 ± 0.004666 [-0.026375, -0.006842] | +0.009568 ± 0.010410 [-0.012220, +0.031356] |
| Maze4 | V2 − ER-E | -0.006289 ± 0.004758 [-0.016247, +0.003670] | -0.016358 ± 0.010146 [-0.037593, +0.004877] |
| Maze4 | V4 − ER-E | -0.026861 ± 0.005582 [-0.038544, -0.015179] | -0.034028 ± 0.010043 [-0.055047, -0.013008] |
| Maze4 | V4 − V2 | -0.020573 ± 0.003582 [-0.028071, -0.013075] | -0.017670 ± 0.009029 [-0.036567, +0.001227] |
| Maze5 | V2 − U2 | -0.007573 ± 0.003058 [-0.013974, -0.001172] | -0.018468 ± 0.006150 [-0.031340, -0.005597] |
| Maze5 | V4 − U4 | -0.021023 ± 0.001767 [-0.024722, -0.017325] | -0.038345 ± 0.004084 [-0.046893, -0.029796] |
| Maze5 | V2 − HER | -0.017962 ± 0.003746 [-0.025803, -0.010121] | -0.026689 ± 0.007470 [-0.042325, -0.011054] |
| Maze5 | V4 − HER | -0.036705 ± 0.002702 [-0.042360, -0.031049] | -0.045721 ± 0.005974 [-0.058225, -0.033217] |
| Maze5 | U2 − HER | -0.010389 ± 0.003790 [-0.018320, -0.002457] | -0.008221 ± 0.008636 [-0.026295, +0.009854] |
| Maze5 | U4 − HER | -0.015681 ± 0.003358 [-0.022710, -0.008653] | -0.007376 ± 0.006864 [-0.021743, +0.006991] |
| Maze5 | V2 − ER-E | +0.011571 ± 0.004553 [+0.002042, +0.021100] | +0.019482 ± 0.009637 [-0.000688, +0.039652] |
| Maze5 | V4 − ER-E | -0.007172 ± 0.004416 [-0.016416, +0.002072] | +0.000450 ± 0.008261 [-0.016841, +0.017742] |
| Maze5 | V4 − V2 | -0.018743 ± 0.002718 [-0.024432, -0.013054] | -0.019032 ± 0.005466 [-0.030473, -0.007590] |
| Maze6 | V2 − U2 | -0.012148 ± 0.003594 [-0.019670, -0.004627] | -0.007095 ± 0.007277 [-0.022325, +0.008136] |
| Maze6 | V4 − U4 | -0.021481 ± 0.002244 [-0.026178, -0.016784] | -0.046227 ± 0.006104 [-0.059004, -0.033451] |
| Maze6 | V2 − HER | -0.021988 ± 0.003529 [-0.029375, -0.014601] | -0.010473 ± 0.006095 [-0.023229, +0.002283] |
| Maze6 | V4 − HER | -0.039260 ± 0.002998 [-0.045534, -0.032985] | -0.031813 ± 0.007032 [-0.046531, -0.017095] |
| Maze6 | U2 − HER | -0.009840 ± 0.003435 [-0.017029, -0.002650] | -0.003378 ± 0.005913 [-0.015754, +0.008997] |
| Maze6 | U4 − HER | -0.017779 ± 0.003164 [-0.024401, -0.011156] | +0.014414 ± 0.005959 [+0.001942, +0.026887] |
| Maze6 | V2 − ER-E | +0.000901 ± 0.003472 [-0.006366, +0.008168] | +0.017680 ± 0.009849 [-0.002934, +0.038294] |
| Maze6 | V4 − ER-E | -0.016371 ± 0.003869 [-0.024469, -0.008273] | -0.003660 ± 0.012584 [-0.029998, +0.022678] |
| Maze6 | V4 − V2 | -0.017272 ± 0.002833 [-0.023202, -0.011342] | -0.021340 ± 0.006667 [-0.035295, -0.007385] |
| Maze7 | V2 − U2 | -0.013585 ± 0.002381 [-0.018568, -0.008602] | -0.021875 ± 0.004895 [-0.032119, -0.011631] |
| Maze7 | V4 − U4 | -0.011726 ± 0.002194 [-0.016319, -0.007133] | -0.023032 ± 0.003593 [-0.030552, -0.015512] |
| Maze7 | V2 − HER | -0.030932 ± 0.003300 [-0.037839, -0.024025] | -0.032060 ± 0.007424 [-0.047599, -0.016522] |
| Maze7 | V4 − HER | -0.040639 ± 0.002557 [-0.045992, -0.035287] | -0.028993 ± 0.005235 [-0.039949, -0.018037] |
| Maze7 | U2 − HER | -0.017347 ± 0.003342 [-0.024341, -0.010352] | -0.010185 ± 0.007403 [-0.025680, +0.005310] |
| Maze7 | U4 − HER | -0.028913 ± 0.003659 [-0.036572, -0.021255] | -0.005961 ± 0.006560 [-0.019690, +0.007769] |
| Maze7 | V2 − ER-E | -0.011429 ± 0.004148 [-0.020111, -0.002748] | -0.025579 ± 0.012540 [-0.051824, +0.000667] |
| Maze7 | V4 − ER-E | -0.021137 ± 0.003874 [-0.029245, -0.013029] | -0.022512 ± 0.011937 [-0.047496, +0.002473] |
| Maze7 | V4 − V2 | -0.009708 ± 0.001895 [-0.013675, -0.005741] | +0.003067 ± 0.003309 [-0.003859, +0.009994] |
| Woods14 | V2 − U2 | +0.004776 ± 0.004675 [-0.005009, +0.014560] | +0.009649 ± 0.009351 [-0.009922, +0.029220] |
| Woods14 | V4 − U4 | -0.007822 ± 0.004437 [-0.017108, +0.001465] | -0.002193 ± 0.006431 [-0.015653, +0.011267] |
| Woods14 | V2 − HER | -0.009942 ± 0.006468 [-0.023479, +0.003596] | -0.002632 ± 0.010517 [-0.024643, +0.019380] |
| Woods14 | V4 − HER | -0.022588 ± 0.006423 [-0.036031, -0.009144] | -0.014254 ± 0.009216 [-0.033543, +0.005035] |
| Woods14 | U2 − HER | -0.014717 ± 0.006019 [-0.027316, -0.002119] | -0.012281 ± 0.009275 [-0.031693, +0.007131] |
| Woods14 | U4 − HER | -0.014766 ± 0.005663 [-0.026619, -0.002913] | -0.012061 ± 0.010072 [-0.033141, +0.009019] |
| Woods14 | V2 − ER-E | -0.011891 ± 0.006610 [-0.025725, +0.001943] | -0.019298 ± 0.017973 [-0.056915, +0.018319] |
| Woods14 | V4 − ER-E | -0.024537 ± 0.007320 [-0.039857, -0.009217] | -0.030921 ± 0.019593 [-0.071929, +0.010087] |
| Woods14 | V4 − V2 | -0.012646 ± 0.003786 [-0.020570, -0.004723] | -0.011623 ± 0.006959 [-0.026188, +0.002942] |
| MazeMA | V2 − U2 | -0.015694 ± 0.006753 [-0.029828, -0.001561] | -0.019444 ± 0.012248 [-0.045080, +0.006191] |
| MazeMA | V4 − U4 | -0.005417 ± 0.006483 [-0.018985, +0.008151] | -0.005278 ± 0.007090 [-0.020118, +0.009562] |
| MazeMA | V2 − HER | -0.026597 ± 0.010033 [-0.047596, -0.005598] | -0.009444 ± 0.012624 [-0.035867, +0.016979] |
| MazeMA | V4 − HER | -0.020937 ± 0.010594 [-0.043112, +0.001237] | +0.020278 ± 0.015417 [-0.011991, +0.052547] |
| MazeMA | U2 − HER | -0.010903 ± 0.010470 [-0.032816, +0.011010] | +0.010000 ± 0.016538 [-0.024613, +0.044613] |
| MazeMA | U4 − HER | -0.015521 ± 0.009487 [-0.035378, +0.004336] | +0.025556 ± 0.014103 [-0.003962, +0.055074] |
| MazeMA | V2 − ER-E | -0.024931 ± 0.009289 [-0.044373, -0.005488] | -0.030556 ± 0.017750 [-0.067706, +0.006595] |
| MazeMA | V4 − ER-E | -0.019271 ± 0.006890 [-0.033692, -0.004850] | -0.000833 ± 0.016482 [-0.035330, +0.033663] |
| MazeMA | V4 − V2 | +0.005660 ± 0.005659 [-0.006184, +0.017503] | +0.029722 ± 0.010388 [+0.007980, +0.051464] |

For the primary VCP-minus-control comparison, intervals lie below zero on Maze4--7
and MazeMA at K=2, and on Maze4--7 at K=4. The remaining intervals include zero.
At the final point they lie below zero on Maze5/Maze7 at K=2 and Maze4--7 at K=4.
No VCP-minus-control interval lies wholly above zero in either view. This does not
imply that VCP loses every registered comparison: for example V2 minus episodic
replay on Maze5 is positive in the primary view. The inference is limited to this
fixed T, bound, hindsight configuration, pair protocol and interaction horizon.

### Exact threshold crossings at registered interaction points

Thresholds are the approved pair-baseline final mean test successes, represented
exactly as rational numbers. A crossing is the first registered nominal point whose
twenty-seed mean is greater than or equal to the threshold, with no interpolation.
It need not remain above the threshold at the final point. The historical baseline
crossing checks also pass. A dash means no crossing in the measured horizon.

| Maze | ACS2 final threshold | ACS2ER final threshold |
| --- | --- | --- |
| Maze4 | 527/1620 | 92/405 |
| Maze5 | 523/2220 | 89/555 |
| Maze6 | 253/888 | 739/4440 |
| Maze7 | 1151/4320 | 259/1440 |
| Woods14 | 15/76 | 11/60 |
| MazeMA | 49/180 | 193/900 |

| Maze | Arm | Steps to ACS2 threshold | Steps to ACS2ER threshold |
| --- | --- | --- | --- |
| Maze4 | V2 | — | — |
| Maze4 | V4 | — | — |
| Maze4 | U2 | — | 100000 |
| Maze4 | U4 | — | 100000 |
| Maze5 | V2 | — | 100000 |
| Maze5 | V4 | — | — |
| Maze5 | U2 | — | 100000 |
| Maze5 | U4 | — | 100000 |
| Maze6 | V2 | — | 100000 |
| Maze6 | V4 | — | 1000000 |
| Maze6 | U2 | — | 100000 |
| Maze6 | U4 | — | 100000 |
| Maze7 | V2 | — | — |
| Maze7 | V4 | — | — |
| Maze7 | U2 | — | 100000 |
| Maze7 | U4 | — | 300000 |
| Woods14 | V2 | — | — |
| Woods14 | V4 | — | — |
| Woods14 | U2 | — | — |
| Woods14 | U4 | — | — |
| MazeMA | V2 | — | 100000 |
| MazeMA | V4 | — | 100000 |
| MazeMA | U2 | — | 30000 |
| MazeMA | U4 | — | 100000 |

None of the four new arms crosses the ACS2 threshold. Woods14 crosses neither
threshold with any new arm. `thresholds.csv` also retains all four reused references.

### Common head disagreement and replay selection cost

These final-point cells are seed means ± SE. Knowledge uses the unchanged complete
goal-conditioned non-identity universe. Test disagreement instead uses the fixed
non-identity transitions from the test block's start-goal pairs. It is measured for
every multi-head arm with training counters paused and no state or stream change.
Total population counts every head; per-head populations and spread are retained in
`diagnostics.csv` and `point-diagnostics.csv` in both registered views.

| Maze | Arm | Knowledge ± SE | Test head variance ± SE | Exact-zero share ± SE | Total classifiers ± SE |
| --- | --- | --- | --- | --- | --- |
| Maze4 | V2 | 0.971586 ± 0.002363 | 12625.71 ± 894.80 | 0.007238 ± 0.000414 | 1900.35 ± 39.95 |
| Maze4 | V4 | 0.953251 ± 0.002238 | 21271.80 ± 998.53 | 0.006942 ± 0.000305 | 3750.00 ± 47.55 |
| Maze4 | U2 | 1.000000 ± 0.000000 | 38.76 ± 1.12 | 0.007459 ± 0.000273 | 2042.00 ± 29.47 |
| Maze4 | U4 | 0.999996 ± 0.000004 | 60.92 ± 1.14 | 0.007312 ± 0.000312 | 4040.00 ± 56.22 |
| Maze5 | V2 | 0.947776 ± 0.002815 | 24691.82 ± 1269.61 | 0.005326 ± 0.000279 | 2842.10 ± 42.19 |
| Maze5 | V4 | 0.914609 ± 0.003161 | 35712.78 ± 939.79 | 0.004868 ± 0.000274 | 5724.75 ± 54.73 |
| Maze5 | U2 | 0.999995 ± 0.000005 | 43.16 ± 1.49 | 0.005326 ± 0.000291 | 3130.10 ± 44.72 |
| Maze5 | U4 | 0.999946 ± 0.000015 | 71.70 ± 3.19 | 0.005269 ± 0.000226 | 6849.20 ± 65.23 |
| Maze6 | V2 | 0.946776 ± 0.002954 | 19713.55 ± 954.18 | 0.005746 ± 0.000276 | 3154.80 ± 41.61 |
| Maze6 | V4 | 0.912410 ± 0.002422 | 29899.99 ± 1058.35 | 0.004858 ± 0.000284 | 6331.95 ± 68.67 |
| Maze6 | U2 | 0.999995 ± 0.000005 | 49.01 ± 1.34 | 0.006102 ± 0.000276 | 3598.85 ± 50.21 |
| Maze6 | U4 | 0.999917 ± 0.000026 | 86.99 ± 3.80 | 0.004917 ± 0.000262 | 7466.95 ± 61.93 |
| Maze7 | V2 | 0.942476 ± 0.002049 | 30954.48 ± 1079.81 | 0.003388 ± 0.000327 | 3318.80 ± 48.24 |
| Maze7 | V4 | 0.903466 ± 0.002851 | 44786.13 ± 1331.56 | 0.003388 ± 0.000223 | 6658.60 ± 85.56 |
| Maze7 | U2 | 1.000000 ± 0.000000 | 47.51 ± 1.32 | 0.003738 ± 0.000313 | 3739.30 ± 43.86 |
| Maze7 | U4 | 0.999932 ± 0.000012 | 99.29 ± 15.54 | 0.003329 ± 0.000153 | 7994.60 ± 77.51 |
| Woods14 | V2 | 0.978318 ± 0.004325 | 13327.71 ± 1781.52 | 0.012255 ± 0.001996 | 606.25 ± 13.08 |
| Woods14 | V4 | 0.952045 ± 0.006817 | 19818.84 ± 985.45 | 0.003922 ± 0.001102 | 1324.30 ± 26.26 |
| Woods14 | U2 | 1.000000 ± 0.000000 | 87.52 ± 5.44 | 0.012255 ± 0.002119 | 605.25 ± 14.50 |
| Woods14 | U4 | 1.000000 ± 0.000000 | 125.17 ± 5.20 | 0.006863 ± 0.001757 | 1279.85 ± 20.81 |
| MazeMA | V2 | 0.969538 ± 0.005012 | 10349.70 ± 2055.32 | 0.001596 ± 0.000871 | 526.30 ± 11.36 |
| MazeMA | V4 | 0.960215 ± 0.004433 | 17178.88 ± 1598.10 | 0.000532 ± 0.000532 | 1168.45 ± 21.12 |
| MazeMA | U2 | 1.000000 ± 0.000000 | 64.37 ± 4.21 | 0.001596 ± 0.000871 | 545.95 ± 12.12 |
| MazeMA | U4 | 1.000000 ± 0.000000 | 100.75 ± 3.54 | 0.000000 ± 0.000000 | 1128.80 ± 24.19 |

The fixed-reference mean variance is much higher for prioritized arms than controls
on every maze, while final knowledge is lower. This is an association, not a causal
mediation test. Exact-zero shares and medians can be small even with a large upper
tail; per-seed p25/p50/p75/p90/p99 and maxima are averaged across seeds, rather than
called pooled quantiles. `distributions.csv` separately pools raw histogram counts,
including exact zeros and both the original equal-width bins and logarithmic bins,
for every point and all three blocks. Those distinguish the observed ranges that
the old first equal-width bin combined.

Training statistics below are cumulative through the final point, summarized per
seed. They describe the proposed and accepted sampling distributions, not the fixed
evaluation set. The bound B remains 250000 throughout. Uniform controls have exactly
one proposal per draw, acceptance one, zero priority value queries, no acceptance
coins and no measured training variance; missing variance is never replaced by zero.

| Maze | Arm | Proposal variance ± SE | Accepted variance ± SE | Acceptance ± SE | Proposals/draw ± SE |
| --- | --- | --- | --- | --- | --- |
| Maze4 | V2 | 8486.56 ± 465.60 | 52.33 ± 0.82 | 0.947496 ± 0.002370 | 1.055539 ± 0.002644 |
| Maze4 | V4 | 13528.71 ± 497.11 | 92.15 ± 1.10 | 0.903927 ± 0.002729 | 1.106476 ± 0.003334 |
| Maze5 | V2 | 14000.50 ± 566.63 | 70.37 ± 0.91 | 0.918494 ± 0.002608 | 1.088905 ± 0.003089 |
| Maze5 | V4 | 20753.24 ± 345.40 | 136.91 ± 1.54 | 0.853778 ± 0.001668 | 1.171350 ± 0.002280 |
| Maze6 | V2 | 14688.48 ± 421.14 | 81.07 ± 1.40 | 0.911803 ± 0.002213 | 1.096851 ± 0.002671 |
| Maze6 | V4 | 22548.43 ± 637.37 | 153.28 ± 3.03 | 0.840597 ± 0.003517 | 1.190027 ± 0.004993 |
| Maze7 | V2 | 16192.17 ± 624.72 | 82.61 ± 1.95 | 0.903858 ± 0.002940 | 1.106593 ± 0.003639 |
| Maze7 | V4 | 23212.43 ± 266.95 | 158.35 ± 3.49 | 0.833673 ± 0.001711 | 1.199607 ± 0.002464 |
| Woods14 | V2 | 8420.90 ± 512.55 | 67.19 ± 4.28 | 0.940607 ± 0.003197 | 1.063380 ± 0.003674 |
| Woods14 | V4 | 11366.45 ± 673.04 | 122.93 ± 4.06 | 0.898289 ± 0.005054 | 1.113922 ± 0.006499 |
| MazeMA | V2 | 5046.62 ± 404.46 | 55.96 ± 3.54 | 0.958698 ± 0.002957 | 1.043271 ± 0.003232 |
| MazeMA | V4 | 7353.04 ± 444.90 | 94.64 ± 3.77 | 0.933955 ± 0.003225 | 1.070960 ± 0.003734 |

Every run has three full replay steps per actual interaction and zero values-only
steps. The independent match identity remains `steps + 3*full + 2*values_only +
priority_value_queries`, so controls have exactly `10*steps` formations. Prioritized
arms add K queries per proposal. Objective scoring, selection and provenance calls,
accepted/proposal route composition, realized relabeled share, full/value-only counts
and classifier-perception tests are retained separately at every point. Selection
objective calls, failed draws and fallback counts are zero for these future-goal
arms. Prioritization is unweighted and changes the realized copy share; it does not
increase the number of applied learning steps.

### Completed allocations, population memory and elapsed time

All allocations use one CPU on `lem-cpu-normal`. These CPUTimeRAW totals include
every attempted grid run, including the seven initial incomplete attempts:

| Maze | V2, CPU-s | V4, CPU-s | U2, CPU-s | U4, CPU-s |
| --- | --- | --- | --- | --- |
| Maze4 | 7881 | 10420 | 3731 | 4109 |
| Maze5 | 9841 | 17198 | 5964 | 7179 |
| Maze6 | 11620 | 20515 | 6713 | 7866 |
| Maze7 | 11559 | 20726 | 7005 | 8241 |
| Woods14 | 2836 | 5279 | 1683 | 1984 |
| MazeMA | 1253 | 2259 | 880 | 1059 |

The grid used **177801 CPU-seconds**; all pilot attempts used **16041**. The total
is **193842 seconds = 53.845 CPU-hours**, within the approved 110, with no outstanding
reservations. Selected grid trajectories account for 175735 seconds and incomplete
attempts for 2066 seconds. The only recovery is the seven-seed exception described
above; all other registered twenty-job batches completed without retries.
The initial expected grid cost was 50.830212 hours; twice it plus the pilot was
106.116258 hours, versus a first-attempt hard bound including the pilot of 108.1225.
The approved recovery admission bound was 108.854167 hours. Forecasts and hard
reservations are not reported as actual usage.

A read-only completeness query on 2026-10-17 listed all allocations of alelys2099
since 2026-10-10 whose job names start with `tu-`, expanding array tasks. All 1655
returned allocations across 99 arrays map to exactly one archived ledger entry;
there are no unmapped allocations or unmapped CPU-seconds. The pair-baseline study
accounts for 24735 seconds including both pilots, the hindsight study for 124605
seconds (34.6125 hours), and this study for 193842 seconds (53.845 hours), comprising
16041 pilot and 177801 grid seconds. Twenty pilot tasks cancelled before allocation
are absent from the allocation-only query; their archived accounting records each
confirm CANCELLED, zero allocated CPUs and zero CPUTimeRAW. The batch-root admission
rule prevents new out-of-scope ledgers, the admission-log check audits the recorded
submission scope, and this independent allocation inventory checks its completeness.

The following are final-point or completed-process means ± SE for the twenty
selected runs per cell. Process CPU (user plus system), training wall time and
evaluation wall time are separate. The known agent/measurement memory lower bound
counts all heads and the shared store once; it excludes unfilled capacities and
allocator overhead. Peak RSS is a process measurement and includes other storage.
Transient accepted-batch peaks, per-head known bytes and population/store payloads
remain separate fields. Sampled training CPU in `run-costs.csv` is an estimate,
distinct from the completed-process CPU and allocation accounting.

| Maze | Arm | Process CPU s ± SE | Train wall s ± SE | Evaluation wall s ± SE | Known agent+measurement MiB ± SE | Peak RSS MiB ± SE |
| --- | --- | --- | --- | --- | --- | --- |
| Maze4 | V2 | 284.289 ± 8.635 | 288.471 ± 8.701 | 0.915 ± 0.021 | 1.090 ± 0.014 | 8.445 ± 0.069 |
| Maze4 | V4 | 510.489 ± 7.356 | 517.976 ± 7.481 | 1.820 ± 0.032 | 1.741 ± 0.017 | 8.693 ± 0.049 |
| Maze4 | U2 | 182.334 ± 5.101 | 184.320 ± 5.175 | 0.977 ± 0.018 | 1.140 ± 0.010 | 8.198 ± 0.014 |
| Maze4 | U4 | 201.315 ± 5.742 | 202.652 ± 5.819 | 1.992 ± 0.049 | 1.842 ± 0.020 | 8.979 ± 0.047 |
| Maze5 | V2 | 483.253 ± 8.007 | 487.848 ± 8.167 | 3.199 ± 0.054 | 1.378 ± 0.015 | 8.494 ± 0.009 |
| Maze5 | V4 | 844.421 ± 13.073 | 852.952 ± 13.543 | 6.303 ± 0.104 | 2.394 ± 0.019 | 10.130 ± 0.046 |
| Maze5 | U2 | 291.966 ± 11.124 | 293.172 ± 11.336 | 3.621 ± 0.087 | 1.479 ± 0.016 | 8.596 ± 0.025 |
| Maze5 | U4 | 349.213 ± 6.054 | 349.153 ± 6.187 | 8.265 ± 0.122 | 2.786 ± 0.023 | 11.177 ± 0.045 |
| Maze6 | V2 | 567.069 ± 11.025 | 575.536 ± 11.444 | 4.362 ± 0.081 | 1.475 ± 0.015 | 8.758 ± 0.035 |
| Maze6 | V4 | 1000.682 ± 12.734 | 1014.159 ± 13.615 | 8.608 ± 0.124 | 2.594 ± 0.024 | 10.982 ± 0.077 |
| Maze6 | U2 | 327.536 ± 7.877 | 329.546 ± 8.087 | 4.874 ± 0.082 | 1.630 ± 0.018 | 9.167 ± 0.045 |
| Maze6 | U4 | 384.448 ± 9.996 | 382.373 ± 10.275 | 9.930 ± 0.163 | 2.989 ± 0.022 | 12.000 ± 0.053 |
| Maze7 | V2 | 565.323 ± 9.796 | 572.623 ± 10.127 | 4.127 ± 0.082 | 1.533 ± 0.017 | 8.795 ± 0.045 |
| Maze7 | V4 | 1014.432 ± 18.391 | 1027.233 ± 19.498 | 8.185 ± 0.156 | 2.710 ± 0.030 | 11.103 ± 0.076 |
| Maze7 | U2 | 341.941 ± 7.799 | 344.394 ± 8.122 | 4.690 ± 0.085 | 1.679 ± 0.015 | 9.158 ± 0.050 |
| Maze7 | U4 | 402.205 ± 7.604 | 400.550 ± 7.807 | 10.124 ± 0.147 | 3.174 ± 0.027 | 12.262 ± 0.041 |
| Woods14 | V2 | 137.644 ± 3.743 | 140.095 ± 3.857 | 0.292 ± 0.007 | 0.555 ± 0.005 | 8.180 ± 0.008 |
| Woods14 | V4 | 254.590 ± 8.316 | 258.694 ± 8.694 | 0.616 ± 0.016 | 0.808 ± 0.009 | 8.178 ± 0.008 |
| Woods14 | U2 | 81.713 ± 2.043 | 82.914 ± 2.151 | 0.287 ± 0.007 | 0.554 ± 0.005 | 8.174 ± 0.007 |
| Woods14 | U4 | 95.997 ± 1.680 | 97.096 ± 1.754 | 0.618 ± 0.010 | 0.791 ± 0.007 | 8.175 ± 0.008 |
| MazeMA | V2 | 60.589 ± 1.245 | 61.623 ± 1.310 | 0.087 ± 0.002 | 0.560 ± 0.004 | 8.138 ± 0.012 |
| MazeMA | V4 | 109.701 ± 2.362 | 111.458 ± 2.456 | 0.191 ± 0.004 | 0.786 ± 0.007 | 8.129 ± 0.016 |
| MazeMA | U2 | 41.484 ± 1.085 | 42.333 ± 1.116 | 0.094 ± 0.002 | 0.566 ± 0.004 | 8.142 ± 0.013 |
| MazeMA | U4 | 50.692 ± 1.268 | 51.660 ± 1.301 | 0.199 ± 0.004 | 0.771 ± 0.008 | 8.132 ± 0.013 |

Raw data, binary, manifests, every allocation/attempt, failed prefixes, grant and
test-only receipts, approvals and frozen verifier are retained in two verified copies:
`~/tu_runs/7/approved-value-consistency-study/` on WCSS and the matching directory
beneath `~/Desktop/tu-runs/7/` locally. The raw manifest lists **6254 files and
530672932 bytes**, excluding itself; its SHA-256 is
`861b6bce501ac4422bf5e69c6b5f8ad97960bb1911ccba6b0ce7b1567f439948`.
Every listed hash and the inventory agree, and independent local and remote full
audits return the same verification result. Frozen verification operations are from
`6f9a9c3691dd96dd2fb4faf6d480697b989f929e`; node execution operations remain at
`f293a07680044a9b6cb116c8cc5656d7b156233e`.

Derived tables, figures, gate/mutation evidence and the Polish report are archived
under `~/tu_runs/7/value-consistency-results/` and the corresponding local results
directory beneath `~/Desktop/tu-runs/7/`. The analysis exports 96 primary summaries,
108 paired comparisons, 96 exact crossings, 7840 checkpoint fractions and 1920 seed
scores. `validate_tables.py` independently reconstructs their rational arithmetic and
cost scope from integer head-success counts. Static SVG/PNG files show test curves,
common disagreement, exact-zero shares and both sets of paired intervals; plotting
software is recorded separately and does not enter measurement. No convergence,
unregistered temperature optimum or superiority of all ensembles is claimed.

## Clippy — the determinism invariants, checked by machine

`clippy.toml` at the workspace root turns two claims this document makes into lints:

- **No `HashMap` or `HashSet` anywhere.** Their iteration order is randomized per process.
  "No container's iteration order became load-bearing" (checkpointing, above) is now checked
  rather than asserted.
- **No float function whose precision Rust documents as non-deterministic** — for `f64` and
  `f32`: `powi`, `powf`, the exponentials and logarithms, `cbrt`, `hypot`, and the
  trigonometric and hyperbolic families. The standard library states that their precision
  "varies by platform, Rust version, and can even differ within the same execution"; most
  call the platform libm. `sqrt` and `mul_add` are documented as correctly rounded and stay
  allowed. Squares are written as products.

The file affects `cargo clippy` only; no build, including the cluster's musl build, reads it. An
exception is made locally and in the open — `#[expect(clippy::disallowed_methods, reason = "…")]`
on code that feeds neither learning nor a reported number.

```bash
cargo clean --release -p acs2-core -p acs2-envs -p acs2-bench -p acs2-measure -p acs2-trajectory -p acs2-agents
cargo clippy --workspace --all-targets --release
```

**Clean first.** Cargo replays cached diagnostics for crates that did not change, and it does
not treat `clippy.toml` as an input. The first version of this section claimed zero violations
from a run that had only re-linted a freshly planted file; the rest was replayed from a cache
built before the rules existed. A run from clean on 2026-09-26 finds 22 warnings, none in code
added by the goal line:

- **5 violations**, all in files that predate the rules. `acs2-core/src/learning/ga.rs:65` squares quality with
  `powi(3)` on the learning path, and `acs2-bench/src/main.rs:99` and `acs2-bench/src/bin/mpx.rs:202` compute
  standard deviations with `powi(2)` — in P9 that is column 4, one of the columns compared
  byte for byte across machines. Bit-identity between the M1 and Bem2 has been measured on runs
  that use them, not guaranteed; LLVM lowers a small constant `powi` to multiplications, and
  writing them explicitly would be bit-identical under that lowering, but these files belong to
  the measured path of the multiplexer line. `acs2-envs/tests/performance_geometry.rs` uses a
  `HashSet` for membership only, never iterating it.
- **17 style and complexity warnings** — `acs2-core/src/learning/ga.rs`, the nine-argument `apply_alp`, `acs2-core/src/representation/mark.rs`,
  `acs2-core/src/representation/population.rs`, `acs2-envs/src/multiplexer.rs` and six test files. None concerns correctness.

All 22 stay on this branch because fixing them edits the measured path. Code added by the goal
line is held to zero.
