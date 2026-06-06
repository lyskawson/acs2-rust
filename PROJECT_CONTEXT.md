# PROJECT_CONTEXT.md — Rust ACS2 (canonical, pyalcs-based)

Master context for any fresh AI chat or coding-agent session. Paste this file first,
together with `SPEC_PYALCS.md`, before issuing build prompts.

---

## 0. Mission

The thesis goal is **a potential Actor-Critic extension of ACS2**. There is currently
no ACS2 implementation owned by the author. The supervisor proposed building one in
**Rust** and measuring how much faster it is than a Python reference, to justify
committing to Rust before extending it.

**This project phase delivers ONLY the baseline: a correct, idiomatic Rust
implementation of canonical ACS2, plus a like-for-like speed comparison against a
Python reference.** Actor-Critic is NOT implemented in this phase. The architecture
must leave clean seams for it (see §7).

---

## 1. Decisions already made (do not re-litigate)

1. **Port target = pyalcs** (`hendrykik/acs2vcp-python`, the `lcs/agents/acs2` core),
   NOT the supervisor's ALCS `cpu_single`/`*CPU3` backend.
   - Rationale: pyalcs is canonical ACS2 (Butz & Stolzmann), uses standard 8-sensor
     wall perception, has clean Actor-Critic attachment points (pluggable
     `action_selector`; the RL bootstrap argument is the single critic hook) and a
     working precedent for a value-head extension (ACS2VCP). It also ships real unit
     tests usable as a porting oracle.
   - The ALCS `cpu_single` backend was rejected as a foundation because: its
     per-phase epsilon/beta schedule is a dead write (epsilon is constant 0.1, the
     0.8/0.2/0.0 schedule never applies), its perception is `(row,col)` coordinates
     not sensor patterns, and it has no parametric policy / value function to attach
     an actor-critic to. These quirks are fine for reproducing its own paper numbers,
     but hostile as an AC foundation.

2. **No Actor-Critic in this phase.** Baseline first. Architecture leaves seams.

3. **Modern stack.**
   - The Rust environment is written natively and follows **Gymnasium** conventions
     (`reset() -> obs`; `step() -> (obs, reward, terminated, truncated, info)`).
   - The Python baseline (pyalcs) runs **as-is on its pinned `gym==0.23.0`** inside an
     isolated virtual environment. Do NOT modernize or edit the baseline — it is the
     thing being measured.
   - Optional, later: PyO3 bindings exposing the Rust agent to a Gymnasium env.

4. **Maze suite = the canonical mazes already present in pyalcs `gym_maze`.** No need
   for the supervisor's `.cpp` files. Rust replicates the same maze geometries and the
   same 8-sensor perception; the Python baseline runs the same maze ids. This keeps
   the timing comparison apples-to-apples.

---

## 2. Source-of-truth hierarchy

1. **`SPEC_PYALCS.md`** (the reverse-engineered specification) is the primary spec.
2. **pyalcs source code** resolves any ambiguity the spec leaves open.
3. **Canonical ACS2 (Butz & Stolzmann)** is consulted only to interpret intent; it
   never overrides what the code/spec actually does.
4. The ALCS `cpu_single` backend and `SPEC_ALCS_CPUSINGLE.md` are IGNORED for
   implementation. They may be used later only as an OPTIONAL secondary timing
   datapoint, reproduced with its quirks intact, never as the algorithm to build on.

---

## 3. Scope

**In scope (this phase):**
- Condition / Effect / Mark / Classifier, fitness = q·r.
- Population with match-set and action-set formation, numerosity, subsumption.
- ALP (expected / unexpected / covering / marking), reliability/inadequacy handling.
- Reinforcement learning update (r, ir) with the match-set-max-fitness bootstrap.
- GA (generalizing) — present but OFF in the benchmark protocol (see §5); implement it
  so it can be toggled, since the GA path is part of canonical ACS2.
- Epsilon-greedy exploration and best-action exploitation.
- Maze environment (8-sensor perception) + maze geometry data.
- A benchmark runner producing per-maze metrics + timing, and a comparison against the
  pyalcs baseline run on the same mazes/protocol.

**Out of scope (this phase):**
- Actor-Critic (leave seams only).
- Probability-Enhanced Effects (PEE) — OFF by default in pyalcs; do not implement now.
- Action planning — OFF by default; do not implement now.
- GPU/tensor backends.
- Multiplexer / parity / Gymnasium-wrapped non-maze environments (later, if needed).

---

## 4. Canonical-ACS2 facts the implementation must honor (from SPEC_PYALCS)

- **Wildcard symbol** is `#`. Condition matches if either side is wildcard.
- **Effect**: a wildcard effect attribute means "no change anticipated at this
  position"; a specified effect symbol anticipates a change to exactly that value.
- **Mark**: one set per attribute; records perceived values at still-unspecified
  positions when the classifier anticipated incorrectly.
- **fitness = q · r** (default; no custom fitness function).
- **q updates**: correct → `q += beta*(1-q)`; incorrect → `q -= beta*q`.
- **r/ir updates**: `P = reward + gamma*max_fitness`; `r += beta*(P-r)`;
  `ir += beta*(reward-ir)`. `max_fitness` is the max q·r over change-anticipating
  match-set classifiers, or 0 at episode end.
- **tav (application average)**: MAM rule — running mean while `exp < 1/beta`, then
  exponential at rate beta.
- **Reliable** iff `q > theta_r` (0.9); **inadequate** iff `q < theta_i` (0.1).
- **ALP deletion**: only incorrectly-anticipating AND inadequate classifiers are
  removed (from population, match set, action set).
- **Subsumption is invoked inside ALP unconditionally** in pyalcs, but in the chosen
  protocol GA is off, so reproduce pyalcs behavior faithfully (see §5 / SPEC D.5).
- **u_max default 100000 disables ALP generalization** — under the shipped config, ALP
  only specializes. Replicate this (do not silently "fix" it) for parity.

### Known pyalcs hazards — DO NOT transcribe these into Rust
The spec flags three real bugs in the orchestration/extension layer. Re-derive the
trial loop cleanly in Rust instead of copying it:
- `_is_preferred_to_delete` checks `is_marked` (the bound method, always truthy).
- `ClassifiersList.copy()` is broken (copies `__slots__`, not the classifiers).
- `ACS2VCP._run_trial_exploit` raises `TypeError`.
Also: exploitation ignores epsilon (always BestAction) — a property of the algorithm,
keep it; just be aware the epsilon values in the exploit scripts are inert.

---

## 5. The benchmark protocol (run IDENTICALLY on Rust and pyalcs)

The comparison is only valid if both run the same algorithm, same mazes, same protocol.
Default protocol (the pyalcs shipped ACS2 maze-script configuration):

- `classifier_length = 8`, `number_of_possible_actions = 8`.
- `beta = 0.05`, `gamma = 0.95`, `theta_i = 0.1`, `theta_r = 0.9`,
  `theta_exp = 20`, `theta_as = 20`, `mu = 0.3`, `chi = 0.8`, `u_max = 100000`.
- `do_ga = False`, `do_pee = False`, `do_action_planning = False`,
  `do_subsumption = True` (affects only the GA path, so effectively inert with GA off).
- Phases: explore 500 episodes (epsilon = 0.8), then exploit (epsilon does not affect
  exploit). Fix a single trial schedule and apply it to BOTH implementations.
- `n_exp` independent repetitions (use 10 to match the spirit of the paper).
- Seed BOTH `random` and `numpy.random` in the Python baseline; inject a seeded RNG in
  Rust. Reproducibility is best-effort; the comparison is about wall-clock + learning
  quality, not bit-identical trajectories.

**GA is a runtime config flag (`do_ga`), not a hardcoded choice.** GA is implemented
fully (it is part of canonical ACS2) but defaults to **OFF**. The flag must be wired
identically on both sides — the Rust bench (`acs2-bench`) and the pyalcs baseline
(`baseline/run_pyalcs_maze.py`) — so the exact same value can be set on both. Rules:
- Whatever value is chosen, it MUST be identical on both implementations for any single
  comparison run; otherwise the benchmark compares two different algorithms, not two
  languages.
- **GA OFF** matches the tested pyalcs maze scripts and avoids the buggy pyalcs GA
  deletion. **GA ON** gives a configuration closer to the supervisor's paper (GA on in
  the explore phase); if enabled, the pyalcs GA bugs (PROJECT_CONTEXT §4) are a known
  semantic gap.
- **Establish correctness parity (P8) and the first benchmark (P9) with GA OFF.** GA is
  an extra stochastic mechanism, so with GA ON the two RNG streams diverge faster and
  deterministic differential testing is harder. Treat GA ON as a separate later run on
  an already-validated core, never as part of the initial parity work.

### Maze environment semantics (pyalcs gym_maze)
- Perception = 8 neighbour cells in fixed order **N, NE, E, SE, S, SW, W, NW**,
  stringified.
- Cell codes: path `0`, wall `1`, reward `9` (agent's own cell never appears in its own
  view). Perceptions carry `'0'`, `'1'`, `'9'`.
- 8 actions = 8 compass moves; a move into a wall leaves the agent in place.
- **Reward = 1000 on entering the reward cell (also sets done), 0 on every other step.**
  (Note: this differs from the ALCS backend's -1 step penalty — use 0.)
- Reset = agent placed at a uniformly random path cell.
- Episode length cap is enforced by the runner (use the paper's 200-step cap).

---

## 6. Validation contract

- **Correctness is a gate, not a target.** The Rust agent must learn the mazes to
  steps-to-goal of the same order as the Python baseline on the same mazes/protocol.
  A wildly different steps value (e.g. 6 vs 60) signals a different algorithm, not a
  faster one.
- **Differential testing** against pyalcs at the component level (match set, ALP
  outcome on a single transition, population after one learning step) must agree
  exactly on deterministic inputs (no RNG). Divergence is acceptable only where RNG
  enters.
- **Oracle**: the pyalcs unit tests
  (`tests/lcs/agents/acs2/test_{Classifier,ClassifierList,Effect,alp}.py`,
  `tests/lcs/strategies/*`) are reusable as Rust test vectors.
- **Timing**: measured on the SAME machine, SAME mazes, SAME protocol. Report
  `t_python / t_rust` per maze and overall. Do NOT compare against the paper's
  published times (different hardware).

---

## 7. Pinned crate architecture (Rust workspace)

Clean Architecture: domain (`acs2-core`) has zero I/O and zero environment knowledge.
SOLID throughout. No God types. RNG injected via a trait (Dependency Inversion).
No code comments — semantics live in names. All identifiers in English.

```
acs2/                      (Cargo workspace)
├── acs2-core/   (lib)     pure ACS2 domain, no I/O
│   ├── symbol              wildcard-aware attribute symbol
│   ├── condition           matching template + specificity + subsumes
│   ├── effect              anticipated change + does_anticipate_correctly
│   ├── mark                per-attribute mark sets + get_differences
│   ├── classifier          fields, q/r/ir/tav updates, fitness, copy
│   ├── population          storage, form_match_set, form_action_set,
│   │                       numerosity, subsumption, deletion
│   ├── alp                 expected_case, unexpected_case, cover, marking
│   ├── ga                  selection, crossover, generalizing mutation, deletion
│   ├── rl                  reward-prediction update + bootstrap target
│   ├── action_selection    trait ActionSelector  <-- ACTOR SEAM
│   │                       impls: EpsilonGreedy, BestAction, RandomAction
│   ├── config              Configuration (all parameters above)
│   ├── rng                 trait abstraction over rand::Rng (injected)
│   └── agent               single shared trial loop (explore / exploit)
│                           bootstrap value is a parameter  <-- CRITIC SEAM
├── acs2-envs/   (lib)     Environment trait (Gymnasium-style) + Maze
│   ├── environment         trait: reset, step -> (obs, reward, terminated, truncated)
│   ├── maze                8-sensor perception, compass actions, reward scheme
│   └── maze_data           the canonical maze geometries (ported from gym_maze)
├── acs2-bench/  (bin)     runs the suite, computes metrics, emits CSV + timing
└── acs2-py/     (lib, LATER, optional)   PyO3 bindings exposing a Gymnasium agent
```

**The two Actor-Critic seams (built now as architecture, implemented later):**
1. **Actor**: the `ActionSelector` trait. This phase ships `EpsilonGreedy`,
   `BestAction`, `RandomAction`. An actor is a future `ActionSelector` impl.
2. **Critic**: the RL bootstrap target is passed into the agent loop as a value, not
   hardcoded. This phase passes `max q·r over the next match set`. A critic later
   supplies a learned `V(s')` at the same call site.

**Representation note / future optimization.** Canonical ACS2 (pyalcs) does NOT
bit-pack; conditions are sequences of symbols. Start with a straightforward Rust
representation (e.g. `[Symbol; 8]` or a small fixed array, wildcard as a sentinel/enum
variant). Element-wise matching first, correctness-parity first. Bit-packing (the
2×u64 trick from the ALCS backend) is a LATER optimization behind the same interface,
introduced only after parity is established — it is where Rust gains the most, but it
must not compromise the initial correctness comparison.

---

## 8. Engineering constraints (author's standing rules)

- Idiomatic Rust, optimized for memory allocation and CPU; minimize per-step heap churn.
- SOLID; Clean Architecture; no God classes; single-responsibility modules.
- **No comments inside code.** Replace comments with clear names for variables and
  functions.
- All code, comments, identifiers in English.
- Inject the RNG; do not use thread-global randomness in the domain.
- Determinism: a seeded run must be repeatable within Rust.
