# PROJECT_CONTEXT.md — what this project is and what it must honour

The standing context for the project: what it is for, which rules the implementation
must not break, and the protocol every measurement runs under. Read it with
`SPEC_PYALCS.md` (the reference semantics) and `ARCHITECTURE.md` (what was built and
why). The two lines have separate records: `docs/AGENT_HANDOFF.md` for MPX scaling,
and the goal-task sections of `docs/ARCHITECTURE.md` plus `experiments/README.md`
for the thesis implementation and registered measurements.

---

## 0. Mission

The thesis is **Experience and Goal Selection Based on Trajectory Utility in the ACS2
Anticipatory Classifier System**. It develops an extension of ACS2HER that uses the
utility of stored trajectories to choose goals from the set of training tasks and
regulates the share of those experiences against HER's, with an optional conservative
value correction. The intended comparison is with ACS2, ACS2ER, ACS2HER and ACS2VCP
on discrete multi-goal mazes with sparse rewards at equal numbers of environment
interactions. Rust supports the compute-bound experiments and an implementation
whose learning mechanisms can be extended and checked independently.

`feature/trajectory-utility` is the thesis line. `feature/mpx264` is additional
multiplexer-scaling research with a long checkpointed cluster chain. They share the
core and environment crates; their runners, metrics and experiment policies stay
separate. `CLAUDE.md` defines the per-line operational boundaries.

**pyalcs is a reference, not a target.** It plays two roles and no others:

- a **correctness oracle** — the implementation is validated against it vector by
  vector (761/761 deterministic cases, zero divergence), which is what licenses any
  claim that results here describe ACS2 rather than a lookalike;
- a **performance baseline** — the like-for-like timing comparison that justified
  committing to Rust.

Neither role makes this a transcription. Where pyalcs and canonical ACS2 disagree,
the deviation is recorded and justified (§4 and the ARCHITECTURE fidelity ledger),
and mechanisms dormant in pyalcs's defaults have been activated and studied here.

**Configuration parameters and validation behavior are separate.** `Configuration`
retains the pyalcs parameter defaults: GA off and `u_max = 100000`, leaving ALP
generalization dormant. The P9 benchmark selects the pinned protocol of §5 and
`TruncationMode::Pyalcs`; the P8/P11 validation callers also select that mode
explicitly. Default learning behavior uses the corrected bootstrap on truncation.
The multiplexer work overrides parameters routinely — `u_max` swept 5 to 24,
GA on, two ALP-generalization variants, an alternative problem encoding, a varied
exploration epsilon, and a second agent (ACS2ER). Goal measurements state every
parameter explicitly rather than inheriting the validation protocol.

Actor-Critic is **not** implemented. The architecture leaves seams for it (see §7):
an `ActionSelector` trait and an injected RL bootstrap value.

---

## 1. Decisions already made (do not re-litigate)

1. **Reference = pyalcs** (`hendrykik/acs2vcp-python`, the `lcs/agents/acs2` core),
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

2. **No Actor-Critic yet.** The architecture leaves seams; see §7.

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

1. **`docs/SPEC_PYALCS.md`** is the reverse-engineered specification for reference
   parity; `CLAUDE.md` and `docs/ARCHITECTURE.md` record corrected default behavior
   and justified deviations.
2. **pyalcs source code** resolves ambiguity in the reference behavior. It is the
   validation oracle, not the definition of this implementation's defaults.
3. **Canonical ACS2 (Butz & Stolzmann)** supplies algorithmic intent. A correction
   must be documented and gated while explicit validation modes preserve parity.
4. The ALCS `cpu_single` backend is not an implementation oracle. Its secondary timing
   comparison is recorded in `docs/archive/CPU_SINGLE_FULL_COMPARISON.md`, reproduced
   with its quirks intact, never as the algorithm to build on.

---

## 3. Scope

**Implemented:**

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
- ACS2ER with uniform replay and agent-owned replay configuration.
- Multiplexer dynamics, diagnostics and checkpointed reach experiments on the MPX line.
- A goal port and adapter, multi-goal mazes, BitFlipping, HandEye and Taxi.
- Read-only goal-task measurements at equal interaction checkpoints, exact random-policy
  floors, reachability references, value diagnostics and component cost accounting.
- Raw episode storage, observed desired-goal candidates, goal-selection/admissibility
  primitives, objective-based relabeling, sampling and exact composition diagnostics.

**Thesis agents not implemented yet:** ACS2HER, the trajectory-utility agent, ACS2VCP,
and the optional conservative value correction. They will use the shared mechanisms
and the measurement contract; the registered maze results so far are a pilot, not
the final train/validation/test comparison. HandEye, Taxi and BitFlipping measurements
describe limitations outside the main maze comparison.

**Not implemented:**

- Actor-Critic (leave seams only).
- Probability-Enhanced Effects (PEE) — OFF by default in pyalcs; do not implement now.
- Action planning — OFF by default; do not implement now.
- GPU/tensor backends.
- Parity and Python/Gymnasium bindings (optional future integration).

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
The spec flags four real bugs. The first three sit in the orchestration/extension
layer; the fourth sits in a learning primitive (apply_alp itself). Re-derive the
trial loop cleanly in Rust instead of copying it, and use deferred deletion in ALP:
- `_is_preferred_to_delete` checks `is_marked` (the bound method, always truthy).
- `ClassifiersList.copy()` is broken (copies `__slots__`, not the classifiers).
- `ACS2VCP._run_trial_exploit` raises `TypeError`.
- `apply_alp` calls `action_set.safe_remove(cl)` while iterating `action_set`,
  so deleting an inadequate classifier makes the Python list iterator skip the next
  member — it is never processed (no experience bump, no mark, no anticipation
  update). Rust uses deferred deletion (iterate a cloned set, delete after the loop)
  and is the correct side per §J. Found via the P8 differential, not in the
  literature.
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
- Trial schedule (pinned, replicates the shipped pyalcs `run_acs2_maze*.py`): explore
  500 episodes (epsilon = 0.8), then THREE exploit phases of 200 episodes each.
  Exploitation always uses BestAction and ignores epsilon, so the three exploit phases
  are behaviorally identical pure-RL evaluations of the (frozen) population. The
  measured metric is mean steps-to-goal over the FINAL 200-episode exploit window.
  Apply this exact schedule to BOTH implementations.
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
- Episode length (truncation) cap is per-maze, taken from the pyalcs gym
  registration, NOT from the supervisor's paper. Shipped values: **50** for
  Maze4/5/7 and Woods1; **500** for Woods100/101/102; use each maze's registered
  `max_episode_steps`. The earlier "200" was a vestige of the abandoned CPU3/paper
  protocol and is wrong here — the source-of-truth hierarchy (§2) makes the pyalcs
  registration authoritative. Store the cap as per-maze data, identical on both the
  Rust runner and the pyalcs baseline.

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
acs2-core/src/                 pure ACS2 domain, no I/O
  representation/             symbol, perception, condition, effect, mark, classifier, population
  learning/                   ALP, GA, RL, subsumption, action selection (actor seam)
  agents/acs2/mod.rs           ACS2 exploration loop
  agents/acs2er/               ACS2ER loop, replay memory and configuration
  agents/trial.rs             LearningAgent, shared exploitation, truncation mode (critic seam)
  ports/                      environment, goal objective/environment and adapter
  support/                    configuration, injected RNG, knowledge, matching counter
  checkpoint.rs               unchanged path for the running MPX chain
acs2-envs/src/                 concrete dynamics, reference/research/performance roles
  maze/                       single-goal dynamics, topology, knowledge and geometries
  maze/geometries/pyalcs/      gym_maze provenance
  maze/geometries/alcs/        22 ALCS comparison geometries
  goal/                       multi-goal maze, bit flipping, hand-eye, taxi, knowledge
  roles.rs                    pins P9 validation order separately from research/performance sets
  multiplexer.rs              MPX dynamics and public path
acs2-bench/src/                P9 benchmark, multiplexer runners, checkpoint codec
acs2-measure/src/              goal-agent contract, interaction budget, evaluation, references, CLI
  runner.rs                   GoalAgent, CoreAgent, TrainingEnvironment, read-only evaluation
  task.rs                     task construction and explicit thesis preset
  reference.rs                start-goal distribution, random-policy floor and reachability
  trajectory.rs               relabeling and replay-diagnostics adapter
acs2-trajectory/src/           raw episodes and goals, no learning or task policy
  store.rs                    TrajectoryStore, FIFO episodes, persistent observed candidates
  selection.rs                goal strategies, weighted distributions and admissibility
  relabel.rs                  objective-derived samples and episode endings
  sampler.rs                  draw volume, fallback, provenance and cumulative route counters
  composition.rs              exact composition without RNG
  cost.rs                     reward/reach evaluations split by purpose
experiments/                   registered protocols and analyses; raw data outside the checkout
  baseline-pilot/             ACS2/ACS2ER pilot grid and its original operations
  relabeling-composition/     random-policy composition summary
```

The workspace has these five crates; optional Python bindings are not a crate yet.
Every former `acs2_core` crate-root module path remains available through re-exports
in `acs2-core/src/lib.rs`. New code uses the grouped paths, and compatibility exports
are removed only after the running MPX chain ends and both lines' gates pass.
New learning agents go in a separate crate outside core (planned package name
`acs2-agents`), one module directory per family with variants inside. That package
is not implemented yet. Shared mechanisms remain in the shared crates.

**The two Actor-Critic seams (built now as architecture, implemented later):**

1. **Actor**: the `ActionSelector` trait. The core provides `EpsilonGreedy`,
   `BestAction`, `RandomAction`. An actor is a future `ActionSelector` impl.
2. **Critic**: the RL bootstrap target is passed into the agent loop as a value, not
   hardcoded. The implemented agents pass `max q·r over the next match set`. A critic later
   supplies a learned `V(s')` at the same call site.

**Representation note / future optimization.** Canonical ACS2 (pyalcs) does NOT
bit-pack; conditions are sequences of symbols. Use a stack-allocated fixed-length
array parameterized by a **const generic** length `[Symbol; N]` (wildcard as a
sentinel/enum variant), so acs2-core stays independent of any environment dimension;
the maze environment fixes `N = 8` at the edge. Do NOT hardcode `8` inside acs2-core.
Element-wise matching first, correctness-parity first. Bit-packing (the 2×u64 trick
from the ALCS backend) is a LATER optimization behind the same interface, introduced
only after parity is established — it is where Rust gains the most, but it must not
compromise the initial correctness comparison.

---

## 8. Engineering constraints (author's standing rules)

- Idiomatic Rust, optimized for memory allocation and CPU; minimize per-step heap churn.
- SOLID; Clean Architecture; no God classes; single-responsibility modules.
- **No comments inside code.** Replace comments with clear names for variables and
  functions.
- All code, comments, identifiers in English.
- Inject the RNG; do not use thread-global randomness in the domain.
- Determinism: a seeded run must be repeatable within Rust.
