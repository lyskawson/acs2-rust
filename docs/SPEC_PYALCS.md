# SPEC_PYALCS.md — Canonical ACS2 Agent (pyalcs implementation)

Language-agnostic specification of the **ACS2** agent as actually implemented in
this repository (`hendrykik/acs2vcp-python`). This is a reverse-engineering of the
code, not a restatement of textbook ACS2. Where the code diverges from Butz &
Stolzmann's canonical ACS2, the code's behavior is what is documented here.

> **Conventions**
> - Every factual claim cites `file:line` relative to `pyalcs/` unless prefixed
>   `openai-envs/`.
> - Claims not directly shown by code are tagged **INFERRED**.
> - The running agent uses the `acs2.*` subclasses (`acs2.Classifier`,
>   `acs2.ClassifiersList`, `acs2.Effect`), which override several `acs.*` base
>   methods. Citations point at the method that **actually executes**. Methods that
>   exist only in the base and are inherited unchanged are cited in `acs/`.
> - Pseudocode is language-neutral; no source code is reproduced beyond short
>   formulas.

---

## A. Classifier data structure

### A.1 Fields (the `acs2.Classifier`, which is what runs)

Declared in `lcs/agents/acs2/Classifier.py:16-17` (`__slots__`) and initialized in
`Classifier.py:19-62`.

| Field | Type | Initial value | Meaning / source |
|-------|------|---------------|------------------|
| `condition` | `acs.Condition` (tuple of `str`/`dict`) | all-wildcard of length `classifier_length` | matching template (`Classifier.py:45`) |
| `action` | `int` or `None` | `None` | discrete action id (`Classifier.py:46`) |
| `effect` | `acs2.Effect` | all-wildcard | anticipated change (`Classifier.py:47`) |
| `mark` | `acs.PMark` | list of empty `set()` per attribute | states where cl failed (`Classifier.py:49`, `acs/PMark.py:9-13`) |
| `q` | `float` | `cfg.initial_q` (default `0.5`) | quality / anticipation accuracy (`Classifier.py:50-53`, default `Configuration.py:33`) |
| `r` | `float` | `0.5` | reward prediction (`Classifier.py:24,55`) |
| `ir` | `float` | `0.0` | immediate-reward prediction (`Classifier.py:25,56`) |
| `num` | `int` | `1` | numerosity (`Classifier.py:26,57`) |
| `exp` | `int` | `1` | experience counter (`Classifier.py:27,58`) |
| `talp` | `int`/`None` | `None` | time of last ALP application (`Classifier.py:28,59`) |
| `tga` | `int` | `0` | time of last GA application (`Classifier.py:29,60`) |
| `tav` | `float` | `0.0` | application-average (mean delay between applications) (`Classifier.py:30,61`) |
| `ee` | `bool` | `False` | "enhanceable" flag for PEE merging (`Classifier.py:62`) |
| `cfg` | `Configuration` | required (raises if `None`) | shared config (`Classifier.py:33-36`) |

`fitness` is a derived property (`acs/Classifier.py:124-129`): `cfg.fitness_fcn(self)`
if a custom function is configured, else `q * r`. The default config sets
`fitness_fcn = None` (`acs/Configuration.py:30`), so **default fitness = q · r**.

> Note the covering path overrides two defaults: a covered classifier is created with
> `experience=0` and `reward=0` (`alp.py:39`), not the field defaults `1`/`0.5`.

### A.2 Condition representation (`acs/Condition.py`)

- A `Condition` is an `ImmutableSequence` of per-attribute symbols
  (`acs/Condition.py:10`, `agents/ImmutableSequence.py:4-15`).
- **Wildcard / "don't care"** is the single character `'#'`
  (`ImmutableSequence.py:6`, `WILDCARD = '#'`). It is configurable as
  `cfg.classifier_wildcard` (`acs/Configuration.py:19`) but defaults to `'#'`.
- `specificity` = count of non-wildcard attributes (`acs/Condition.py:16-24`).
- `does_match` treats a wildcard on **either** side as a match
  (`acs/Condition.py:51-70`).

### A.3 Effect representation (`acs2/Effect.py`, subclass of `acs/Effect.py`)

- An `Effect` is an `ImmutableSequence` whose attributes are either plain `str`
  symbols or `ProbabilityEnhancedAttribute` (PEA) objects (`acs2/Effect.py:11-23`).
- A wildcard `'#'` in the effect is a **pass-through** symbol: "this attribute is
  anticipated not to change" (`acs/Effect.py:18-28`).
- A non-wildcard effect symbol anticipates a **change** to that exact value
  (`acs/Effect.py:30-43`).
- `specify_change` is `True` if any attribute is non-wildcard, or any attribute is
  PEA-enhanced (`acs2/Effect.py:25-39`).

### A.4 Mark representation (`acs/PMark.py`)

- `PMark` is a `TypedList` of one `set` per attribute (`PMark.py:8-13`).
- A mark records, for each condition position left **unspecified** (wildcard), the
  perceived value(s) seen when the classifier anticipated **incorrectly**
  (`PMark.py:39-54`).
- `is_marked()` is `True` if any per-attribute set is non-empty (`PMark.py:15-21`).
- Re-marking an already-marked classifier *complements* (adds values to) only the
  already-specified positions (`PMark.py:23-37,42-45`).

### A.5 Probability-Enhanced Effects (PEE / PEA)

- Implemented (`acs2/ProbabilityEnhancedAttribute.py`, a `dict` symbol→probability)
  and wired into `Effect` and `specialize`.
- **OFF by default**: `cfg.do_pee` defaults to `False` (`acs2/Configuration.py:13`).
  All three ACS2 maze example scripts leave it unset
  (`pyalcs-experiments/scripts/ACS2/run_acs2_maze4.py:128-141`). So with the shipped
  configuration, **effects are single-symbol; no probability-enhanced attributes
  are created**. The PEE branches in `specialize` (`Classifier.py:175-178`),
  `expected_case` (`alp.py:61-62`), and `apply_alp` (`ClassifiersList.py:169-174`)
  are dead under default config.

### A.6 Feature-flag defaults (shipped configuration)

From `acs/Configuration.py` and `acs2/Configuration.py`:

| Flag | Default | Source |
|------|---------|--------|
| `do_subsumption` | `True` | `acs/Configuration.py:33` |
| `do_ga` | `False` | `acs2/Configuration.py:16` |
| `do_pee` | `False` | `acs2/Configuration.py:13` |
| `do_action_planning` | `False` | `acs2/Configuration.py:27` |
| `beta` | `0.05` | `acs/Configuration.py:36` |
| `gamma` | `0.95` | `acs2/Configuration.py:10` |
| `theta_i` (inadequacy) | `0.1` | `acs/Configuration.py:39` |
| `theta_r` (reliability) | `0.9` | `acs/Configuration.py:43` |
| `theta_exp` | `20` | `acs/Configuration.py:50` |
| `theta_as` | `20` | `acs/Configuration.py:51` |
| `theta_ga` | `100` | `acs2/Configuration.py:18` |
| `epsilon` | `0.5` | `acs/Configuration.py:47` |
| `u_max` | `100000` | `acs/Configuration.py:49` |
| `mu` | `0.3` | `acs2/Configuration.py:21` |
| `chi` | `0.8` | `acs2/Configuration.py:24` |

---

## B. Control flow — one interaction step

Reference: `_run_trial_explore` (`acs2/ACS2.py:29-132`). Within one explore episode,
each loop iteration after the first performs **learning on the previous action set**,
then selects and executes a new action. Numbered pseudocode for the steady-state
iteration:

```
1.  state ← Perception(env observation); assert len(state) == cfg.classifier_length
2.  (optional) if cfg.do_action_planning and time % action_planning_frequency == 0:
        run action-planning sub-loop  [ACS2.py:43-51, 172-289]   (off by default)
3.  match_set ← population.form_match_set(state)
        # all cl in population with cl.condition.does_match(state)   [ClassifiersList.py:44-46]
4.  if steps > 0:   # learning applies to the PREVIOUS action_set
        a. apply_alp(population, match_set, action_set, prev_state, action, state, time+steps, theta_exp, cfg)
                                                                   [ACS2.py:60-69, ClassifiersList.py:110-188]
        b. apply_reinforcement_learning(action_set, last_reward, match_set.get_maximum_fitness(), beta, gamma)
                                                                   [ACS2.py:70-76]
        c. if cfg.do_ga: apply_ga(time+steps, population, match_set, action_set, state, ...)
                                                                   [ACS2.py:77-89]   (off by default)
5.  action ← cfg.action_selector(match_set)            # EpsilonGreedy   [ACS2.py:91]
6.  action_set ← match_set.form_action_set(action)
        # all cl in match_set with cl.action == action   [ClassifiersList.py:48-50]
7.  prev_state ← state
8.  (raw_state, last_reward, done, _) ← env.step(action)        [ACS2.py:96]
9.  state ← Perception(raw_state)
10. if done: run terminal learning (apply_alp with empty match_set, RL with bootstrap 0, optional GA)
                                                                   [ACS2.py:99-128]
11. steps ← steps + 1
```

**Covering** is not a separate pre-step; it happens *inside* `apply_alp`: if no
classifier in the action set anticipated correctly (`was_expected_case == False`),
a covering classifier is created via `cover(...)` and added (`ClassifiersList.py:176-179`).
See D.4.

**Match-set formation** is a full linear scan of the population
(`ClassifiersList.py:44-46`). **Action-set formation** filters the match set by exact
action equality (`ClassifiersList.py:48-50`).

`_run_trial_exploit` (`ACS2.py:134-170`) is simpler: it forms the match set, applies
**only reinforcement learning** to the previous action set (no ALP, no GA, no
covering — *no model learning during exploitation*), and always selects the best
action via `BestAction` regardless of `epsilon` (`ACS2.py:158`).

---

## C. Update formulas

Let β = `cfg.beta`, γ = `cfg.gamma`.

### C.1 Quality (q) — `acs2/Classifier.py:136-142`
```
increase_quality:  q ← q + β·(1 − q)         (== (1−β)·q + β)
decrease_quality:  q ← q − β·q               (== (1−β)·q)
```
(The `acs2` overrides are algebraically identical to the `acs` base versions
`acs/Classifier.py:207-211`.)

A reverse operation exists for PEE merging
(`reverse_increase_quality`, `Classifier.py:217-218`):
```
q ← (q − β) / (1 − β)
```

### C.2 Reward prediction (r) and immediate reward (ir) — `reinforcement_learning.py:25-54`
```
P  ← step_reward + γ · max_fitness
r  ← r + β·(P − r)
ir ← ir + β·(step_reward − ir)
```
where `max_fitness` is the bootstrap term passed in. In explore/exploit it is
`match_set.get_maximum_fitness()`, i.e. the **maximum fitness (q·r) over match-set
classifiers that anticipate a change** (`acs/ClassifiersList.py:21-38`), or `0` at
episode end (`ACS2.py:73,113,165`). This bootstraps on q·r, *not* on a pure
reward-prediction term — see J.

> `bucket_brigade_update*` and `simple_q_learning` also exist
> (`reinforcement_learning.py:3-22,57-61`) but are **not called** by the ACS2 agent.

### C.3 Application average (tav) — `acs2/Classifier.py:236-255`
Uses the "Moyenne Adaptive Modifiée" (MAM) rule. With `last_applied = time − talp − tav`:
```
if exp < 1/β:   tav ← tav + last_applied / exp
else:           tav ← tav + β · last_applied
talp ← time
```

### C.4 Thresholds — `acs/Classifier.py:201-205`
```
is_reliable()   :  q > cfg.theta_r        (default 0.9)
is_inadequate() :  q < cfg.theta_i        (default 0.1)
```

---

## D. ALP (Anticipatory Learning Process)

Orchestrated by `ClassifiersList.apply_alp` (`acs2/ClassifiersList.py:110-188`).
For every classifier in the (previous) action set:
```
cl.increase_experience()                              [ClassifiersList.py:147]
cl.update_application_average(time)                   [ClassifiersList.py:148]
if cl.does_anticipate_correctly(p0, p1):              [ClassifiersList.py:150]
    new_cl ← expected_case(cl, p0, time);  was_expected_case ← True
else:
    new_cl ← unexpected_case(cl, p0, p1, time)
    if cl.is_inadequate():                            [ClassifiersList.py:156-163]
        remove cl from population, match_set, action_set
if new_cl is not None:
    new_cl.tga ← time;  alp.add_classifier(new_cl, action_set, new_list, theta_exp)
```
After the loop: optional PEE enhancement check (off by default), then **covering**
if nothing anticipated correctly, then `new_list` is merged into the action set,
the population, and (matching members) the match set (`ClassifiersList.py:181-188`).

`does_anticipate_correctly` (`acs/Classifier.py:272-297` → `acs/Effect.py:90-92`):
for each attribute, a wildcard effect requires `p0 == p1`; a specified effect
requires `p0 != p1` **and** the effect symbol `== p1` (`acs/Effect.py:30-43`).

### D.1 Marking — `acs2/Classifier.py:220-234`, `acs/PMark.py:39-54`
On the unexpected case, `set_mark(p0)` records the perceived value at every condition
position that is still a wildcard (or complements already-marked positions). In the
`acs2` override, a successful (re)mark resets `ee ← False` (`Classifier.py:233-234`).

### D.2 Expected case (specialization toward the mark) — `alp.py:48-101`
```
diff ← cl.mark.get_differences(p0)                    [PMark.py:56-85]
if diff.specificity == 0:
    (if do_pee and marked: ee ← True);  cl.increase_quality();  return None
child ← copy_from(cl, time)
# u_max-based generalization of over-specified classifiers:
#   with default u_max = 100000 these while-loops never execute  [alp.py:78-94]
child.condition.specialize_with_condition(diff)       [alp.py:96]
if child.q < 0.5: child.q ← 0.5
return child
```
`get_differences` (`PMark.py:56-85`): if any marked position holds a value `p0` has
*not* shown there (`nr1>0`), it specializes **one randomly chosen** such position;
otherwise, if any position has `>1` recorded values (`nr2>0`), it specializes **all**
such positions.

> **Consequence (default config):** because `u_max = 100000`, the expected-case
> generalization branches are unreachable; ALP only *specializes*. With GA off in the
> example scripts, the model gains no generalization pressure at all (see J).

### D.3 Unexpected case (effect specialization) — `alp.py:104-138`
```
cl.decrease_quality();  cl.set_mark(p0)
if not cl.effect.is_specializable(p0, p1): return None   [Effect.py:45-67/83-87]
child ← copy_from(cl, time)
child.specialize(p0, p1, leave_specialized = not do_pee) [alp.py:127-133]
if child.q < 0.5: child.q ← 0.5
return child
```
`specialize` (`acs2/Classifier.py:148-180`): for each position where `p0 != p1`, if
the effect attribute is wildcard set effect←p1 and condition←p0; (PEE branch adds a
symbol to a `ProbabilityEnhancedAttribute` when already specialized — off by default).

### D.4 ALP covering — `alp.py:8-45`
Triggered only when no action-set classifier anticipated correctly
(`ClassifiersList.py:176-179`). Creates a fresh classifier with the executed
`action`, `experience=0`, `reward=0` (`alp.py:39`), sets `tga=talp=time`, then
`specialize(p0,p1)` so it anticipates the observed change. (A code comment notes this
diverges from the original C++, which used defaults `exp=1`, `r=0.5` — `alp.py:36-38`.)

### D.5 Insertion / subsumption within ALP — `anticipatory_learning_process.py:4-48`
`add_classifier(child, population, new_list, theta_exp)`:
```
find most-general subsumer of child in population (does_subsume)   [aLP.py:26-31]
else find an identical (==) classifier in new_list, then population
if found: found.increase_quality()
else:     new_list.append(child)
```
**Subsumption is invoked here unconditionally** (`anticipatory_learning_process.py:1,28`),
**independent of `cfg.do_subsumption`** — that flag gates only the GA path (see F).

### D.6 Deletion in ALP
A classifier that anticipated incorrectly **and** is inadequate (`q < theta_i`) is
removed from population, match set, and action set (`ClassifiersList.py:156-163`).
This is the only ALP-driven deletion.

---

## E. GA (genetic generalization)

`ClassifiersList.apply_ga` (`acs2/ClassifiersList.py:190-241`); helpers in
`lcs/strategies/genetic_algorithms.py`. **Off by default** (`do_ga=False`) and
**not enabled by any ACS2 maze script**.

### E.1 Trigger — `genetic_algorithms.py:10-44`
GA fires on an action set when the numerosity-weighted mean of the classifiers'
`tga` lags the current time by more than `theta_ga`:
```
apply iff  time − (Σ tga·num / Σ num) > theta_ga        (default theta_ga = 100)
```
On firing, all action-set `tga` are set to the current epoch (`set_timestamps`,
`genetic_algorithms.py:47-63`).

### E.2 Parent selection — `genetic_algorithms.py:65-86,273-282`
Roulette-wheel selection of **two** parents, weighted by `q³ · num`
(`ClassifiersList.py:207-208`). The two draws are independent (a parent may be
selected twice).

### E.3 Crossover & mutation — `ClassifiersList.py:210-228`
```
child1, child2 ← copy_from(parent1/parent2, time)
generalizing_mutation(child1, mu);  generalizing_mutation(child2, mu)
if random() < chi and child1.effect == child2.effect:
    two_point_crossover(child1, child2)                 [genetic_algorithms.py:100-124]
    child1.q = child2.q = mean(q);  child1.r = child2.r = mean(r)
child1.q /= 2;  child2.q /= 2
```
- **Generalizing mutation** (`genetic_algorithms.py:89-97`): each *specified*
  condition attribute is generalized to wildcard with probability `mu`. (Effect is
  never mutated.)
- **Two-point crossover** (`genetic_algorithms.py:100-124`): swaps a random
  `[left,right)` slice of the **condition** between the two children (chosen with
  `numpy.random.choice` over `0..classifier_length`). Crossover is gated on the two
  children having identical effects.

### E.4 Population-cap / deletion in GA — `genetic_algorithms.py:164-228`
Only children whose condition specificity > 0 are kept (`ClassifiersList.py:230-231`).
Before insertion, `delete_classifiers` enforces the **action-set size cap**:
```
while (insize + Σ num over action_set) > theta_as:        (default theta_as = 20)
    pick a deletion victim by repeatedly scanning expanded action set,
    each micro-classifier selected with prob 0.3, keeping the "worse" one
    (_is_preferred_to_delete);  decrement num or remove if num == 1
```
`_is_preferred_to_delete` (`genetic_algorithms.py:203-228`) prefers deleting lower
quality (Δq < −0.1), then marked-over-unmarked, then higher `tav`.

> **INFERRED bug:** at `genetic_algorithms.py:224` the condition reads
> `cl.is_marked` (the bound method, not `cl.is_marked()`), which is always truthy.
> The intended marked/unmarked tie-break is therefore effectively bypassed.

### E.5 GA insertion / subsumption — `genetic_algorithms.py:127-161,231-270`
`add_classifier`: find an existing subsumer/identical classifier; if found and it is
*not marked*, increment its numerosity; otherwise append the child to population +
action set (+ match set if it matches `p`). Subsumer search (`_find_old_classifier`)
uses `find_subsumers` **only if `do_subsumption` is True**; otherwise it falls back to
exact-equality (`_find_similar`).

---

## F. Population management

### F.1 Storage and iteration
- Population, match set, and action set are all `acs2.ClassifiersList`
  (`acs2/ClassifiersList.py:18`), a `TypedList` wrapping a plain Python `list`
  (`lcs/TypedList.py:9-19`).
- There is **no index / hash bucket**: every match-set and action-set formation is an
  O(N) linear scan (`ClassifiersList.py:44-50`). Deletion (`safe_remove`) is a
  list `.remove` (`TypedList.py:25-29`).
- `expand()` materializes micro-classifiers by repeating each macro-classifier `num`
  times (`ClassifiersList.py:63-73`).

### F.2 Numerosity & macro/micro distinction
- A macro-classifier carries `num`. Subsumption / similarity increments `num`; GA
  deletion decrements it; identity is `(condition, action, effect)`
  (`acs/Classifier.py:54-63`).

### F.3 Insertion
- Via ALP (`anticipatory_learning_process.add_classifier`) and via GA
  (`genetic_algorithms.add_classifier`). Both prefer subsumption/merge over adding a
  new structure.

### F.4 Subsumption conditions — `lcs/strategies/subsumption.py`
`does_subsume(cl, other, theta_exp)` is True iff **all** hold:
```
is_subsumer(cl):  exp > theta_exp  AND  q > theta_r  AND  not marked   [subsumption.py:55-77]
cl.is_more_general(other):  cl.condition.specificity < other.condition.specificity   [acs/Classifier.py:314-329]
cl.condition.subsumes(other.condition)        [acs/Condition.py:72-77]
cl.action == other.action
cl.effect.subsumes(other.effect)              (== effect equality)  [acs/Effect.py:14-15]
```

### F.5 Deletion (whole-population)
There is **no global population-size cap**. The only deletions are:
1. ALP removal of incorrectly-anticipating inadequate classifiers (D.6).
2. GA action-set-size enforcement against `theta_as` (E.4).

`u_max` is *not* a population cap; it bounds the number of specified attributes a
classifier may have during ALP (`alp.py:78-94`).

---

## G. Environment interface

### G.1 Contract the agent expects
The agent treats the environment as an OpenAI-Gym-style object (`ACS2.py:29-170`):
- `env.reset() -> observation` (a sequence convertible to `Perception`).
- `env.step(action) -> (observation, reward, done, info)` (`ACS2.py:96`).
- `env.action_space.sample()` (used once to seed an initial action, `ACS2.py:37`).
- Optionally `env.env.get_goal_state()` for action planning (`ACS2.py:213-222`).

`Perception` is an immutable tuple of `str` attributes (`lcs/Perception.py:4-14`).
The agent asserts `len(state) == cfg.classifier_length` every step (`ACS2.py:54`).

**Action space**: a flat discrete integer set of size
`cfg.number_of_possible_actions`. Actions are plain `int`s; no adapter is applied by
default (`EnvironmentAdapter` is an identity pass-through, `agents/EnvironmentAdapter.py`).

**Reward handling**: scalar reward per step, fed directly into RL (C.2). **Episode
termination**: driven entirely by the env's `done` flag; on `done` the agent runs a
terminal ALP + RL (bootstrap 0) pass (`ACS2.py:99-128`).

### G.2 The maze environment (`openai-envs/gym_maze`)

- **Perception = 8-neighbour wall sensors**, NOT coordinates. `perception()` returns
  the values of the 8 adjacent cells in fixed order **N, NE, E, SE, S, SW, W, NW**
  (`gym_maze/common/maze_utils.py:36-114`, `gym_maze/internal/abstract_maze.py:21-25`),
  stringified (`abstract_maze.py:25` → `map(str, ...)`). Hence `classifier_length = 8`
  for mazes (`run_acs2_maze4.py:132`).
- **Cell encoding** (`gym_maze/common/__init__.py:1-4`): path `0`, wall `1`,
  animat/agent `5`, reward `9`; perceptions carry the stringified ints `'0','1','9'`
  (the agent's own cell is never in its own 8-neighbour view).
- **Action semantics** (`gym_maze/internal/maze_impl.py:9-77`): the 8 actions are the
  8 compass moves; a move only succeeds into a non-wall neighbour, otherwise the agent
  stays put.
- **Reward scheme** (`gym_maze/maze.py:49-56`): `1000` upon reaching the reward cell
  (which also sets `done`), `0` on every other step.
- **Reset** (`gym_maze/maze.py:26-30`): rebuilds the matrix and inserts the agent at a
  uniformly random path cell (`abstract_maze.py:27-35`).
- **Termination**: `done` is `True` once the reward cell is entered
  (`maze_impl.py:31-32,71-76`).
- Maze matrices are fully wall-bordered (e.g. `Maze4.py`), so the `None` edge case in
  `adjacent_cell_values` is never reached. **INFERRED.**
- `get_transitions()` / `get_goal_state()` exist to support knowledge metrics and the
  (default-off) action-planning module (`maze.py:58-65`).

---

## H. Experiment / metrics

### H.1 Phase protocol (the ACS2 maze scripts)
`run_acs2_maze4.py`, `run_acs2_maze5.py`, `run_acs2_maze7.py` are **identical** except
the maze id (verified). Each (`run_acs2_maze4.py:96-145`):
1. `agent.explore(maze, 500)` with `epsilon = 0.8`.
2. `agent.exploit(maze, 200)` with `epsilon = 0.2`.
3. `agent.exploit(maze, 200)` with `epsilon = 0.0`, then a third `exploit(maze, 200)`.

The two phases are run **sequentially**, not alternated. Note `exploit` ignores
`epsilon` (it always uses `BestAction`, `ACS2.py:158`), so the epsilon changes between
exploit phases have no effect on behavior. `Agent.explore_exploit` (an alternating
mode, `agents/Agent.py:72-95`) exists but is **not** used by these scripts.

Config (all three scripts): `classifier_length=8`, `number_of_possible_actions=8`,
`beta=0.05`, `gamma=0.95`, `chi=0.8`, `mu=0.3`, and crucially **`do_ga` is not passed
→ GA is OFF**; `do_pee` / `do_action_planning` likewise default off; `do_subsumption`
defaults on (but, per D.5, only affects the unused GA path).
**Net: the example runs learn purely via ALP specialization + covering + RL.**

### H.2 Metric definitions
- **knowledge** (`run_acs2_maze4.py:48-65`): percentage of all ground-truth
  `(p0, action, p1)` transitions (`env.get_transitions()`) that at least one
  **reliable** classifier predicts successfully (`predicts_successfully`,
  `acs/Classifier.py:243-270`). Range 0–100.
- **population** (`lcs/metrics.py:18`): number of macro-classifiers.
- **numerosity** (`lcs/metrics.py:19`): Σ `num`.
- **reliable** (`lcs/metrics.py:20`): count of classifiers with `q > theta_r`.
- **generalization**: *not* computed by these scripts. (`specificity` /
  `specified_unchanging_attributes` exist on the classifier but no script aggregates a
  generalization metric.) **INFERRED** (absence).
- Base per-trial metrics: `trial`, `steps_in_trial`, `reward`, `perf_time`
  (`lcs/metrics.py:1-8`). Collected every `metrics_trial_frequency` trials
  (`agents/Agent.py:138-149`).

There is **no single standardized benchmark harness**: each maze has its own copy of
the runner script with hardcoded trial counts (500/200/200/200) and the
above-described config.

---

## I. RNG / determinism

Randomness is **not centralized** and **no global seed is plumbed through**:
- Python `random`: parent selection, mutation, GA victim selection, mark differences,
  PEE merge partner, covering randomness, maze reset / goal generation
  (`genetic_algorithms.py:1`, `PMark.py:1`, `ClassifiersList.py:4`, `alp.py:1`,
  `abstract_maze.py:1`, `maze_impl.py:1`).
- `numpy.random`: epsilon-greedy coin flip and random action
  (`action_selection/EpsilonGreedy.py:15`, `RandomAction.py:10`, `BestAction.py:15`),
  two-point crossover cut points (`genetic_algorithms.py:112`).
- `MazeObservationSpace.np_random` returns a **fresh** `np.random.RandomState()` per
  access (`gym_maze/common/maze_observation_space.py:25-27`), so seeding it is
  ineffective. **INFERRED** consequence: runs are not reproducible without external
  seeding of both `random` and `numpy.random`.

---

## J. Canonical conformance & surprises

Overall this is a faithful structural port of ACS2 (condition/effect/mark, ALP
expected/unexpected/cover, q/r/ir updates, GA generalization, subsumption). Notable
divergences and non-obvious behaviors:

1. **Maze perception is 8-neighbour wall sensing, not coordinates** — confirmed
   (`maze_utils.py:36-114`). Answer to the explicit question: **8-neighbor wall
   perception**, stringified cell codes.
2. **Probability-Enhanced attributes are OFF by default** (`do_pee=False`,
   `acs2/Configuration.py:13`) and unused by every ACS2 maze script. Answer to the
   explicit question: **not on by default.**
3. **`biased_exploration_prob` is dead configuration.** The config builds an
   `EpsilonGreedy` and passes `biased_exploration_prob`
   (`acs2/Configuration.py:36-43`), but `EpsilonGreedy.__init__` reads only `epsilon`
   and discards it (`EpsilonGreedy.py:9-18`); `RandomAction` is pure uniform
   (`RandomAction.py:10`). Canonical ACS2 biased exploration (action-delay /
   knowledge-array bias) is **not active**, even though `ActionDelay` and
   `KnowledgeArray` strategy classes exist in the tree.
4. **ALP subsumption ignores `do_subsumption`.** `anticipatory_learning_process.py:28`
   always calls `does_subsume`; the flag gates only the GA path (D.5/E.5).
5. **`u_max` default of 100000 disables ALP generalization.** The expected-case
   over-specialization generalization branches never execute (`alp.py:78-94`). With
   GA also off in the examples, structural learning is **specialize-only** — no
   generalization pressure under the shipped config.
6. **RL bootstrap is on fitness (q·r)** — *canonical, but a porting note.* This
   **matches** Butz & Stolzmann's ACS2 (the bootstrap is the maximum q·r over
   change-anticipating classifiers); it is **not** a divergence. `get_maximum_fitness`
   returns that maximum (`acs/ClassifiersList.py:21-38`), used as the target
   `P = reward + γ·maxfit` (`reinforcement_learning.py:50`). Flagged only because a
   porter must replicate the q-into-bootstrap coupling exactly (and an Actor-Critic
   variant substitutes a learned `V(s')` precisely here — see K.2).
7. **Best-action selection diverges from the base.** The `acs2` override of
   `get_best_classifier` filters to change-anticipating classifiers, shuffles, and
   ranks by `fitness · num` (`acs2/ClassifiersList.py:252-260`), unlike the base
   `max(all, key=fitness)` (`acs/ClassifiersList.py:40-41`). If no
   change-anticipating classifier exists, `BestAction` falls back to a uniform random
   action (`BestAction.py:11-15`).
8. **No model learning during exploitation** — `_run_trial_exploit` runs RL only
   (`ACS2.py:134-170`).
9. **`_is_preferred_to_delete` bound-method bug** (`genetic_algorithms.py:224`),
   INFERRED (E.4).
10. **`ClassifiersList.copy()` is broken** (`acs2/ClassifiersList.py:58-61`): it copies
    `self.__slots__` (a class attribute) into `items`, not the classifiers. INFERRED;
    not on the main ACS2 path but a hazard for a porter who assumes it works.

---

## K. Extension surface for Actor-Critic

### K.1 How the existing variants extend the base
Reading ACS2ER, ACS2HER, and ACS2VCP reveals one consistent pattern — and it is
**not** subclassing `ACS2`:

- **All variants subclass the abstract `Agent`** (`agents/Agent.py:18`), *not* `ACS2`:
  `ACS2ER(Agent)` (`acs2er/ACS2ER.py:15`), `ACS2HER(Agent)` (`acs2her/ACS2HER.py:16`),
  `ACS2VCP(Agent)` (`acs2vcp/ACS2VCP.py:18`).
- **They re-implement (copy-paste) the trial loops** `_run_trial_explore` /
  `_run_trial_exploit` rather than overriding hook points. The bodies are near-clones
  of `ACS2.py` differing in *when* and *on what data* learning is applied
  (`acs2er/ACS2ER.py:30-105`, `acs2her/ACS2HER.py:37-95`, `acs2vcp/ACS2VCP.py:63-126`).
- **They reuse the static learning primitives unchanged**:
  `ClassifiersList.apply_alp`, `.apply_reinforcement_learning`, `.apply_ga`, and
  `form_match_set` / `form_action_set` (e.g. `acs2her/ACS2HER.py:137-177`).
- **They subclass `Configuration`** to add hyperparameters:
  `acs2er.Configuration(acs2.Configuration)` adds buffer params
  (`acs2er/Configuration.py:4-15`); `acs2her.Configuration` adds HER goal/strategy
  params (`acs2her/Configuration.py`).
- They keep the same surface: `self.population`, `self.cfg`, `get_population()`,
  `get_cfg()`, returning `TrialMetrics(steps, reward)`.

So the de-facto extension contract for any new variant — including Actor-Critic — is:
**subclass `Agent`, subclass `Configuration`, and write a new pair of trial loops that
call the existing `ClassifiersList.*` static methods, replacing only the bootstrap and
action-selection seams.**

### K.2 Concrete seams for a critic (state-value estimate)
- **Bootstrap argument to RL.** Every loop passes a bootstrap value as the third
  argument of `apply_reinforcement_learning(action_set, reward, P, beta, gamma)`
  (`ACS2.py:70-76`). Today `P = match_set.get_maximum_fitness()`. A critic supplies a
  learned `V(s')` here instead. This is the cleanest single attachment point.
- **`reinforcement_learning.update_classifier`** (`reinforcement_learning.py:25-54`) is
  the per-classifier value-update function; an AC variant can add a parallel
  `apply_value_learning` static method on `ClassifiersList` (mirroring
  `apply_reinforcement_learning`, `ClassifiersList.py:243-250`) that performs a TD
  update on the critic.
- **`cfg.fitness_fcn`** (`acs/Classifier.py:124-129`, `acs/Configuration.py:30`) lets a
  variant redefine fitness (e.g. blend in a learned advantage) without touching the
  classifier class.

### K.3 Concrete seams for an actor (policy)
- **`cfg.action_selector`** is a pluggable callable `__call__(match_set) -> int`
  (`acs2/Configuration.py:39-43`, used at `ACS2.py:91`). Replacing `EpsilonGreedy` with
  a softmax/policy-gradient selector over classifier preferences is the actor seam.
  New selectors live alongside `EpsilonGreedy`, `BestAction`, `RandomAction` in
  `lcs/strategies/action_selection/`.
- For exploitation, `_run_trial_exploit` hardcodes `BestAction` (`ACS2.py:158`); an AC
  variant's exploit loop would call its actor instead.

### K.4 Per-classifier state for AC
- Add fields (e.g. value estimate, eligibility trace) by extending
  `Classifier.__slots__` (`acs2/Classifier.py:16-17`) and the constructor — exactly how
  `acs2.Classifier` already extended `acs.Classifier` with `ir`, `num`, `exp`, `tga`,
  `ee`.

### K.5 Nearest existing precedent: ACS2VCP already builds a proto-critic
`ACS2VCP` maintains an **ensemble of `ACS2HER` heads** and computes, per
(state, action), a Q-prediction = max action-set fitness
(`compute_q_prediction`, `acs2vcp/ACS2VCP.py:48-51`) across heads, then uses the
**variance** across heads to prioritize replay via `PrioritizedReplayBuffer`
(`ACS2VCP.py:53-61, 210-220`). This is structurally an uncertainty-weighted critic
over classifier fitness — the closest in-repo analogue to where an Actor-Critic value
head would attach. (Caveat: `ACS2VCP._run_trial_exploit` iterates
`range(self.ensemble_heads)` over a *list* and would raise `TypeError` — the exploit
path is **broken**. INFERRED, `ACS2VCP.py:133`.)

### K.6 Classes/methods an AC variant must add or override (summary)
1. `Configuration` subclass — actor/critic learning rates, trace decay, custom
   `action_selector`.
2. A new action-selection strategy class (the **actor**) implementing
   `__call__(match_set) -> int`.
3. A critic update — either a new `ClassifiersList.apply_*` static method or a new
   function in `lcs/strategies/reinforcement_learning.py`, plus substituting `V(s')`
   for `get_maximum_fitness()` at the RL bootstrap call sites.
4. `Agent` subclass with re-implemented `_run_trial_explore` / `_run_trial_exploit`
   threading actor selection and critic updates.
5. (Optional) `Classifier.__slots__` extension for per-classifier value/eligibility.

---

## Fitness as a porting target

The component seams are clean and well-isolated: condition/effect/mark, the
ALP/GA/RL strategy functions, action-selection strategies, and configuration are all
small, single-responsibility units with explicit, citable formulas, and they carry
real unit tests (`tests/lcs/agents/acs2/test_{Classifier,ClassifierList,Effect,alp,
ProbabilityEnhancedAttribute}.py`, plus `tests/lcs/strategies/*`). A Rust
reimplementation can mirror this module decomposition almost one-to-one, and the
Actor-Critic attachment points (`action_selector` for the actor; the RL bootstrap
argument plus a critic update for the critic) are narrow and well-defined. The
**orchestration layer is the weak spot**: there is no `test_ACS2.py` exercising the
trial loop, the variant agents copy-paste the loop instead of sharing it (so behavior
drifts between them), HER/VCP are essentially untested, and at least three concrete
bugs live in the extension/management code (the `_is_preferred_to_delete` bound-method
check, `ClassifiersList.copy`, and the `ACS2VCP` exploit loop). A porter should treat
the **per-component logic as the trustworthy specification** and **re-derive the trial
loop and variant orchestration from scratch** — rebuilding it around a single shared,
hook-based loop with a pluggable actor and critic, rather than transcribing the
duplicated Python loops.
