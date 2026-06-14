# ALCS Anomaly Verification

**Subject:** `github.com/ounold/ALCS` — the ACS2 multi-backend framework benchmarked in
Unold, *"A high-performance ACS2 based on vectorization and GPU"* (GECCO Companion '26) + supplement.

**Method:** Read-only. Repo cloned to `/tmp/ALCS_verify` (default branch, HEAD `b5e16a6`/snapshot upload).
No file in any ALCS checkout was modified. Code-reading was conclusive for all three issues; no
benchmark was run. File:line references are to the cloned tree; paths are identical in the local
copy at `/Users/aleklyskawa/Desktop/ALCS`.

**Bottom line:**

| Issue | Status | Verdict |
|---|---|---|
| 1. Per-phase epsilon dead write on CPU | **CONFIRMED** | Real bug. CPU runs *all three phases* at ε≈0.1; GPU applies 0.8/0.2/0.0 correctly. Material to the exploit-step numbers and to the "GPU has lower steps" interpretation; **not** to the headline runtime/population rankings. |
| 2. MazeF1 & MazeMA goal in a wall | **CONFIRMED** | Both `goalState` cells are `OBSTACLE` → goal unreachable → the 200.000 ± 0.000 entries are the episode cap, not a learning outcome. |
| 3. MiyazakiA goal transposition | **CONFIRMED** | `PRIZE` at (3,6), `goalState = {6,3}` (row/col swapped). Runtime terminates at (6,3), a corridor — never the intended prize. |

---

## Issue 1 — Per-phase epsilon is a dead write on the CPU path (INVESTIGATED → CONFIRMED)

### 1a. CPU path: the write and the read target different attributes

**Write** (experiment runner, per phase):

`src/experiment_runnerCPU3.py:132-140`
```python
for phase_name, phase_cfg in experiment_config.phases.items():
    if phase_cfg.episodes <= 0:
        continue
    agent.epsilon = phase_cfg.epsilon              # line 136  -> sets agent.epsilon
    agent.beta    = phase_cfg.beta                  # line 137  -> sets agent.beta
    agent.cfg.do_alp          = phase_cfg.alp       # line 138  -> via agent.cfg  (works)
    agent.cfg.do_ga           = phase_cfg.ga        # line 139  -> via agent.cfg  (works)
    agent.cfg.do_decay_epsilon = phase_cfg.decay    # line 140  -> via agent.cfg  (works)
```

Note the inconsistency in this very block: lines 138–140 route through `agent.cfg.*`; lines 136–137
write bare attributes `agent.epsilon` / `agent.beta`.

**Read** (action selection):

`src/models/acs2/acs2CPU3.py:51`
```python
elif explore and random.random() < self.cfg.epsilon:   # reads self.cfg.epsilon, NOT self.epsilon
    action = random.randint(0, self.cfg.num_actions - 1)
```

**The assigned attribute is dead.** `ACS2CPU3.__init__` (`acs2CPU3.py:13-18`) sets `self.cfg`,
`self.population_dict`, `self.population_by_action`, `self.time`, `self.curr_ep_idx` — and **no**
`self.epsilon`. The class has exactly one `@property` (`population`, line 20-22) and **no
`epsilon`/`beta` property or setter** (verified: `grep "def epsilon|def beta|@.*setter"` →
no matches). So `agent.epsilon = …` creates a fresh instance attribute that nothing ever reads
(repo-wide grep: the only readers of an *agent-level* `.epsilon` are the GPU agent and the GPU
runner; no code reads `ACS2CPU3.epsilon`).

`self.cfg.epsilon` therefore keeps its build-time value for the entire run. That value is the
global ε: default `0.1` (`src/models/acs2/confCPU3.py:22`, `src/defaults_cpu3_gpu4.py:41`,
`src/configCPU3.py:232`), matching the paper's stated global ε = 0.1 (§5.1).

**One more load-bearing fact:** the runner calls `agent.run_step(state)` with the default
`explore=True` in *every* phase (`experiment_runnerCPU3.py:147`; signature default at
`acs2CPU3.py:39`). Phases are *only* differentiated by epsilon — there is no `explore=False` path.
So with epsilon stuck at 0.1, the `explore and random()<ε` branch is live in all phases.

**Effective epsilon on the CPU backend (cpu_single and cpu_mp — both use this runner):**

| Phase | Intended ε (paper §5.1) | Effective ε on CPU |
|---|---|---|
| explore  | 0.8 | **0.1** |
| exploit1 | 0.2 | **0.1** |
| exploit2 | 0.0 | **0.1** |

(`agent.beta` at line 137 is *also* a dead write — `acs2CPU3.py:73-74` reads `self.cfg.beta` — but
β = 0.05 in all three phases and globally, so it is inert. Epsilon is the only one that matters.)

### 1b. GPU path: wired correctly

The GPU agent (GPU4 = the tensorized backend used for all GPU rows) stores epsilon as an
**instance attribute** and reads that same attribute:

- `src/models/acs2/acs2GPU4.py:51` — `self.epsilon = 0.1` (init, instance attribute)
- `src/experiment_runnerGPU4.py:155` — `agent.epsilon = phase.epsilon` (per-phase write)
- `src/models/acs2/acs2GPU4.py:163-169` — action selection:
  ```python
  if explore:
      do_explore = torch.rand((self.n_exp,), ...) < self.epsilon   # line 164: reads self.epsilon
      ...
  ```

The GPU loop calls `agent.run_step(prev_states, active_mask=active_mask)` (`experiment_runnerGPU4.py:172`)
with default `explore=True` in all phases — **symmetric** to the CPU design. The *only* difference
is the read site: GPU reads `self.epsilon` (the attribute the runner writes), CPU reads
`self.cfg.epsilon` (an attribute the runner never touches). Consequently in exploit2 the GPU does
`rand < 0.0` → all-False → pure `argmax` greedy. **GPU applies 0.8 / 0.2 / 0.0 correctly; CPU does
not.** This asymmetry is the corroboration the supplement evidence predicts.

Root cause is plausibly a copy of the runner line `agent.epsilon = phase.epsilon` onto two agents
whose internal conventions had diverged (GPU4: `self.epsilon`; CPU3: `self.cfg.epsilon`). It happens
to bind on GPU and dangles on CPU.

### 1c. Corroborating evidence — does the code explain the data? (Supplement Table 1, exploit2 steps)

Yes, on the subset of mazes where the policy actually converges to short paths — exactly where a
10% random-action rate is visible. CPU runs exploit2 at ε≈0.1 (≈10% random) while GPU runs ε=0.0
greedy, so CPU steps should be systematically *higher* there:

| Maze | CPU Single | CPU MP | GPU Seq | GPU CPU | CPU > GPU? |
|---|---|---|---|---|---|
| Woods100   | 2.246  | 2.246  | 2.000  | 2.023  | yes |
| Woods1     | 3.094  | 3.094  | 2.094  | 2.309  | yes |
| Cassandra4x4 | 5.035 | 5.035 | 3.101  | 3.065  | yes |
| MazeB      | 5.514  | 5.514  | 3.994  | 4.208  | yes |
| MazeD      | 4.286  | 4.286  | 3.374  | 3.401  | yes |
| Maze4      | 5.394  | 5.394  | 4.031  | 4.006  | yes |
| Littman57  | 5.702  | 5.702  | 4.739  | 4.332  | yes |
| MiyazakiA  | 4.631  | 4.631  | 3.508  | 3.464  | yes |
| MiyazakiB  | 5.144  | 5.144  | 3.862  | 3.874  | yes |
| MazeE2     | 7.778  | 7.778  | 4.529  | 4.184  | yes |
| MazeE3     | 15.930 | 15.930 | 8.376  | 8.100  | yes |

On these converging mazes the CPU/GPU gap is systematic and of the magnitude a ~10% random rate
would produce (e.g. Woods100 +12%; longer optimal paths inflate more because a random step can
require several corrective steps, e.g. MazeE2/E3). **The code in 1a/1b explains this pattern.**

**Fairness caveat — the pattern is NOT universal.** On hard, non-converging mazes the epsilon
signal is swamped by the population-management differences the paper itself flags, and several
rows even reverse: e.g. Maze10 (CPU 147.995 < GPU 171.242 / 174.528) and MazeF3 (CPU 103.982 <
GPU 142.385 / 126.307). So the four mazes named in the prompt are representative of the
*converging subset only*, not of all 24. State it as "systematic on the converging/short-path
subset," not as a universal effect.

**Strongest single corroborator:** CPU Single and CPU MP report *identical* exploit-step values on
every maze (and identical macro-population, 130.15). The paper says they share the learner and
differ only in scheduling (§4.1) — so the identical numbers confirm the defect is shared and
deterministic across both CPU modes, which is exactly what a runner-level dead write predicts.

### 1d. Severity — precise and fair

- **Does it corrupt exploit2 steps (the paper's main quality metric)?** For the CPU modes, **yes**:
  exploit2 is supposed to be ε=0.0 greedy ("stable late-stage policy quality under the final
  exploitation regime", §4.4 / §5.1), but on CPU it runs at ε≈0.1. The reported CPU exploit2
  numbers include ~10% exploration noise and do not measure what the protocol says they measure.
- **Does it reduce explore-phase exploration?** **Yes** — 500 explore episodes run at ε=0.1 instead
  of 0.8, i.e. far less state-space coverage during rule discovery. This plausibly contributes to
  the smaller CPU macro-population (130.15 vs GPU ~168, Symbolic 254.60), though that gap is
  multi-causal (backend population-management differs too), so epsilon cannot be isolated as *the*
  cause — present it as a plausible contributor, not a proven one.
- **Runtime and population rankings:** the bug is shared *identically* by both CPU modes, so the
  CPU-vs-CPU ordering — the paper's headline runtime claims (CPU MP best total time, CPU Single
  best per-experiment time) — is **unaffected**. Population comparisons across backends are already
  heavily caveated by the paper.

**Verdict: real bug.** It is material to (a) the CPU exploit-step *numbers*, which don't reflect the
intended greedy exploitation, and (b) the *interpretation* of the GPU advantage: the paper
attributes GPU's lower exploit-steps to "tensorized execution and backend-specific
population-management" (§6.1/§6.2), but a simpler partial explanation is that **GPU runs exploit2
greedy and CPU does not.** It is **not** material to the paper's primary, foregrounded conclusions
about runtime and maintained population size. A fair framing for the supervisor: this sharpens the
step-quality discussion and removes a confound, rather than overturning the systems result.

### 1e. Git history

The CPU runner exists in a single squashed "Add files via upload" commit (`b5e16a6`); the agent read
site in another upload commit (`36b694f`). `git log -S "agent.epsilon = phase_cfg.epsilon"` returns
only that one upload commit, and the runner has no prior revisions. So in this repository the line
has been a dead write **since introduction**; whether it was ever wired correctly in pre-upload
history **cannot be determined — the history is a snapshot upload, not incremental commits.**

### 1f. The fix (one line)

`src/experiment_runnerCPU3.py:136`
```python
-        agent.epsilon = phase_cfg.epsilon
+        agent.cfg.epsilon = phase_cfg.epsilon
```
This routes the per-phase value into the attribute action-selection actually reads
(`acs2CPU3.py:51`), matching how lines 138–140 already handle `do_alp`/`do_ga`/`do_decay_epsilon`.
For consistency, line 137 `agent.beta = phase_cfg.beta` → `agent.cfg.beta = phase_cfg.beta`
(currently inert, but the same dead-write pattern). No change is needed on the GPU path.

---

## Issue 2 — MazeF1 and MazeMA have an unreachable goal (CONFIRMED)

**Discriminating mechanism (decides all of Issue 2 and 3).** The Python loader takes the goal from
the `.goalState` field **literally** and **never uses the `PRIZE` matrix marker** for the goal:

- `environment/maze_loader.py:47-50` — `goalState` parsed as `(group1, group2)`.
- `:67`, `:93` — that pair becomes `goal_pos`.
- `:77-82` — the grid is split into `OBSTACLE` vs everything-else; `PRIZE` is *not* `OBSTACLE`, so
  it is silently lumped into **corridors**. The prize location is discarded.
- Runtime termination uses `goal_pos` only: `environment/runtime_cpu3.py:103`
  (`if (curr_row, curr_col) == self.goal_pos`) and `:137`
  (`done = tuple(final_pos) == self.goal_pos`; reward `1000.0` iff `done`). `PRIZE` is never
  referenced in the runtime.

`goal_pos` is consumed as `(row_idx, col_idx)` (compared against `(row_idx, col_idx)` corridor/grid
positions throughout the loader and runtime). So the goal cell is `grid[goalState[0]][goalState[1]]`.

### 2a. Matrix value at goalState

**MazeF1** (`environment/acs2_mazes/MazeF1.cpp`) — `mazeWidth=4, mazeHeight=6`, `goalState = {1, 3}`
(line 15). Row 1 is `{ OBSTACLE, CORRIDOR, PRIZE, OBSTACLE }` (line 19).
- `grid[1][3]` = **OBSTACLE** ← the goal cell.
- The `PRIZE` is at `grid[1][2]`. The commented-out reference grid (`{1,0,2,1}`, lines 24-29)
  confirms the intended prize marker `2` sits at col 2, not col 3.

**MazeMA** (`environment/acs2_mazes/MazeMA.cpp`) — `mazeWidth=11, mazeHeight=7`, `goalState = {1, 7}`
(line 15). Row 1 is `{ OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE,
OBSTACLE, OBSTACLE, PRIZE, OBSTACLE }` (line 19).
- `grid[1][7]` = **OBSTACLE** ← the goal cell.
- The `PRIZE` is at `grid[1][9]` (comment grid `…, 2, 1` at col 9, line 26).

In both mazes the goal cell is an `OBSTACLE`. The agent only ever occupies corridor/prize cells, so
`current_pos` can never equal an obstacle `goal_pos` → the goal is **unreachable by construction**.

### 2b. Table 1 shows the episode cap for both, across all five modes

Supplement Table 1 (exploit2 steps):

| Maze | Symbolic | CPU Single | CPU MP | GPU Seq | GPU CPU |
|---|---|---|---|---|---|
| MazeF1 | 200.000 (0.000) | 200.000 (0.000) | 200.000 (0.000) | 200.000 (0.000) | 200.000 (0.000) |
| MazeMA | 200.000 (0.000) | 200.000 (0.000) | 200.000 (0.000) | 200.000 (0.000) | 200.000 (0.000) |

`n_steps = 200` (§5.1). 200.000 with zero variance, on every mode and every repetition, is exactly
the signature of "goal never reached, every episode hits the cap." Confirmed.

### 2c. Plain statement

MazeF1 and MazeMA are **degenerate at source**: the parsed `goalState` lands on an `OBSTACLE` cell,
so no agent in any backend can reach the goal. Their 200.000 ± 0.000 entries in the published
averages are the **episode-length cap, not a learning outcome**, and should not be read as policy
quality. Note this is symmetric across all backends (it comes from the shared maze loader, not from
the epsilon bug), so it does not bias the cross-backend comparison — but it does mean two of the 24
mazes contribute a constant 200 to every mode's average rather than a measured result.

---

## Issue 3 — MiyazakiA goal row/col transposition (CONFIRMED)

### 3a. PRIZE location vs goalState

`environment/acs2_mazes/MiyazakiA.cpp` — `mazeWidth=8, mazeHeight=8`, `goalState = {6, 3}` (line 15).
Row 3 is `{ OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, PRIZE, OBSTACLE }` (line 21).
- `PRIZE` is at `grid[3][6]` → intended goal **(3, 6)**. The commented reference grid confirms the
  marker `2` at row 3, col 6 (`{1,1,0,0,0,1,2,1}`, line 29).
- `goalState = {6, 3}` → **(6, 3)**, i.e. the coordinates are **transposed** relative to the prize.

### 3b. Which one the runtime uses

The runtime uses `goalState` (= `goal_pos`), not the prize marker — same path as Issue 2:
`maze_loader.py:47-50/67/93` → `runtime_cpu3.py:103,137`. So termination is checked against
**(6, 3)**.
- `grid[6][3]`: row 6 is `{ OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR,
  OBSTACLE }` (line 24) → `grid[6][3]` = **CORRIDOR**.

So unlike MazeF1/MazeMA, the transposed cell is a *reachable corridor* — which is why MiyazakiA does
**not** cap at 200 but instead yields finite step counts (Table 1: 7.571 Symbolic, 4.631 CPU,
3.508/3.464 GPU). The agent learns to reach an ordinary corridor cell (6,3); the actual intended
prize at (3,6) is never the termination target for any backend.

### 3c. Plain statement

Yes — this is a **row/col transposition in the source**: `PRIZE` sits at (3,6) but
`goalState = {6,3}`. The runtime terminates at **(6,3)** (a corridor), which is what actually runs
for all five modes. The reported MiyazakiA steps therefore measure the distance to the wrong cell.
Because the (6,3) goal is reachable, the numbers look "normal" and the error is easy to miss — the
commented-out reference grid (prize `2` at row 3 col 6) is the clearest evidence that (3,6) was
intended. As with Issue 2, the error is in the shared maze loader/source and applies symmetrically
across backends, so it does not by itself bias the CPU-vs-GPU comparison; it does mean MiyazakiA's
"steps-to-goal" is not the steps-to-prize the maze name implies.

---

## Notes on scope and fairness

- **All three are honest, reproducible source/code facts**, not interpretation: the dead write is a
  plain attribute mismatch with no setter; the maze goals are `OBSTACLE`/transposed cells read
  directly from the `.cpp`; the loader/runtime provably ignore `PRIZE`.
- **The "Symbolic ACS2" baseline is out of scope here.** Its action selection (`acs2/acs2.py:39`)
  also reads `self.cfg.epsilon`, but it is a separate, clearly weaker learner (e.g. Woods100:
  Symbolic 73.076 vs CPU 2.246) and is not dispatched through the CPU3 benchmark runner
  (`run_maze_benchmarks.py:21-30` lists only `cpu_single`/`cpu_mp`/`gpu*`). It should be assessed on
  its own if relevant, not folded into the epsilon corroboration.
- **What does NOT change:** the paper's primary systems conclusions — CPU MP best total runtime,
  CPU Single best per-experiment runtime, GPU substantially slower, population as a partial cost
  explanation — survive all three findings. Issues 2 and 3 are shared by every backend; Issue 1 is
  shared identically by both CPU modes. The findings refine the *step-quality* discussion and the
  *attribution* of the GPU's lower exploit-steps, and they flag three maze/metric entries that are
  artifacts rather than measurements. That is the fair, defensible scope to present to the
  supervisor ahead of the July conference.
