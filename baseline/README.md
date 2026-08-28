# pyalcs ACS2 baseline (Python reference)

Isolated, reproducible environment that runs the **unmodified** pyalcs ACS2 agent
on the `gym_maze` mazes. This is the Python reference whose wall-clock time and
learning quality the Rust port is measured against. The pyalcs and openai-envs
sources are never edited — they are the thing being measured.

## What it pins

- **Python 3.10** (`.python-version`, fetched and managed by `uv`).
- `gym==0.23.0`, `numpy==1.23.5` (gym 0.23 relies on numpy symbols removed in
  numpy 2.0, e.g. `np.bool8`; 1.23.5 is the last comfortable pairing).
- `networkx==2.5`, `bitstring==3.1.7`, `dataslots>=1.0.1` (transitive needs of
  openai-envs / pyalcs).
- `dill` and `mlflow-skinny`. `lcs/agents/Agent.py` does `import dill` and
  `import mlflow` unconditionally at module top even though both are only listed
  as setup.py *extras*. `use_mlflow` defaults to `False`, so no mlflow call ever
  executes; `mlflow-skinny` satisfies the import without pulling a numpy that
  conflicts with the `numpy==1.23.5` pin.
- `pyalcs` and `parrotprediction-openai-envs` installed **editable** from local
  paths via `[tool.uv.sources]`.

Exact resolved versions are locked in `uv.lock`.

## Layout assumption

The two reference repositories must be cloned as siblings of this project:

```
Desktop/
├── acs2-rust/baseline/      (this directory)
└── acs2vcp-python/
    ├── pyalcs/              (import root: lcs)
    └── openai-envs/         (import roots: gym_maze, gym_woods)
```

The editable path sources in `pyproject.toml` are `../../acs2vcp-python/pyalcs`
and `../../acs2vcp-python/openai-envs`.

## Setup

```bash
cd baseline
uv sync
```

`uv sync` fetches Python 3.10, resolves the pinned set, and builds editable
installs of the two local packages.

## Run

```bash
uv run python run_pyalcs_maze.py Maze4-v0
uv run python run_pyalcs_maze.py Woods1-v0 --seed 42 --repeats 10
```

Arguments:

- `maze` — gym_maze environment id (e.g. `Maze4-v0`, `Maze5-v0`, `Maze7-v0`,
  `Woods1-v0`, `Woods100-v0`).
- `--seed` (default `42`) — base seed. Repeat *i* uses `seed + i`.
- `--repeats` (default `1`) — number of independent repetitions. The pinned
  benchmark (PROJECT_CONTEXT §5) uses 10.

## Protocol (PROJECT_CONTEXT §5)

Per repetition, with a fresh agent:

1. `explore` 500 trials at `epsilon = 0.8`.
2. `exploit` 200 trials, three times.

Exploitation always uses `BestAction` and ignores `epsilon`, so the three
exploit phases are independent pure-RL evaluations of the frozen population. The
headline metric is **mean steps-to-goal over the final 200-trial exploit
window**. The per-maze episode cap is the `TimeLimit` registered by `gym_maze`
(50 for Maze4/5/7 and Woods1; 500 for Woods100/101/102) and is applied
automatically by `gym.make` — it is not overridden here.

Config matches the shipped `run_acs2_maze*.py`: `beta=0.05`, `gamma=0.95`,
`theta_i=0.1`, `theta_r=0.9`, `theta_exp=20`, `theta_as=20`, `u_max=100000`,
`mu=0.3`, `chi=0.8`, `do_ga=False`, `do_pee=False`, `do_action_planning=False`,
`do_subsumption=True`.

## Output

Per repeat the script prints the full per-trial steps-to-goal for the explore
phase and each exploit phase, then a summary: explore/exploit/total wall-clock,
final-window steps (mean/std/min/max), and population counts (macro,
numerosity, reliable). A trailing block aggregates the final-window mean and
per-repeat time across repeats.

Filter the verbose per-trial lines and gym deprecation warnings when you only
want the summaries:

```bash
uv run python run_pyalcs_maze.py Maze4-v0 2>/dev/null | grep -E "===|summary|###"
```

## ACS2ER differential fixture

`dump_acs2er_differential.py` is the oracle for the ACS2ER (experience replay)
agent. It runs the unmodified pyalcs `ACS2ER` explore loop on a deterministic
environment and emits `fixtures/acs2er_differential.json`, consumed by
`acs2-core/tests/p11_acs2er_differential.rs`.

```bash
uv run python dump_acs2er_differential.py
```

Output is byte-stable across runs: action selection is scripted and every draw
from `random` is recorded, so re-running it should leave the fixture unchanged.
Like the other dumpers it instruments pyalcs **in-process only** and never
modifies it. It also counts ALP deletions, because those trigger the pyalcs
mid-iteration skip that would invalidate an end-to-end population comparison;
the gate configuration uses `theta_i = 0` so the count must stay at 0.

## Notes / caveats

- **Reproducibility is best-effort** (PROJECT_CONTEXT §5). `random`, `numpy`,
  and the env's action space are all seeded; maze reset placement is driven by
  Python `random` (SPEC §I), so it is seed-controlled. The comparison targets
  wall-clock and learning quality, not bit-identical trajectories.
- **Woods mazes use a different perception alphabet.** Maze4/5/7 perceptions are
  `'0'`/`'1'`/`'9'` (path/wall/reward) as in PROJECT_CONTEXT §5; the `gym_woods`
  family (`Woods1-v0`, …) emits `'.'`/`'O'`/`'F'`. pyalcs treats symbols as
  opaque strings, so both run unchanged — relevant only for the later Rust maze
  port (P7).
- gym 0.23 prints a NumPy-2.0 deprecation banner and "Overriding environment"
  registration warnings on stderr. Both are harmless.
