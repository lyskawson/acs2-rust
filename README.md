# acs2-rust

An independent Rust implementation of **ACS2** (Anticipatory Classifier System,
Butz & Stolzmann) and **ACS2ER** (ACS2 with experience replay), written for a
master's thesis on prioritised sample selection from replay memory.

It solves Boolean multiplexers well beyond what the ACS literature reports.

```bash
cargo build --release
cargo test --release                    # 73 tests
./target/release/acs2-bench             # maze benchmark
./target/release/mpx_reach --sizes 20   # multiplexer, solves in seconds
```

## What it does

**Multiplexer scaling.** Published ACS/ACS2 results stop at 20–37 bits. This
implementation reaches 1.0 knowledge — anticipating *every* transition correctly —
at 70 and 135 bits:

| Problem | Result | Cost | Final population |
|---|---|---|---|
| MPX-70 | solved on **5 of 5 seeds** | 17.8–66.4 M trials | 268–277 reliable rules, specificity 7.00–7.04 (ideal 7) |
| MPX-135, modified encoding | solved on **4 of 5 seeds** | 30.2–55.6 M trials | 532–539 reliable rules, specificity 8.00–8.01 (ideal 8) |
| MPX-135, canonical encoding | solved, **1 seed so far** | 428 M trials | 532 reliable rules, specificity 8.02 |

Two things make those numbers readable:

- **`knowledge` is stricter than the accuracy the literature reports.** It requires
  the model to anticipate every transition, including the null ones a wrong answer
  produces under the canonical encoding. On MPX-135 the agent reaches **answer
  accuracy 1.0000 while knowledge sits at 0.7499** — by the criterion ExSTraCS and
  ACS2ER use, that is already solved. Both metrics are instrumented; report them
  together.
- **Two problem encodings.** Canonical MPX leaves the perception unchanged after a
  wrong answer, so the rules for those transitions must anticipate identity. The
  `outcome` encoding gives every action an observable effect. It is about ten times
  cheaper but is formally a different problem, so its results are **not comparable
  to the multiplexer literature**; canonical ones are.

`reports/MPX_final.md` is the narrative; `reports/MPX<k>_runs.md` is every run at
that size in one table.

**Performance.** Against the pyalcs reference on the pinned maze protocol
(5 mazes, `n_exp=10`, GA off, 500 explore + 3×200 exploit, same machine,
sequential): **139× on total time** (306.20 s → 2.20 s). Exploit steps-to-goal
agree within 1.00–1.07×, so it is the same algorithm, not a faster different one.
Against `ounold/ALCS` `cpu_single` over 22 mazes: ~27× median.

**Correctness.** 800 random `(population, p0, action, p1, time, reward)` inputs run
through both an instrumented pyalcs and this core, comparing match set, action set,
next-state match set, RL bootstrap and the population after one learning step:
**761 of 761 deterministic cases agree, zero divergence.** The other 39 are excluded
for cause — 24 where pyalcs picks a mark candidate at random by design, 15 where
pyalcs skips a classifier mid-iteration, a bug this implementation does not
reproduce. ACS2ER is validated the same way (`p11_acs2er_differential.rs`).

## Relationship to pyalcs

`hendrykik/acs2vcp-python` is a **correctness oracle and a performance baseline**,
not a target. Where it and canonical ACS2 disagree, the deviation is recorded and
justified in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The shipped defaults deliberately match pyalcs — GA off, `u_max = 100000` (which
leaves the ALP generalization branch dormant), the pinned maze protocol — because
differential validation and a fair timing comparison both need an identical
configuration on the two sides. Every one of them is a flag, and the multiplexer
work overrides them routinely.

Four pyalcs bugs are **not** reproduced: three orchestration bugs
(`_is_preferred_to_delete`, `ClassifiersList.copy()`, `_run_trial_exploit`) and the
`apply_alp` mid-iteration skip caused by deleting from a list while iterating it.
All four were surfaced by the differential testing. See
[`reports/P8_differential.md`](reports/P8_differential.md).

## Layout

Cargo workspace, Clean Architecture — the domain crate has no I/O and no knowledge
of any environment:

| Crate | Kind | Purpose |
|---|---|---|
| `acs2-core` | lib | Classifier, population, ALP, RL, GA, action selection, config, injected RNG, the shared trial loop, and both agents (`agent::Agent`, `acs2er::Acs2ErAgent`) behind the `trial::LearningAgent` trait. |
| `acs2-envs` | lib | The `Environment` trait: the 8-sensor maze with geometry in `mazes/` (5 canonical pyalcs mazes plus 22 from `ounold/ALCS`), and the multiplexer with both encodings. |
| `acs2-bench` | bin | `acs2-bench` runs the maze protocol; `mpx_reach` runs the multiplexer scaling experiment. |

`baseline/` holds the pinned Python reference (its lockfile is part of the
validation evidence). `tools/` is a **separate** Python project for parsing and
plotting, kept apart so plotting dependencies never perturb that pinned
environment. `slurm/` holds the cluster scripts, `fixtures/` the golden vectors,
`reports/` every result.

## Running experiments

### Maze benchmark

```bash
./target/release/acs2-bench
```

No flags gives the pinned protocol: all five mazes, `n_exp=10`, seed 42, GA off,
500 explore + 3×200 exploit, written to `reports/bench_rust.csv`.

Flags: `--mazes <a,b,...>`, `--n-exp <k>`, `--seed <s>`, `--do-ga`,
`--explore-trials`, `--exploit-trials`, `--exploit-phases`, `--out <path>`,
`--agent acs2|acs2er`.

> Always time the **release** binary. A debug build can read slower than CPython
> and would invert the result.

### Multiplexer

```bash
./target/release/mpx_reach --sizes 20 --n-exp 1 --seed 42
```

Sizes are not continuous — `k = a + 2^a` gives 6, 11, 20, 37, 70, 135, 264, 521.
20 and 37 solve in seconds to minutes; 70 takes hours; 135 takes days.

| Flag | Default | Meaning |
|---|---|---|
| `--sizes <a,b,...>` | `37,70,135` | Which multiplexers to run. |
| `--seed <s>` / `--n-exp <k>` | `42` / `3` | Repeat *r* runs at seed `s + r`. |
| `--time-cap-secs <s>` | `600` | Wall-clock budget per repeat. |
| `--u-max derived\|<int>` | `derived` | ALP specificity ceiling. `derived` is `a + 2`, which is too tight at 135 — that needs 11. |
| `--encoding flip\|outcome` | `flip` | `flip` is canonical; see above. |
| `--epsilon <f>` | `0.8` | Exploration rate. `1` removes the greedy bias entirely, which is what solves 135 canonically. |
| `--alp-gen-variant pyalcs\|butz` | `pyalcs` | Which ALP generalization to use. |
| `--agent acs2\|acs2er` | `acs2` | Which agent. |
| `--eval-interval <n>` | `6000` | Trials between knowledge evaluations. |
| `--rss-cap-gb <f>` | `5.6` | Abort if peak RSS exceeds this. Raise it at 264 bits. |

Knowledge is exhaustive for k ≤ 20 and **sampled** (50,000 inputs, fixed evaluation
seed) for larger sizes — state that caveat when reporting.

### Diagnostics

All off by default and read-only over the population, so turning one on cannot
change a result:

| Flag | Answers |
|---|---|
| `--log-trajectory` | The S-curve: knowledge, reliable count, specificity, population. |
| `--log-accuracy` | How often the greedy choice answers correctly — the metric the literature reports. |
| `--log-coverage` | Knowledge split into four action × correctness classes. **This is where a 0.75 or 0.50 ceiling is explained** — it means whole classes are empty. |
| `--log-quadrant-detail` | Per class, the share covered by any classifier and the best quality among them — separates "never created" from "never reliable". |
| `--log-diagnostics` | Population-wide specificity, quality spread, mark density, experience, address-bit enrichment. |

### ACS2ER

Both agents run the **same** experiment code; only the agent is swapped.

```bash
./target/release/acs2-bench --agent acs2er --out /tmp/maze_acs2er.csv
./target/release/mpx_reach  --agent acs2er --sizes 20 --n-exp 3
```

| Flag | Default | Meaning |
|---|---|---|
| `--er-buffer-size` | `10000` | Replay buffer capacity; oldest evicted first. |
| `--er-min-samples` | `1000` | Warmup — below this the agent does not learn at all. |
| `--er-samples-number` | `3` | Samples replayed per step, drawn without replacement. |

ACS2ER performs `--er-samples-number` learning applications per step and none on
the current transition, so per-trial wall time is roughly that multiple of ACS2's,
and it is memory-hungry: at k=70, `m = 13` needs well over 8 GB. Long comparisons
belong on a cluster.

### On a cluster (SLURM)

```bash
sbatch --export=ALL,TAG=<tag>,U_MAX=11,ENCODING=outcome,EPSILON=1 \
    slurm/mpx_reach.sh <size> <seed> <time_cap_secs> [extra flags]
./slurm/mpx_status.sh
```

Results land outside the checkout. Pull them into the repository and rebuild the
archive with a single command:

```bash
./tools/sync_runs.sh --commit
```

Do this after every batch. Until it runs, a result that cost days of compute exists
in exactly one copy, on the cluster.

## Reproducing the reports

Every figure and table regenerates from committed CSVs, never from raw logs:

```bash
python3 tools/parse_mpx_logs.py reports/slurm_*.out    # logs  -> CSVs
python3 tools/rebuild_tables.py                        # CSVs  -> reports/MPX<k>_runs.md
uv run --project tools python tools/plot_mpx.py --size 135 \
    --encoding flip --epsilon 1 --u-max 11 --suffix _canonical_eps1
```

**Always narrow a plot to one experimental arm at k ≥ 135.** One seed now has runs
under both encodings, two epsilons and several `u_max` values; splicing them makes
a curve that never happened. The tool refuses a mixed selection and names what is
mixed. `tools/README.md` documents the rest, including the
`seed = base_seed + repeat` rule that silently corrupts results if ignored.

The pyalcs comparison needs the pinned baseline environment
(Python 3.10, `gym==0.23.0`, `numpy==1.23.5`, managed by [`uv`]) and the reference
repositories cloned as siblings — see `baseline/README.md`:

```bash
cd baseline && uv sync
uv run --project baseline python baseline/run_pyalcs_maze.py \
    Maze4-v0 Maze5-v0 Maze7-v0 Woods1-v0 Woods100-v0 \
    --repeats 10 --csv reports/bench_pyalcs.csv
uv run --project baseline python baseline/compare_bench.py
```

### Benchmark methodology

The speedup figure is only meaningful under these constraints, which are the
project's own rules from `ARCHITECTURE.md`:

- Optimized Rust only — the timed binary is `target/release/acs2-bench`.
- Sequential and uncontended — the two sides never overlap, same machine.
- Symmetric timed region — only explore + exploit are timed; metrics are computed
  after it on both sides.
- GA off on both sides, identical protocol, identical per-maze step caps.
- Two readings reported: per-maze `t_py/t_rust` and the total-time headline.

## Reading the code

| Document | Read it for |
|---|---|
| [`docs/ACS2_PRIMER.md`](docs/ACS2_PRIMER.md) | ACS2 from first principles, anchored to this code. Start here if the algorithm is new to you. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Implementation decisions and the named hazards. |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | What the project is, the pinned protocol, the fidelity rules. |
| [`docs/SPEC_PYALCS.md`](docs/SPEC_PYALCS.md) | The reference semantics the oracle enforces. |
| [`docs/ACS2_RULE_DUMPS_GUIDE.md`](docs/ACS2_RULE_DUMPS_GUIDE.md) | How to read a learned population. |
| [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md) | Live experiment state — what is running and what is still open. |
| [`reports/MPX_final.md`](reports/MPX_final.md) | The multiplexer results as a narrative. |
| [`reports/MPX_literature_review.md`](reports/MPX_literature_review.md) | What has been published on this environment. |

## Scope

Actor-Critic is **not** implemented; the architecture leaves seams for it (an
`ActionSelector` trait and an injected RL bootstrap value). Prioritised replay —
the thesis contribution — is not implemented either: ACS2ER provides uniform replay
and the measurements that a prioritisation criterion has to beat.

Determinism is guaranteed by an injected RNG: the same seed and configuration
reproduce a run trial for trial, verified across architectures. Trials-to-success
is therefore the machine-independent metric; wall-clock is not.

[`uv`]: https://docs.astral.sh/uv/
