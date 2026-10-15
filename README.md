# acs2-rust

An independent Rust implementation of **ACS2** (Anticipatory Classifier System) and
**ACS2ER** (ACS2 with experience replay), supporting the master's thesis
**Experience and Goal Selection Based on Trajectory Utility in the ACS2 Anticipatory
Classifier System**.

The thesis develops an ACS2HER extension that uses the utility of stored trajectories to
choose goals from training tasks and regulates the share of those experiences against
HER's, with an optional conservative value correction. The intended comparison is with
ACS2, ACS2ER, ACS2HER and ACS2VCP on discrete multi-goal mazes with sparse rewards at
equal numbers of environment interactions. The goal tasks, measurement layer, trajectory store, episodic replay, HER variants
and value consistency prioritization are implemented; trajectory utility remains planned.

Two lines of work share this repository:

| Line | Branch | Purpose and implementation record |
|---|---|---|
| Trajectory utility | `feature/trajectory-utility` | The thesis; [goal-task architecture](docs/ARCHITECTURE.md#goal-port) and [experiment index](experiments/README.md). |
| MPX scaling | `feature/mpx264` | Additional research; [live handoff](docs/AGENT_HANDOFF.md) and the multiplexer results below. |

It solves the 70- and 135-bit Boolean multiplexer at full knowledge; published ACS/ACS2
results stop at 20–37 bits. Numbers and method are in
[`reports/MPX_final.md`](reports/MPX_final.md); every individual run is tabulated in
`reports/MPX<k>_runs.md`.

## Quick start

Needs a stable Rust toolchain and nothing else.

```bash
git clone https://github.com/lyskawson/acs2-rust && cd acs2-rust
cargo build --release
cargo test --workspace --release                    # includes reach regressions
./target/release/mpx_reach --sizes 20 --n-exp 1       # solves in seconds
./target/release/acs2-bench                           # maze suite, ~2 s
```

These commands build and check the Rust workspace. Python is needed for the experiment
analyses, archive/plotting tests and the pinned reference oracles; it is separate from
the Rust implementation.

Archive and plotting tests: `uv run --project tools python -B -m unittest discover -s tools -p 'test_*.py'`
using the separate tools environment.

## Running experiments

### Goal-task measurement — `acs2-measure`

Use the authorized TU cluster clone for learning runs. This example measures both
implemented agents on Maze4 with all coordinate goals, cap five, seed 42 and three
environment-interaction checkpoints; choose an output path outside every Git checkout:

```bash
cargo run --release -p acs2-measure -- \
    --task maze4 --cap 5 --pool full --encoding coordinates \
    --agents acs2,acs2er --seeds 42 --targets 1000,3000,10000 \
    --out /tmp/goal-measure.jsonl
```

Evaluation is read-only, uses separate random streams and runs after a complete episode
meets each nominal point. Rows record actual interactions, success, the random-policy
floor, reachability within the cap, value diagnostics, update/matching counts and
component memory sizes. See [the measurement contract](docs/ARCHITECTURE.md#goal-measurement-layer)
and [the experiment index](experiments/README.md) for the registered protocols, archived
data provenance and analysis commands. The maze benchmark below is the P9
validation/timing protocol.

### Multiplexer — `mpx_reach`

```bash
./target/release/mpx_reach --sizes 20 --n-exp 1 --seed 42 --log-trajectory
```

Sizes are not continuous: `k = a + 2^a` gives 6, 11, 20, 37, 70, 135, 264, 521. **20 and
37 finish on a laptop; 70 takes hours and 135 takes days** — those belong on a cluster.

| Flag | Default | Meaning |
|---|---|---|
| `--sizes <a,b,...>` | `37,70,135` | Which multiplexers to run. |
| `--seed <s>` / `--n-exp <k>` | `42` / `3` | Repeat *r* runs at seed `s + r`. |
| `--time-cap-secs <s>` | `600` | Wall-clock budget per repeat. Raise it or the run stops early. |
| `--u-max derived\|<int>` | `100000` | ALP generalization limit; the default keeps it effectively inactive. Select `derived` explicitly for `a + 2` (Pyalcs) or `a + 3` (Butz). The SLURM wrapper selects `derived`; the k=135 runs use 11. |
| `--encoding flip\|outcome` | `flip` | `flip` is canonical. `outcome` gives a wrong answer an observable effect: ~10x cheaper, but a different problem, so not comparable to the multiplexer literature. |
| `--epsilon <f>` | `0.8` | Exploration rate. `1` removes the greedy bias and is what solves k=135 canonically. |
| `--agent acs2\|acs2er` | `acs2` | Which agent. |
| `--eval-interval <n>` | `6000` | Trials between knowledge evaluations. |
| `--rss-cap-gb <f>` | `5.6` | Abort above process peak RSS; use `--isolate-repeats` for a fresh process per repeat. |
| `--isolate-repeats` | off | Isolate each size/repeat so prior runs cannot contaminate its RSS peak or cap. |
| `--strict-resource-limits` | off | Recheck time and RSS after evaluation and diagnostics, before SUCCESS. |
| `--checkpoint-path <p>` | off | Save the learning state there and resume from it if it exists. A resumed run is identical to an uninterrupted one, trial for trial. Requires `--n-exp 1` and one size. |
| `--checkpoint-every <n>` | `0` | Trials between saves; `0` saves only when the run stops. A save is rounded up to the next 500-trial batch boundary. |
| `--checkpoint-allow-eval-change` | off | Permit resuming a checkpoint under a different `--eval-interval`. Refused without it: two sampling rates spliced into one run make trials-to-success meaningless. |
| `--alp-gen-variant pyalcs\|butz\|butz-checked` | `pyalcs` | `butz-checked` fixes exhausted-condition counting; `butz` preserves historical trajectories. |

Diagnostics are off by default and preserve population and learning RNG state: learning
is identical at equal trial counts. Their evaluation cost can change when a wall-clock
cap stops a run.
`--log-trajectory` (the S-curve), `--log-accuracy` (the metric the literature reports),
`--log-coverage` (**start here when knowledge sticks near 0.75 or 0.50** — check for
classes without reliable coverage), `--log-quadrant-detail`, `--log-diagnostics`.

### Checkpointing — running past the queue limit

One k=264 seed is 700-4500 CPU-hours against a hard 504 h per job, so no single job
can finish one. `--checkpoint-path` saves everything a trial depends on -- population,
both RNG streams, the trial and ALP clocks, the accumulated wall time and the peak
trackers -- and resuming reproduces an uninterrupted run **trial for trial**, which
`acs2-bench/tests/reach_regressions.rs` pins for both ACS2 and ACS2ER.

```bash
# the same command for every job in the chain: it starts fresh, then resumes
./target/release/mpx_reach --sizes 264 --n-exp 1 --seed 42 --time-cap-secs 1800000 \
    --checkpoint-path ~/mpx_runs/checkpoints/mpx264_s42.ckpt --checkpoint-every 100000
```

Three things the file format guarantees, and one it does not:

- **Resuming under another configuration is refused, not merged.** The checkpoint
  carries the seed, encoding, epsilon, `u_max`, GA and agent settings it was written
  for and the run aborts if they differ. The stopping limits are deliberately not part
  of that identity -- a chained run raises them per job.
- **A closed run is never relearned.** A checkpoint that records SUCCESS makes the next
  job restate that verdict — marked `reopened=true` — and exit without touching the state,
  so the tail of an over-long chain costs a few seconds each. It restates rather than
  staying quiet because the job that closed the run can be killed between saving its
  checkpoint and printing, and then the reopening job's line is the only place the SUCCESS
  reaches the archive. The parser keeps one verdict per run.
- **The segments must not overlap.** One checkpoint is one run's learning state, and
  nothing locks it. Chain the jobs so only one runs at a time:

  ```bash
  previous=$(sbatch --parsable --export=ALL,TAG=k264,CHECKPOINT=on … slurm/mpx_reach.sh 264 42 1800000)
  for _ in $(seq 9); do
      previous=$(sbatch --parsable --dependency=afterany:$previous \
          --export=ALL,TAG=k264,CHECKPOINT=on … slurm/mpx_reach.sh 264 42 1800000)
  done
  ```

  `afterany`, not `afterok`: a segment that stops on its wall clock has done its job.
- **The write is staged, flushed and renamed**, and the generation it replaces is kept as
  `<name>.prev`. Version 3 requires a SHA-256 footer covering every preceding byte, including
  the `end` terminator; a valid-looking change to an RNG position or float is refused.
  Older versions are refused rather than silently accepted without integrity protection.
  Rotation still has a gap between moving the old checkpoint and publishing the new one:
  if interrupted there, restore `.prev` to the main path before submitting another job.
  Recovery is manual, and directory synchronization is currently best-effort.
  The staging process id is unique only within a node; job dependencies remain necessary.
- **An evaluation owed when a job stopped is paid before the next one trains.** A
  resource cap is checked before the evaluation block, so a job can stop on the very
  batch a measurement was due; resuming straight into another batch would shift that
  measurement and every later one, and trials-to-success with them.
- **The archive reads a chain as one run.** Each job writes its own log; the wrapper
  emits a `run-segment:` line and `tools/parse_mpx_logs.py` collapses the segments into
  one run, supersedes work a killed job did after its last checkpoint, and keeps one
  verdict instead of one per job.
  Include the initial segment: the parser refuses missing or timezone-free timestamps,
  unresolved timestamp ties, reversed requeue attempts, and resume points unsupported by
  the preceding log's progress. A checkpoint saved between measurements may require more
  evidence before a sparse chain can be archived; the parser does not guess its order.

`slurm/mpx_reach.sh` takes `CHECKPOINT=on` and derives the path from size, seed and tag
so two jobs cannot share one learning state, and gives each job its own log file.

A terminal row prints `knowledge=unmeasured` when the final population has not been
measured at that trial. It does not rerun evaluation after a resource cap. SUCCESS
knowledge values are unchanged by this reporting correction. `--strict-resource-limits`
and `--isolate-repeats` are explicit protocol changes and are recorded in new headers;
without them, historical stopping/resource behavior remains available.

### Maze — `acs2-bench`

```bash
./target/release/acs2-bench                    # pinned protocol -> reports/bench_rust.csv
./target/release/acs2-bench --agent acs2er --out /tmp/er.csv
```

No flags gives the pinned protocol: five mazes, `n_exp=10`, seed 42, GA off, 500 explore
+ 3×200 exploit. Other flags: `--mazes`, `--n-exp`, `--seed`, `--do-ga`,
`--explore-trials`, `--exploit-trials`, `--exploit-phases`, `--out`.

### ACS2ER

`--er-buffer-size` (10000), `--er-min-samples` (1000, warmup — below it the agent does not
learn at all), `--er-samples-number` (3, replayed per step). It performs that many learning
applications per step and none on the current transition, so per-trial time is roughly that
multiple of ACS2's, and it is memory-hungry: k=70 at `m = 13` needs well over 8 GB.

### On a cluster (SLURM)

The wrapper expects a clone at `~/acs2-rust-repo` on the cluster and a release binary
at `target/x86_64-unknown-linux-musl/release/mpx_reach` inside it. Prepare that clone
and build there with `cargo build --release --target x86_64-unknown-linux-musl`
(install the Rust target and musl linker first); the local quick start does not
prepare this cluster binary. Run the submission command from that clone.

```bash
sbatch --export=ALL,TAG=<tag>,U_MAX=11,ENCODING=outcome,EPSILON=1 \
    slurm/mpx_reach.sh <size> <seed> <time_cap_secs> [extra flags]
# resumable across jobs (k=264 needs this):
sbatch --export=ALL,TAG=<tag>,CHECKPOINT=on,CHECKPOINT_EVERY=100000 \
    slurm/mpx_reach.sh <size> <seed> <time_cap_secs>
./slurm/mpx_status.sh
./tools/sync_runs.sh --commit      # pull results back into the repo
```

Run output lands outside the checkout, so **`sync_runs.sh` is not optional** — until it
runs, a result exists only on the cluster.

## Things that will bite you

- **Always time the release binary.** A debug build can be slower than CPython.
- **`--time-cap-secs` defaults to 600.** Anything at k≥70 needs far more, and a run that
  hits the cap reports TIME-LIMITED, not failure.
- **A flat knowledge value is not a converged one.** At k=135 one seed sat at exactly
  0.0000 on a coverage class for 404 M trials, then filled it and finished 23 M later.
- **`seed = base_seed + repeat`** — an `n_exp=3` log at seed 42 holds seeds 42, 43 and 44.
- **Knowledge is exhaustive only for k ≤ 20**; above that it is sampled over 50,000 inputs
  at a fixed evaluation seed. Say so when reporting.
- **Narrow plots to one experimental arm** (`--encoding`, `--epsilon`, `--u-max`,
  `--agent`, `--do-ga`, `--er-*`) and one `--source` per seed. The plotter rejects
  mixed arms and independent runs sharing a seed; use `--block` for repeated headers.

## Regenerating reports and figures

Optional, needs [`uv`](https://docs.astral.sh/uv/):

```bash
./tools/sync_runs.sh --local     # full local archive -> CSVs and tables
# Omit --local to pull current cluster logs first.
uv run --project tools python tools/plot_mpx.py --size 135 \
    --figures reach --encoding flip --epsilon 1 --u-max 11 --agent acs2 \
    --source slurm_mpx135_s43_eps1_u11.out \
    --source slurm_mpx135_s42_eps1_u11.out --suffix _canonical_eps1
```

The pyalcs comparison additionally needs the pinned baseline (Python 3.10,
`gym==0.23.0`, `numpy==1.23.5`) and the reference repositories cloned as siblings — see
[`baseline/README.md`](baseline/README.md):

```bash
uv sync --project baseline
uv run --project baseline python baseline/run_pyalcs_maze.py \
    Maze4-v0 Maze5-v0 Maze7-v0 Woods1-v0 Woods100-v0 --repeats 10 \
    --csv reports/bench_pyalcs.csv
uv run --project baseline python baseline/compare_bench.py
```

## Layout

| Crate | Purpose |
|---|---|
| [`acs2-core`](acs2-core/src/lib.rs) | Shared representation, learning mechanisms, task ports, support code, agent/trial contract and checkpoint contract/format; module map below. |
| [`acs2-envs`](acs2-envs/src/lib.rs) | [Single-goal maze](acs2-envs/src/maze/mod.rs), [geometry catalogue](acs2-envs/src/maze/geometries/mod.rs), [goal environments](acs2-envs/src/goal/mod.rs), [task/role registry](acs2-envs/src/roles.rs), and [multiplexer](acs2-envs/src/multiplexer.rs) with both encodings. |
| [`acs2-bench`](acs2-bench/src/main.rs) | P9 maze benchmark and the [MPX reach runner](acs2-bench/src/bin/mpx_reach.rs). |
| [`acs2-measure`](acs2-measure/src/lib.rs) | [Goal-agent and environment contracts, runner, evaluation and counters](acs2-measure/src/runner.rs), [task construction and preset](acs2-measure/src/task.rs), [floor/reachability references](acs2-measure/src/reference.rs), [CLI](acs2-measure/src/main.rs), [output guard](acs2-measure/src/output.rs), and [replay diagnostic adapter](acs2-measure/src/trajectory.rs). |
| [`acs2-agents`](acs2-agents/src/lib.rs) | Every learning agent: ACS2 in `acs2/`; ACS2ER, its FIFO memory/parameters and one-update variant in `acs2er/`; original-only episodic control in `acs2er/episodic.rs`; hindsight full/values/shortened variants in `acs2her/`; independent-head hindsight learners and exact consistency sampling in `acs2vcp/`; shared episodic mechanisms in `episodic_replay.rs`. Depends on core and trajectory storage, never on measurement. |
| [`acs2-trajectory`](acs2-trajectory/src/lib.rs) | [Raw episode store](acs2-trajectory/src/store.rs), [goal selection and admissibility](acs2-trajectory/src/selection.rs), [objective-based relabeling](acs2-trajectory/src/relabel.rs), [sampler and route counters](acs2-trajectory/src/sampler.rs), [exact composition](acs2-trajectory/src/composition.rs), and [objective cost accounting](acs2-trajectory/src/cost.rs). |

The measurement crate constructs the named presets in `runner.rs`, adapts episodic and
one-update agents in [`agents.rs`](acs2-measure/src/agents.rs), and embeds the shared
archived-name mapping in [`agent_names.rs`](acs2-measure/src/agent_names.rs).

Core modules are grouped by role under [`acs2-core/src/`](acs2-core/src/). Each file
below defines the module of the same name under its group:

| Role | Modules and files |
|---|---|
| Rule representation and population | [`representation/symbol.rs`](acs2-core/src/representation/symbol.rs), [`perception.rs`](acs2-core/src/representation/perception.rs), [`condition.rs`](acs2-core/src/representation/condition.rs), [`effect.rs`](acs2-core/src/representation/effect.rs), [`mark.rs`](acs2-core/src/representation/mark.rs), [`classifier.rs`](acs2-core/src/representation/classifier.rs), [`population.rs`](acs2-core/src/representation/population.rs). |
| Shared learning mechanisms | [`learning/alp.rs`](acs2-core/src/learning/alp.rs), [`ga.rs`](acs2-core/src/learning/ga.rs), [`rl.rs`](acs2-core/src/learning/rl.rs), [`subsumption.rs`](acs2-core/src/learning/subsumption.rs), [`action_selection.rs`](acs2-core/src/learning/action_selection.rs), [`replay.rs`](acs2-core/src/learning/replay.rs) (`ReplaySample` and the full replay learning step). |
| Agent and trial contract | [`trial.rs`](acs2-core/src/trial.rs) defines `LearningAgent`, `TrialMetrics`, `TruncationMode` and the shared exploitation loop. |
| Task ports | [`ports/environment.rs`](acs2-core/src/ports/environment.rs) and [`ports/goal.rs`](acs2-core/src/ports/goal.rs), including the goal-conditioned adapter. |
| Support | [`support/config.rs`](acs2-core/src/support/config.rs), [`rng.rs`](acs2-core/src/support/rng.rs), [`knowledge.rs`](acs2-core/src/support/knowledge.rs), [`measurement.rs`](acs2-core/src/support/measurement.rs) (opt-in match counter). |
| Checkpoint contract and format | [`checkpoint.rs`](acs2-core/src/checkpoint.rs) defines `AgentState` and `Checkpointed`; [`checkpoint/codec.rs`](acs2-core/src/checkpoint/codec.rs) owns format 3, integrity checks and file storage. |

Callers use the defining modules, for example `acs2_agents::acs2::Agent`,
`acs2_agents::acs2er::replay::ReplayConfiguration` and
`acs2_core::ports::goal::GoalEnvironment`. Former paths have no compatibility exports
or facade modules. `acs2-trajectory` depends only on core; benchmarks and measurement
construct agents from `acs2-agents`. Every agent type is named `Agent` in its defining
module; callers using several agents qualify modules or alias imports.

The measurement CLI names are `acs2`, `acs2er`, `acs2er_one_update`,
`acs2er_episodic`, `acs2her`, `acs2her_values` and `acs2her_values_shortened`.
ACS2VCP adds `acs2vcp_k2`, `acs2vcp_k4`, `acs2vcp_k2_uniform` and
`acs2vcp_k4_uniform`; the positive-temperature presets accept
`--vcp-temperature` from 1 through 24, with default 1. Uniform presets fix T=0.
These four names register full hindsight copy updates, future goals, every-transition
admissibility and proportion 2/3. Other hindsight choices have no measurement name.
The [dated name table](docs/ARCHITECTURE.md#agent-run-names-2026-10-14) relates them to
closed experiment records. Archived rows and fixtures are read through the shared
mapping in `experiments/grid/agent_names.json`; the CLI rejects old names with their
replacements.

All learning agents live in `acs2-agents`, with one module directory per family and
variants within it. Shared mechanisms stay in shared crates. The running MPX chain
continues from its separate branch and clone; these moves do not rebuild it. The HER
experiment is complete: validation tuning, the twenty-seed grid and matched-proportion
controls used 34.6125 of sixty CPU-hours. Its registration and analysis tools are in
`experiments/her-variants/`; [hindsight copy updates and held-pair replay results](docs/ARCHITECTURE.md#hindsight-copy-updates-and-held-pair-replay-results)
document the update rules, comparisons and finite measurement horizon.

`baseline/` pinned Python reference · `tools/` parsing and plotting (a **separate** Python
project, so plotting deps never perturb the pinned baseline) · `slurm/` cluster scripts ·
`fixtures/` golden vectors · `reports/` results · [`experiments/`](experiments/README.md)
registered protocols and analyses; raw TU data stays outside the repository.

| Document | Read it for |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Implementation decisions and the named hazards. |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | The pinned protocol and the fidelity rules. |
| [`docs/SPEC_PYALCS.md`](docs/SPEC_PYALCS.md) | The reference semantics the oracle enforces. |
| [`experiments/README.md`](experiments/README.md) | Experiment questions, measured tasks/agents/seeds/budgets, provenance and reproduction commands. |
| [`reports/MPX_final.md`](reports/MPX_final.md) | The multiplexer results as a narrative. |
| [`tools/README.md`](tools/README.md) | The analysis pipeline and its traps. |

## Correctness and provenance

`hendrykik/acs2vcp-python` (pyalcs) is a **correctness oracle and a timing baseline**, not
a target. 800 random single-learning-step inputs were run through both an instrumented
pyalcs and this core: **761 of 761 deterministic cases agree, zero divergence** (the other
39 are excluded for cause — random mark selection, and a pyalcs bug this implementation
does not reproduce). ACS2ER is validated the same way. Details in
[`reports/P8_differential.md`](reports/P8_differential.md).

Configuration defaults retain the pyalcs parameter values — GA off, `u_max = 100000` —
and the benchmark selects the pinned maze protocol for differential validation and a
fair timing comparison. Default learning behavior includes the corrected truncation
bootstrap; the P8/P11/P9 validation callers explicitly select `TruncationMode::Pyalcs`.
On that pinned protocol this
implementation runs **139× faster than pyalcs** in total time, at exploit steps-to-goal
within 1.00–1.07×, i.e. the same algorithm rather than a faster different one.

Runs are deterministic from an injected RNG: the same seed and configuration reproduce
learning trial for trial on the tested 64-bit platforms (Apple M1 and x86_64 Bem2).
No equivalence is claimed across pointer widths: random `usize` sampling can differ
on 32-bit targets. Trials-to-success is the comparison metric on these tested platforms;
wall-clock depends on the machine and instrumentation.

## Scope

ACS2 and ACS2ER are implemented, as are the goal environments, read-only measurement
layer, raw trajectory store, goal-selection primitives and objective-based relabeling.
ACS2ER supplies uniform replay; ACS2HER supplies hindsight replay and ACS2VCP adds
independent heads and consistency prioritization with an unprioritized ensemble control.
The trajectory-utility agent and optional conservative value correction remain planned. Trajectory utility
selects training-task goals and regulates experience composition; it is not simply
priority sampling from ACS2ER's replay buffer.

Actor-Critic is **not** implemented; the architecture leaves seams for it.

## License

MIT — see [`LICENSE`](LICENSE). If you build on this for your own thesis or paper, a
citation is appreciated but not required.
