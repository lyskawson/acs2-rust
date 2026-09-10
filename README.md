# acs2-rust

An independent Rust implementation of **ACS2** (Anticipatory Classifier System) and
**ACS2ER** (ACS2 with experience replay), written for a master's thesis on prioritised
sample selection from replay memory.

It solves the 70- and 135-bit Boolean multiplexer at full knowledge; published ACS/ACS2
results stop at 20–37 bits. Numbers and method are in
[`reports/MPX_final.md`](reports/MPX_final.md); every individual run is tabulated in
`reports/MPX<k>_runs.md`.

## Quick start

Needs a stable Rust toolchain and nothing else.

```bash
git clone https://github.com/lyskawson/acs2-rust && cd acs2-rust
cargo build --release
cargo test --workspace --release                      # 98 tests, including reach regressions
./target/release/mpx_reach --sizes 20 --n-exp 1       # solves in seconds
./target/release/acs2-bench                           # maze suite, ~2 s
```

If those four commands work you have everything. The Python parts below are optional and
only needed to regenerate figures or re-run the pyalcs comparison.

## Running experiments

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
- **The write is atomic** (staged under a name derived from the destination plus the
  process id, then renamed), so no reader ever sees a half-written file and a job killed
  mid-save leaves the previous checkpoint intact. It is not `fsync`ed: a node losing power
  can still fall back to the previous checkpoint, which is what `--checkpoint-every`
  bounds. The process id keeps two jobs *on one node* off each other's staging file; it
  is not unique across nodes, which is why the sequencing above is the actual guarantee.
- **An evaluation owed when a job stopped is paid before the next one trains.** A
  resource cap is checked before the evaluation block, so a job can stop on the very
  batch a measurement was due; resuming straight into another batch would shift that
  measurement and every later one, and trials-to-success with them.
- **The archive reads a chain as one run.** Each job writes its own log; the wrapper
  emits a `run-segment:` line and `tools/parse_mpx_logs.py` collapses the segments into
  one run, supersedes work a killed job did after its last checkpoint, and keeps one
  verdict instead of one per job.

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
| `acs2-core` | Classifier, population, ALP, RL, GA, action selection, injected RNG, both agents behind `trial::LearningAgent`, and the `environment::Environment` trait. No I/O or environment-specific knowledge. |
| `acs2-envs` | Implementations of `acs2_core::environment::Environment`: 8-sensor maze (27 geometries in `mazes/`) and the multiplexer with both encodings. |
| `acs2-bench` | The `acs2-bench` and `mpx_reach` binaries. |

`baseline/` pinned Python reference · `tools/` parsing and plotting (a **separate** Python
project, so plotting deps never perturb the pinned baseline) · `slurm/` cluster scripts ·
`fixtures/` golden vectors · `reports/` results.

| Document | Read it for |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Implementation decisions and the named hazards. |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | The pinned protocol and the fidelity rules. |
| [`docs/SPEC_PYALCS.md`](docs/SPEC_PYALCS.md) | The reference semantics the oracle enforces. |
| [`reports/MPX_final.md`](reports/MPX_final.md) | The multiplexer results as a narrative. |
| [`tools/README.md`](tools/README.md) | The analysis pipeline and its traps. |

## Correctness and provenance

`hendrykik/acs2vcp-python` (pyalcs) is a **correctness oracle and a timing baseline**, not
a target. 800 random single-learning-step inputs were run through both an instrumented
pyalcs and this core: **761 of 761 deterministic cases agree, zero divergence** (the other
39 are excluded for cause — random mark selection, and a pyalcs bug this implementation
does not reproduce). ACS2ER is validated the same way. Details in
[`reports/P8_differential.md`](reports/P8_differential.md).

The shipped defaults deliberately match pyalcs — GA off, `u_max = 100000`, the pinned maze
protocol — because differential validation and a fair timing comparison both need an
identical configuration on both sides. All of them are flags. On that protocol this
implementation runs **139× faster than pyalcs** in total time, at exploit steps-to-goal
within 1.00–1.07×, i.e. the same algorithm rather than a faster different one.

Runs are deterministic from an injected RNG: the same seed and configuration reproduce
learning trial for trial on the tested 64-bit platforms (Apple M1 and x86_64 Bem2).
No equivalence is claimed across pointer widths: random `usize` sampling can differ
on 32-bit targets. Trials-to-success is the comparison metric on these tested platforms;
wall-clock depends on the machine and instrumentation.

## Scope

Actor-Critic is **not** implemented; the architecture leaves seams for it. Prioritised
replay — the thesis contribution — is not implemented either: ACS2ER provides uniform
replay and the measurements a prioritisation criterion has to beat.

## License

MIT — see [`LICENSE`](LICENSE). If you build on this for your own thesis or paper, a
citation is appreciated but not required.
