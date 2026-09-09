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
cargo test --release                                  # 73 tests, ~1 s
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
| `--rss-cap-gb <f>` | `5.6` | Abort above this peak RSS. |

Diagnostics are off by default and preserve population and learning RNG state: learning
is identical at equal trial counts. Their evaluation cost can change when a wall-clock
cap stops a run.
`--log-trajectory` (the S-curve), `--log-accuracy` (the metric the literature reports),
`--log-coverage` (**start here when knowledge sticks near 0.75 or 0.50** — check for
classes without reliable coverage), `--log-quadrant-detail`, `--log-diagnostics`.

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
- **Narrow plots to one experimental arm** (`--encoding`, `--epsilon`, `--u-max`) at
  k ≥ 135, or curves from unrelated runs get spliced together. The tool refuses to.

## Regenerating reports and figures

Optional, needs [`uv`](https://docs.astral.sh/uv/):

```bash
python3 tools/parse_mpx_logs.py reports/slurm_*.out    # logs -> CSVs
python3 tools/rebuild_tables.py                        # CSVs -> reports/MPX<k>_runs.md
uv run --project tools python tools/plot_mpx.py --size 135 \
    --encoding flip --epsilon 1 --u-max 11 --suffix _canonical_eps1
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
| [`docs/ACS2_PRIMER.md`](docs/ACS2_PRIMER.md) | ACS2 from first principles, anchored to this code. **Start here if the algorithm is new to you.** |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Implementation decisions and the named hazards. |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | The pinned protocol and the fidelity rules. |
| [`docs/SPEC_PYALCS.md`](docs/SPEC_PYALCS.md) | The reference semantics the oracle enforces. |
| [`docs/ACS2_RULE_DUMPS_GUIDE.md`](docs/ACS2_RULE_DUMPS_GUIDE.md) | How to read a learned population. |
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
