# acs2-rust — an idiomatic Rust port of canonical ACS2

This is a Rust implementation of **ACS2** (an Anticipatory Learning Classifier
System, Butz & Stolzmann). Its port target is the **pyalcs** library
(`hendrykik/acs2vcp-python`, the `lcs/agents/acs2` core), which is treated as the
source of truth for behaviour.

The purpose is to **measure how much faster a Rust ACS2 runs than the Python
reference** on the same mazes, same protocol, same machine — a baseline to decide
whether to commit to Rust before a possible Actor-Critic extension of ACS2.

**This phase is baseline-only.** Actor-Critic is **not** implemented. The
architecture only leaves clean seams for it: an `ActionSelector` trait (the actor
seam) and an injected RL bootstrap value (the critic seam). See
`ARCHITECTURE.md` and `PROJECT_CONTEXT.md` §7.

---

## Results

Numbers below are taken verbatim from `reports/P9_comparison.md` and
`reports/P8_differential.md`. GA OFF, `n_exp=10`, protocol = 500 explore (ε=0.8) +
3×200 exploit, final exploit-window mean steps, sequential timed runs on one
machine.

### Speedup (P9) — `t_python / t_rust`

| Maze | pyalcs total_s | Rust total_s | speedup |
|---|---|---|---|
| Maze4-v0 | 63.569 | 0.415 | 153.3× |
| Maze5-v0 | 93.924 | 0.756 | 124.2× |
| Maze7-v0 | 117.444 | 0.916 | 128.3× |
| Woods1-v0 | 19.840 | 0.066 | 298.8× |
| Woods100-v0 | 11.428 | 0.047 | 242.6× |

**Headline — total-time speedup (Σt_py / Σt_rust): 139.2×** (pyalcs 306.20s vs
Rust 2.20s over 5 mazes × 10 repeats). Secondary reading: mean of per-maze
speedups (equal weight) is 189.4×.

Exploit steps-to-goal agree across every maze (ratios 1.00–1.07×), confirming Rust
runs the *same* ACS2 algorithm — not a faster, different one. The correctness gate
(every maze within 2× of pyalcs steps) **passes**.

### Differential validation (P8)

800 random deterministic `(population, p0, action, p1, time, reward)` inputs were
run through both an instrumented (unmodified) pyalcs and the Rust core, comparing
match set, action set, next-state match set, RL bootstrap, and population after one
learning step.

| Bucket | Count | Compared exactly? |
|---|---|---|
| Deterministic, RNG-free | **761** | Yes — **0 divergences** |
| RNG-excluded (≥2 mark candidates; pick is random by design) | 24 | No |
| pyalcs-bug-excluded (`apply_alp` mid-iteration skip) | 15 | No |
| **Total generated** | 800 | |

The 39 excluded cases are **legitimately excluded** (RNG or a pyalcs bug Rust does
not reproduce), not failures. Result: **761/761 deterministic cases agree with zero
divergence.**

---

## Workspace layout

Cargo workspace, Clean Architecture (the domain crate has no I/O and no environment
knowledge):

| Crate | Kind | Purpose |
|---|---|---|
| `acs2-core` | lib | Pure ACS2 domain: classifier, population, ALP, RL, GA, action selection, config, injected RNG, and the shared trial loop. |
| `acs2-envs` | lib | The `Environment` (Gymnasium-style) trait impl: the 8-sensor maze plus geometry definitions in `mazes/` — the 5 canonical pyalcs mazes ported from `gym_maze` (the default run) and 22 additional ounold/ALCS mazes (opt-in via `--mazes`). |
| `acs2-bench` | bin | Runs the maze suite under the benchmark protocol, computes metrics + timing, emits CSV. |

The Python reference (pyalcs baseline + dump/compare scripts) lives under
`baseline/`, whose lockfile is pinned as part of the differential-validation
evidence. Figure generation for the MPX experiments is a **separate** Python
project under `tools/` (see `tools/README.md`) precisely so that plotting
dependencies never perturb that pinned environment. Cluster job scripts are in
`slurm/`. Source-of-truth docs: `PROJECT_CONTEXT.md`, `SPEC_PYALCS.md`,
`BUILD_PLAN.md`, `ARCHITECTURE.md`; the live experiment state and decision tree
are in `docs/AGENT_HANDOFF.md`. Generated artefacts and reports are in
`reports/` (figures in `reports/figures/`); golden vectors in `fixtures/`.

---

## Build and run

### Rust benchmark

```bash
cargo build --release
./target/release/acs2-bench
```

With no flags the runner uses the pinned protocol: all five mazes
(Maze4-v0, Maze5-v0, Maze7-v0, Woods1-v0, Woods100-v0), `n_exp=10`, seed 42,
**GA OFF**, 500 explore + 3×200 exploit. The CSV lands at `reports/bench_rust.csv`.

Flags (all optional): `--mazes <a,b,...>`, `--n-exp <k>`, `--seed <s>`,
`--do-ga` (turns GA on — must match the baseline for a valid comparison),
`--explore-trials`, `--exploit-trials`, `--exploit-phases`, `--out <path>`.

> Always time the **release** binary. A debug build can read slower than CPython
> and would invert the result.

### Python baseline (pyalcs)

The baseline runs the **unmodified** pyalcs agent in a pinned environment (Python
3.10, `gym==0.23.0`, `numpy==1.23.5`) managed by [`uv`]. It expects the reference
repos cloned as siblings — see `baseline/README.md` for the layout and pinning
rationale.

One-time setup:

```bash
cd baseline && uv sync
```

Run the full suite and write the comparison CSV (from the repo root):

```bash
uv run --project baseline python baseline/run_pyalcs_maze.py \
  Maze4-v0 Maze5-v0 Maze7-v0 Woods1-v0 Woods100-v0 \
  --repeats 10 --csv reports/bench_pyalcs.csv
```

`run_pyalcs_maze.py` takes one or more maze ids (positional), plus `--seed`
(default 42), `--repeats` (default 1; the pinned benchmark uses 10), `--do-ga`,
and `--csv <path>`. Without `--csv` it prints verbose per-trial output instead.

### Reproduce the comparison

`compare_bench.py` joins the two CSVs into the P9 report (per-maze learning quality
+ speedup, with the total-time headline) and flags any maze whose exploit steps
differ by more than 2× (run from the repo root, where the default CSV paths
resolve):

```bash
uv run --project baseline python baseline/compare_bench.py
# reads reports/bench_rust.csv + reports/bench_pyalcs.csv, writes reports/P9_comparison.md
```

### MPX figures

Long MPX runs happen on the WCSS cluster (`slurm/mpx_reach.sh`); check them with
`./slurm/mpx_status.sh`. Their logs reduce to committed CSVs, and the figures are
generated from those CSVs — never from the logs — so they regenerate from the
repository alone:

```bash
python3 tools/parse_mpx_logs.py reports/mpx_m3_e1_traj70_pyalcs.log reports/slurm_mpx70_*.out
uv run --project tools python tools/plot_mpx.py
```

`tools/README.md` documents the log-format traps the parser exists to absorb (the
`seed = base_seed + repeat` rule is the one that silently corrupts results) and
the chart conventions.

### Tests

```bash
cargo test --release
```

This runs the fixture-backed unit tests (P3–P6: matching, population, learning
core, agent loop), the maze-parity tests against dumped `gym_maze` probes, and the
**differential tests** (`p8_differential.rs`, `episode_differential.rs`) that
replay the pyalcs-instrumented fixtures.

---

## Methodology (why the speedup is trustworthy)

Constraints below are quoted from the **ARCHITECTURE.md** "P9 benchmark
methodology" section — they are the project's own rules, not claims invented here:

- **Optimized Rust only** — the timed binary is `target/release/acs2-bench`; a
  debug build is never timed.
- **Sequential, uncontended** — the pyalcs and Rust timed runs never overlap; one
  runs, then the other, on the **same machine**.
- **Symmetric timed region** — only explore + exploit are timed; population/reliable
  metrics are computed *after* the timed region on both sides.
- **GA OFF on both sides** — `do_ga` is one flag (`--do-ga`), default OFF for this
  comparison, set to the same value on the Rust bench and the pyalcs baseline.
- **Identical protocol** — 500 explore (ε=0.8) + 3×200 exploit, final-window mean
  steps, `n_exp=10`, same mazes and per-maze step caps on both sides.
- Two readings are reported: per-maze `t_py/t_rust` and the total-time
  `Σt_py/Σt_rust` (the labelled headline).

The speedup reflects language/runtime only; the bit-packing optimization is
deferred behind the same interface and is **not** part of these numbers.

---

## Scope and fidelity

The port reproduces **pyalcs behaviour** (the source of truth), not the textbook
Butz & Stolzmann ACS2. Two consequences, summarized from the ARCHITECTURE.md
"fidelity deviations" section and `PROJECT_CONTEXT.md` §4:

- **Four pyalcs bugs are NOT reproduced.** Three orchestration bugs
  (`_is_preferred_to_delete`, `ClassifiersList.copy()`, `_run_trial_exploit`) plus a
  fourth in a learning primitive — the `apply_alp` mid-iteration skip that pyalcs's
  list-deletion-while-iterating causes — were surfaced by the differential testing.
  Rust re-derives the loop cleanly and uses deferred deletion, so it is the correct
  side. Details: `PROJECT_CONTEXT.md` §4, the ARCHITECTURE.md fidelity ledger, and
  `reports/P8_differential.md`. (The task brief referenced an `AUDIT_REPORT.md`;
  that file does not exist in the repo — the four bugs are documented in those three
  files instead.)
- **Some pyalcs config choices are reproduced deliberately** for benchmark parity:
  ALP generalization is disabled by `u_max = 100000` (under the shipped config ALP
  only specializes), exploitation ignores epsilon (always BestAction), and
  `biased_exploration` is dead config. These depart from canonical ACS2 on purpose,
  so P9 compares the *same* algorithm on both sides.

[`uv`]: https://docs.astral.sh/uv/
