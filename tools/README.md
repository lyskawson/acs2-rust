# Figure generation (MPX experiments)

Turns `mpx_reach` logs into the figures for the thesis. Two stages, deliberately
separated:

```
logs (.log/.out)  --parse_mpx_logs.py-->  reports/*.csv  --plot_mpx.py-->  reports/figures/*.pdf
        stdlib only                        committed              matplotlib
```

## Why this is a separate project from `baseline/`

`baseline/` pins an **unmodified pyalcs** (Python 3.10, `gym==0.23.0`,
`numpy==1.23.5`) and its lockfile is part of the differential-validation evidence
for the port. Adding plotting dependencies there would perturb an environment
whose whole job is to stay fixed. So figures live here, with their own lock.

## Why the CSV sits in the middle

Thesis provenance requires every figure to regenerate from the repository alone.
Raw logs are large, arrive in several vintages, and mostly live on the cluster;
the CSVs are small, reviewable, and committed. **Nothing downstream reads a log.**

## Setup

```bash
uv sync --project tools
```

## Parse

Run from the repo root. Accepts any mix of laptop `.log`/`.txt` and cluster
`.out` files:

```bash
./tools/sync_runs.sh --local    # rebuild the complete local archive
```

Needs no venv (standard library only). Writes `reports/mpx_trajectory.csv` (one
row per evaluation point, with accuracy and the four coverage classes merged in),
`reports/mpx_diagnostics.csv` and `reports/mpx_verdicts.csv` (one row per repeat);
override all three outputs with `--trajectory-csv` / `--diagnostic-csv` /
`--verdict-csv` when parsing a subset. Never overwrite the archive from a subset.
The complete input set is `reports/slurm_*.out`, `reports/*.cancelled`,
`reports/mpx_m2b_reach*.log`, and `reports/mpx_m3_e1_traj70_*.log`.
`qdetail:` candidate coverage and best quality are merged into the trajectory row.

To rebuild the whole archive after pulling the cluster logs:

```bash
./tools/sync_runs.sh
```

### What makes a row reproducible

Every row carries the configuration that determines the result: `size`, `seed`,
`u_max`, `variant`, `encoding`, `epsilon`, `agent`, `do_ga`, `er_buffer_size`,
`er_min_samples`, `er_samples_number`, `eval_interval`, plus
`commit` and `tag` for runs submitted after the wrapper started recording them.
Missing header values remain empty. `source`, one-based header `block`, `size`, and
`repeat` identify each run, even when one file repeats an identical configuration.
Equal seeds and complete learning configurations reproduce learning at equal trial
counts on the tested 64-bit platforms, independent of diagnostic overhead.

`encoding_source` is the honest part. The header only carries `encoding` from the
commit that added it, so for older logs the value is reconstructed and this column
says how — `header` (stated by the run), `filename` (the tag convention), 
`submission-record` (known from how the job was submitted, listed in
`KNOWN_ENCODINGS`) or `wrapper-default` (inferred from the wrapper defaulting to
`flip`). Anything but `header` is an inference; re-check it before it goes into a
paper. The column exists because it immediately caught one run — the k=264 probe —
that every other rule would have mislabelled.

`peak_rss_gb = 0` on a cluster log means **unmeasured**, not zero: `ru_maxrss` was
read as bytes on Linux until `f93b71e`.

Three log-format traps it handles, all of which have bitten this project:

- **`seed = base_seed + repeat`** (`mpx_reach.rs`). An `n_exp=3` log at seed 42
  holds seeds 42, 43 *and* 44 — collapsing them into one series would hide the
  seed variance that turns out to be the dominant effect at k=70.
- **Trajectory lines carry no repeat index.** They are attributed to the repeat
  closed by the next verdict line; an unclosed tail (run in flight, or killed)
  becomes the next, unfinished repeat.
- **Older logs predate `u_max` and `alp_gen_variant`**, and one file may hold
  several runs behind banner lines. A header line resets the parse context.

## Sync — run this after every batch

Nothing copies a cluster run into the repo on its own. `slurm/mpx_reach.sh` writes
only to `~/mpx_runs/` on the cluster, deliberately outside any checkout, so until
this runs the cluster home is the **single copy** of a result that cost days.

```bash
./tools/sync_runs.sh            # pull, rebuild the CSVs and tables, report
./tools/sync_runs.sh --commit   # the same, then commit
```

Use `--local` to skip the cluster pull. `--commit` detects untracked logs as well
as tracked modifications, including header-only logs without CSV rows.

It prints what is new or changed, then lists every solved run in the archive — if a
run you remember solving is missing from that list, its log never left the cluster.

This gap has already bitten once: 32 logs existed only on the cluster, and five
were committed as zero-byte stubs, so the repo *looked* like it had them.

## Summarise

Turns one size into a page a person can scan, in-flight runs included:

```bash
python3 tools/summarize_mpx.py --size 135     # writes reports/MPX135_runs.md
```

Grouped by experimental arm (encoding, epsilon, agent, GA, replay parameters), one row per repeat, with
the four coverage classes alongside knowledge -- at k=135 a ceiling of 0.75 or 0.50
is read in those columns, not in the knowledge column.

## Plot

```bash
uv run --project tools python tools/plot_mpx.py --size 70 --figures anatomy \
    --agent acs2 --source slurm_mpx70_s42_addr.out
```

**Always narrow to one arm and one source per seed.** A seed now has runs under both encodings,
two epsilons, several `u_max` values, and different replay settings; splicing them into one curve produces a
trajectory that never happened. The tool refuses to plot a mixed selection and
names what is mixed:

```bash
uv run --project tools python tools/plot_mpx.py --size 135 --figures reach \
    --encoding flip --epsilon 1 --u-max 11 --agent acs2 \
    --source slurm_mpx135_s43_eps1_u11.out \
    --source slurm_mpx135_s42_eps1_u11.out --suffix _canonical_eps1
```

Writes PDF (for LaTeX `\includegraphics`) and PNG (for previewing and README
embedding) into `reports/figures/`. Useful flags: `--size`, `--variant`,
`--anatomy-seed`, `--figures reach,anatomy`, `--formats pdf,png,pgf`.

| Figure | File stem | What it argues |
|---|---|---|
| reach | `mpx<k>_reach_<variant>` | knowledge vs trials, one line per seed — the reach claim *and* the seed variance behind it |
| anatomy | `mpx<k>_anatomy_s<seed>_<variant>` | one seed as three stacked panels — why the apparent plateau is not a stall |

## Chart conventions (do not quietly break these)

- **No dual-axis plots.** `anatomy` is stacked panels sharing one x-axis rather
  than several y-scales on one plot: the measures have unrelated units, and
  twin axes invite the eye to read crossings that mean nothing.
- **Colour follows the seed, never its rank**, and the mapping is built from every
  seed in the CSV — so filtering to a subset never repaints the survivors.
  The palette is a validated categorical order (worst adjacent colour-vision
  deficiency ΔE 35.9 on white); aqua sits marginally under 3:1 contrast, which is
  why every series is also direct-labelled.
- **A run cut off by its time cap gets a hollow end marker** and a footnote saying
  so. Truncation is a budget artifact; a reader must never mistake it for a
  failure to converge.
- Figure titles stay descriptive. The argument belongs in the LaTeX caption.

## Adding the remaining figures

`scaling` (trials-to-success vs k, log y) and `ablation` (M1 specialize-only vs
M2a GA-on vs M2b canonical `u_max`) both read `mpx_verdicts.csv`, which already
carries what they need. They are not written yet because the k=70 confirmation
seeds and the k=135 run are still in flight — see `docs/AGENT_HANDOFF.md` §4.

## Review-fix protocol and checks

`plot_mpx.py` and `summarize_mpx.py` share filters for agent, GA, replay capacity,
warmup, replay count, encoding, epsilon, and `u_max`. `--source` is repeatable;
`--block` selects a header occurrence. Plotting rejects independent runs sharing a
seed, even if their learning parameters agree, and success markers must belong to
that exact run. Signal plots apply the same filters plus `--variant` and
`--signal-seed`. Legacy blank agent fields predate the shared `--agent` header
(`a35a095`) and are treated as ACS2 by selection; the CSV preserves the raw field.

`knowledge_trials` and `knowledge_status` describe verdict measurement timing.
Historical values remain verbatim: SUCCESS is `at-verdict`; a known earlier
trajectory evaluation makes the value `stale`; logs without enough timing evidence
are `legacy-unverified`. New terminal `knowledge=unmeasured` becomes an empty CSV
cell and status `unmeasured`. The readable tables show the knowledge trial separately
and only attach accuracy/coverage measured at the verdict trial. No final population
is reconstructed from a log.

New headers carry `strict_resource_limits` and `rss_scope`. Empty historical values
mean unrecorded; `--isolate-repeats` emits a separate header with the actual seed and
repeat 0 for each fresh child process. `--strict-resource-limits` checks resources
after evaluation before granting SUCCESS. Both options default off. The corrected
Butz algorithm is `--alp-gen-variant butz-checked`; legacy `butz` is unchanged.

`qdetail:`-only points and unfinished blocks at a subsequent header are retained.
Header-only logs have no measurement rows, but are still archived and committed.
The historical diagnostic `correct` remains a complete-address structural proxy,
not a complete test for all correct minimal MPX rules.

Run the existing 73 workspace tests, the expanded archive/plot regressions, and
isolated runner checks (synthetic populations, no learning experiments):

```bash
cargo test --workspace --release
uv run --project tools python -B -m unittest discover -s tools -p 'test_*.py'
python3 tools/check_reach_protocol.py
```

The last command builds a temporary Rust test harness against the production runner.
It checks verdict timing, post-evaluation resource limits, and fresh-process RSS.
The normal workspace gate keeps its original 73 tests, with additional assertions
for the Butz counter and invalid replay/GA boundaries in existing tests.

Rebuild the seven committed figure pairs with explicit source choices:

```bash
uv run --project tools python tools/rebuild_figures.py
```

`reports/figures/manifest.json` records those choices and the input CSV hashes.
The epsilon-1 figure uses seed 42's completed time-limited first run; its separate
longer rerun is not spliced into that curve.
