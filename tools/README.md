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
cd tools && uv sync
```

## Parse

Run from the repo root. Accepts any mix of laptop `.log`/`.txt` and cluster
`.out` files:

```bash
python3 tools/parse_mpx_logs.py reports/mpx_m3_e1_traj70_pyalcs.log reports/slurm_mpx70_*.out
```

Needs no venv (standard library only). Writes `reports/mpx_trajectory.csv` (one
row per evaluation point, with accuracy and the four coverage classes merged in),
`reports/mpx_diagnostics.csv` and `reports/mpx_verdicts.csv` (one row per repeat);
override with `--trajectory-csv` / `--diagnostic-csv` / `--verdict-csv`.

To rebuild the whole archive after pulling the cluster logs:

```bash
rsync -az -e "ssh -i ~/.ssh/id_rsa_wcss" alelys2099@ui.wcss.pl:'~/mpx_runs/' /tmp/mpx_logs/
cp /tmp/mpx_logs/*.out /tmp/mpx_logs/*.cancelled reports/
python3 tools/parse_mpx_logs.py reports/slurm_*.out reports/*.cancelled \
    reports/mpx_m2b_reach*.log reports/mpx_m3_e1_traj70_*.log
```

### What makes a row reproducible

Every row carries the configuration that determines the result: `size`, `seed`,
`u_max`, `variant`, `encoding`, `epsilon`, `agent`, `eval_interval`, plus
`commit` and `tag` for runs submitted after the wrapper started recording them.
Two runs that agree on all of these should agree trial for trial.

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

## Summarise

Turns one size into a page a person can scan, in-flight runs included:

```bash
python3 tools/summarize_mpx.py --size 135     # writes reports/MPX135_runs.md
```

Grouped by experimental arm (encoding, epsilon, agent), one row per repeat, with
the four coverage classes alongside knowledge -- at k=135 a ceiling of 0.75 or 0.50
is read in those columns, not in the knowledge column.

## Plot

```bash
uv run --project tools python tools/plot_mpx.py
```

**Always narrow to one arm at k>=135.** A seed now has runs under both encodings,
two epsilons and several `u_max` values; splicing them into one curve produces a
trajectory that never happened. The tool refuses to plot a mixed selection and
names what is mixed:

```bash
uv run --project tools python tools/plot_mpx.py --size 135 --figures reach \
    --encoding flip --epsilon 1 --u-max 11 --suffix _canonical_eps1
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
