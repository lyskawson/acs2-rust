# Per-goal reward ablation

This study adds online `acs2_per_goal_reward` (ACS2 r(g)) to the five frozen ACS2,
episodic ER and shortened HER arms. `registration.json` fixes the 138 seed42 cost-pilot
cells, analysis, 20 seeds and 15 CPU-hour authorization. `plan.json` is written only
after the cost pilot and requires separate approval before any full-grid run.

`geometries.py ALCS_CHECKOUT --write` rebuilds the new source fixture from committed
ALCS blobs at ed0e762; omit `--write` to check hashes and matrices. Benchmark deviations
are explicit. Four research matrices have new ids; existing benchmark matrices and
lists remain frozen. Every visible 9 is a walkable landmark in multi-goal research;
single-goal topology construction still requires exactly one.

`cargo run --release -p acs2-measure --example ablation_geometry` exports the runtime
perceptions, matrices, finite/disconnected distances and successors to `geometry.json`.
`register_protocol.py` rebuilds `pair-protocol.json` using only that geometry and exact
integer random-walk counts; `--check` compares the committed registration. The protocol
id is `alcs-maze-pairs-v1`. Original six tasks retain `maze-pairs-v1` bytes and caps.
`maze-names.json` owns thesis labels, including the two pyalcs disambiguations.

`execution.py prepare ROOT --repo REPO` freezes clean source/binary identity and an
operations snapshot in a new study-owned cluster root. `execution.py dispatch ROOT`
re-derives seed42 assignments, checks all attempts and reservations, the shared grant
and scheduler admission before each array. At most two 20-job streams may run. Node
launchers check source/binary/hardware/limits and use the unchanged grid collector
through a study-local six-arm validator. A real failure writes shared STOP.

`references.py` indexes 600 original archived reference runs with SHA-256 and can
compare the first seed42 checkpoint of all 30 reference cells locally. The analysis
uses canonical names and preserves binary64 values. The three earlier-study contrasts
must reproduce 36 earlier intervals exactly. They are checks, not new results.

`prepare_plan.py ROOT --summary-only` retains the first-wave audit path. The plan
records paired test-block checkpoint means, final points, both within-seed interactions, floor/easy/ceiling
labels, separate group verdicts and descriptive sustained-0.95 checkpoints. Other
comparisons and mechanism analyses are post hoc. No registered grid analysis or
figures are produced by the cost-pilot stage.

## Approved 30k cost pilot

The 2026-11-10 amendment authorizes `cost-pilot.json`: fresh 30k runs in all 138 cells,
including 18 bitwise prefix checks against the stopped first wave. The latter's 9810
CPU-seconds remain charged to the same 15 CPU-hour budget. `execution.py` and the old
registration remain first-wave evidence; use `cost_pilot.py` for the active pilot.
Its independent operations snapshot may advance while the measurement clone and binary
remain pinned at 41b025a. `cost_pilot.py prepare ROOT --repo REPO --operations SNAPSHOT
--prior STOPPED_ROOT` verifies both identities and the immutable stopped ledger.
`dispatch` accounts for both roots, checks the grant and test-only before every array,
and permits at most two streams. New worst-case reservations plus prior spend are
14.725 CPU-hours. `prefix_identity.py` compares every learning and per-pair field in
binary64 bits before a stopped prefix may be reused as cost-growth evidence.

Only cost, memory, population and per-goal entry summaries are allowed for this pilot.
The full protocols, budgets and all 28 mazes remain unchanged. No full new run completed
in the first wave, so the later full grid still needs all 2760 runs, including seed42.

## Completed cost evidence and pending plan

`cost_evidence.py --index INDEX --prior STOPPED_ROOT --out COST_PROFILES` exports
cost-only profiles from the 600 original full runs and 18 stopped attempts. It verifies
artifact hashes and allocation accounting before projecting any cost. `cost_plan.py
--prior STOPPED_ROOT --pilot PILOT_ROOT --reference-profiles COST_PROFILES
--reference-index INDEX --out OUTPUT` audits all 138 fresh completions and both sealed
archives, authenticates all 18 stopped prefixes, and writes checkpoint CPU estimates,
exact process/allocation CPU, peak memory, population, per-goal entries and forecasts.
The generated `registered-plan.json` is committed here as `plan.json`.

Every cell has a standardized 2M forecast and its unchanged registered-budget forecast.
The scenario range transfers joint seed variation and late cost growth from the full
archives and observed growth from longer attempts. It is not a confidence interval or
a guaranteed bound. ACS2 r(g) uses ACS2 archive growth and its own measured pilot and
longer attempts. No unmeasured discount is applied to compact evaluation seeds. The
proposed allocation limits add an operational margin and require separate approval.
The pilot does not supply any completed full run; all 20 seeds remain in the new grid.
No pilot success summary, registered outcome analysis or figures are produced here.

## Approved full grid

`grid_execution.py prepare ROOT --repo REPO --operations SNAPSHOT` creates an execution
wrapper beside an exact copy of the approved `plan.json` (SHA-256 `7d9606...ac66`).
Measurement remains pinned at 41b025a and its original musl binary. The wrapper records
the 2026-11-10 approval, the test-only review fix a8e8c15, 3583 cumulative CPU-hours,
2048 MiB per allocation and immutable evidence for all 156 pilot attempts.

Use `dispatch ROOT` for an immediate submission or `enqueue ROOT` for pairs of
arrays with SLURM `afterok` dependencies. Each pair waits for both predecessors.
Invalid dependencies close queued arrays after a failure. Resume enqueueing when
the association has submission capacity. The dispatcher selects the 2520 ordinary
runs first, then the twelve seed42 MazeE2/MazeE3 scouts.
The remaining 228 runs require every scout to complete and its allocation CPU to stay
within that cell's approved high forecast. `scout-cost-gate.json` retains each cost and
threshold. Every array groups equal registered time limits, uses at most 20 concurrent
tasks and shares the two-stream cap. Before every submission, accounting deduplicates
all pilot and grid allocations by JobIDRaw, checks spend plus reservations against the
approved cumulative limit, and checks the shared grant and scheduler test-only result.

`job_grid.sh` and `launch_grid.py` check the source, binary, allocation, hardware,
preset and row identity. The launcher runs the collector and checks completion metadata.
The collector owns STOP handling. Timeout retries require approval. Run the controller
commands in the active session.

`grid_results.py ROOT --out EVIDENCE` audits array admissions, allocation CPU and
resources against the registered plan. It writes attempt and checkpoint cost tables.
For a complete grid, supply `--archive-index`, `--original-root`, `--mirror-root`
and `--reference-costs` to build `full-index.json` and the resource table for the
600 original runs. The index uses verified measurements from the grid and the
original archives.

`artifact_analysis.py --index FULL_INDEX --earlier EARLIER_COMPARISONS --out OUTPUT`
requires all 3360 full runs: the new 2760 and the 600 original archives. It verifies
source hashes, normalizes archived agent names, projects each checkpoint without
retaining per-pair arrays, and checks the 36 earlier intervals in binary64 bits before
writing any results. Outputs include registered contrasts and labels, per-group
verdicts, per-seed measures and paired differences, checkpoint means, descriptive
sustained-0.95 points and secondary observations. The original analysis formulas and
approved plan remain unchanged.

`success_curves.py --data OUTPUT/curves.csv --out FIGURES` writes every maze/arm curve
as an individual PNG/PDF and paginated overviews by observation aliasing. Curves show
20-seed means at registered checkpoints, without smoothing or an interpolated
threshold. All mazes remain visible. Registered paired primary/final contrast
intervals are in the tables; the plots do not add pointwise inference.
