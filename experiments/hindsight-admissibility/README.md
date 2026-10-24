# Full hindsight replay under goal admissibility rules

`acs2her_nontrivial` excludes copied goals reached at the transition's current
state. `acs2her_shortened` excludes goals reached anywhere in the episode prefix,
while retaining first arrivals. Both fix full updates, future goals and p=2/3;
the shared trajectory sampler handles original-goal fallbacks.

The seed-42 pilot covers both arms on all six immutable pair configurations at
their original 2M/4M budgets and all registered points. Twelve ten-minute
allocations reserve exactly two CPU-hours. No grid is authorized by the pilot.
`prepare_pilot.py` snapshots clean experiment operations, a single-root accounting
scope and the manifest derived from `protocol.py` and frozen `measurement.json`.
Use the snapshot's `experiments/grid/submit.py`; it checks the grant, test-only
schedule, source and binary identity, row schema, STOP and hard allocation bound.
The grid's scope-membership guard applies before accounting or scheduler calls.

Raw artifacts belong under `~/tu_runs/5b/` and `~/Desktop/tu-runs/5b/`, outside
every checkout. Pull accounting with allocation-level CPUTimeRAW, then checksum
both copies independently. `archive.py` verifies the complete twelve-cell pilot,
every attempt, operations, manifests, rows, hardware, matching volume and raw
hashes. `prepare_plan.py` derives the regression fixture and the one pending
240-run registration from those verified rows and costs. Never change an archived
registration or fixture to accommodate new measurements.

The planned analysis uses archived HER and episodic replay from `her-variants/`,
and ACS2/ACS2ER from `maze-baselines/`, through `grid/agent_names.json`.
`plan.json` fixes both test-success views, five paired contrasts, marginal t(19)
intervals, exact rational threshold crossings, route/wall/value/model diagnostics
and purpose-specific objective costs. The forecast doubles observed cost and
charges actual twenty-job hard reservations. Approval of the registered bytes
and a cumulative CPU-hour ceiling is required before implementing or launching
stage-2 execution. A different-update shortened comparison is post hoc only.

Run tests from the checkout root:

```bash
python3 -B -m unittest discover -s experiments/hindsight-admissibility -p 'test_*.py'
```
