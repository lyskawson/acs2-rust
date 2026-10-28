# Requested hindsight proportion and dose control

The study parameterizes only `acs2her_shortened` and `acs2her`. Each copy receives
full learning; future goals, preset admissibility, episode schedule, three draws,
storage, exploration and evaluation remain unchanged. Requested p is a probability
of asking for a copy. Realized copies are counted separately from original draws
and original-goal fallbacks. At p=2/3 both presets reproduce the earlier agents.

`protocol.py` derives the six immutable pair configurations and five seed-42 cost
pilot settings. Four settings use all three evaluation blocks: shortened at 1/3,
1/2 and 4/5, and HER at 1/2. Shortened at 2/3 uses validation only and must reproduce
every shared learning, knowledge and validation field of the admissibility archive
at every registered point. Thirty ten-minute allocations reserve five CPU-hours.
The shared grid authorization and node identity guards stop on a real mismatch.

The completed pilot has 30 successful first attempts, 245 rows and 4784 allocated
CPU-seconds (1.328889 CPU-hours), with no reservations remaining. All 49 shortened
2/3 validation/learning/knowledge identity rows pass in binary64 bits. The new
`fixtures/hindsight_proportion_regression.json` pins the first two points of all
24 new full-block trajectories (48 rows); every earlier fixture is unchanged.

The pre-results [registration](plan.json), SHA-256
`894596d2f792a2327ff5fd061c4344a17bcb96b01b9a96dd4f58a61a1e3a2888`, includes four candidates. `approval.json` authorizes its exact bytes and 45 cumulative allocated CPU-hours, including the 4784-second pilot.
With 4/5, the expected total is 17.8275 CPU-hours, the doubled forecast 34.326111,
and the all-array hard bound 40.328889; the suggested cumulative approval is 45.
Without 4/5, the corresponding values are 16.258056, 31.187222 and 37.995556,
with suggested approval 40. Forecasts include the complete pilot and the worst
conditional test scope, one selected non-reference setting on every maze. A cut
requires regenerating the registration before approval; no further pilot is needed.
`prepare_plan.py --proportions` accepts any registered finite candidates in [0, 1],
including the mandatory 2/3 identity setting.

The measurement `agent` remains its thesis name. Manifests and annotated rows use
explicit proportion labels such as `acs2her_shortened_p1of3`,
`acs2her_shortened_p1of2`, `acs2her_shortened_p4of5`,
`acs2her_shortened_p2of3` and `acs2her_p1of2`. Binary rows remain unchanged beside
the annotated rows. Full HER at 1/2 keeps every-transition admissibility.

`prepare_pilot.py` snapshots clean operations and five-seed reference rows from the
verified admissibility archive. Measurement source and musl binary are frozen in
`measurement.json`. Pilots and subsequent roots share the same accounting scope
root inventory, with zero pilot offset: each allocation is counted once through
its own ledger. The future approval adds its own scope authorization without
altering the frozen pilot scope. Every submission checks source, binary, operation
hashes, grant, test-only scheduling, shared STOP and used plus outstanding hard
limits. Results belong under `~/tu_runs/hindsight-proportion/` and
`~/Desktop/tu-runs/hindsight-proportion/`, outside every checkout.

`local_check.py` runs six sequential short validation checks on Maze4 and Woods14.
It tests mechanics only and ranks nothing. `archive.py` independently checks
terminal allocation accounting, complete row grids, raw artifact hashes, operation
snapshots, preset/hardware identities and node identity records. `prepare_plan.py`
uses the verified pilot to create the new bit regression and one pre-results plan
with forecasts both with and without 4/5. Earlier plans, pilots and fixtures are
immutable.

The plan registers validation-only tuning for shortened HER, seeds 42--46, and a
separate p per maze. `tuning.py` requires all registered candidates and points;
selection maximizes the mean of within-seed checkpoint-mean success, with smaller
p winning within 1e-12 of the maximum. The tuning table includes five-seed SE and
descriptive across-maze means. At 2/3 all five seeds must reproduce the earlier
validation and learning/knowledge fields before selection.

`execution.py` derives tuning, independent HER@1/2 control and conditional test
manifests. These are full scientific manifests. `approved_execution.py` uses the shared
`grid_dispatch.py` path to submit one array per invocation, at most two active
streams and forty running jobs. Each invocation derives the manifest again,
checks completed attempts and identities, and checks the grant and hard budget
before test-only scheduling and submission. No process watches the laptop or
renews a lease. Pilot accounting files remain immutable; the common scope
writes refreshed allocation records to `shared-accounting.psv` and new roots.
Test derivation recomputes the saved selection from tuning and repeats
its identity checks. Where 2/3 wins, reuse the admissibility grid and run nothing.
New test settings reproduce the five tuning seeds on validation and seed 42 on
all pilot blocks. The control also reproduces its seed-42 pilot.

The registered analysis uses both test checkpoint-mean and final-point views,
paired twenty-seed marginal t(19) intervals, both exact baseline thresholds and
route/value/model/cost diagnostics. The two dose/rule contrasts must sum exactly
to the earlier shortened-minus-HER contrast. All success comparisons with ACS2 are
post hoc. Tuning rows have no test value diagnostics; those remain unmeasured until
a three-block run. Historical missing wall counts remain null. The dose match is
approximate and its actual realized shares must be reported.

The pilot ends with registration and the external Polish report. The saved
approval opens tuning, control and conditional testing under the cumulative
45-hour limit; the measurement source and binary stay at the pilot identities.

```bash
python3 -B -m unittest discover -s experiments/hindsight-proportion -p 'test_*.py'
python3 -B experiments/hindsight-proportion/local_check.py --out <new external directory>
python3 -B experiments/hindsight-proportion/prepare_pilot.py --out <new external study root> --destination <cluster study root> --admissibility <verified admissibility study>
python3 -B experiments/hindsight-proportion/archive.py <study root>
python3 -B experiments/hindsight-proportion/prepare_plan.py <study root>
```

## Approved execution

`prepare_execution.py` preserves the pilot and snapshots clean operations in
`execution-operations/` beside the raw study. Source and musl binary remain at
the pilot identity; do not rebuild the measurement clone. The approval and one
scope include all thirteen roots with zero pilot offset, so the existing pilot
ledgers charge every one of their 4784 CPU-seconds exactly once.

Run each command from the immutable operation snapshot on WCSS. An invocation
submits only the next registered array, or returns complete. Wait until its
stream is terminal before invoking it again; tuning and control may be the two
active streams. Save selection only after complete tuning and all five-seed
2/3 identities. Test derivation recomputes it before every submission.

```bash
python3 -B <operations>/experiments/hindsight-proportion/approved_execution.py --root <study> --repo <own pinned clone> --purpose tuning
python3 -B <operations>/experiments/hindsight-proportion/approved_execution.py --root <study> --repo <own pinned clone> --purpose control
python3 -B <operations>/experiments/hindsight-proportion/approved_execution.py --root <study> --repo <own pinned clone> --purpose test --configuration-id <selected non-reference maze>
```

A diagnosed infrastructure failure preserves its STOP record, every attempt and
all allocations. An explicit recovery record binds the archived STOP hash, failed
array IDs, approval and unchanged measurement binary. Pass it with `--recovery`;
the dispatcher repeats the first incomplete registered assignment in new attempt
directories. Completed prefixes from failed attempts must reproduce the retry in
binary64 bits. Neither a retry nor an operations update changes the full saved
scientific manifest or the node operation snapshot.

`save_selection.py` writes the exact selection only after the complete tuning
catalog and all 245 earlier 2/3 validation/learning/knowledge identities pass.
`verify.py` recomputes that selection, every manifest and reference identity,
the append-only pilot admission prefix, the failed-attempt inventory and the
single allocation scope. `analyze.py` exports results and diagnostics; allocated
CPU includes failed attempts, while historical reference cost is reported
separately. `validate_tables.py` independently reconstructs paired statistics,
selection, exact decomposition and threshold crossings from integer success
counts and exact per-seed scores. `plot_results.py` exports scientific PNG/PDF
figures beside the external analysis.

```bash
python3 -B <operations>/experiments/hindsight-proportion/save_selection.py --root <study>
python3 -B experiments/hindsight-proportion/verify.py --root <local study>
python3 -B experiments/hindsight-proportion/analyze.py --root <local study> --admissibility-root <admissibility study> --her-root <HER study> --baseline-root <pair baseline> --out <external analysis>
python3 -B experiments/hindsight-proportion/validate_tables.py --analysis <external analysis>
python3 -B experiments/hindsight-proportion/plot_results.py --analysis <external analysis> --out <external figures>
```
