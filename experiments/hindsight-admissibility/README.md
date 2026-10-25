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
hashes. `prepare_plan.py` derived the regression fixture and the fixed
240-run registration from those verified rows and costs. Never change an archived
registration or fixture to accommodate new measurements.

The registered analysis uses archived HER and episodic replay from `her-variants/`,
and ACS2/ACS2ER from `maze-baselines/`, through `grid/agent_names.json`.
`plan.json` fixes both test-success views, five paired contrasts, marginal t(19)
intervals, exact rational threshold crossings, route/wall/value/model diagnostics
and purpose-specific objective costs. The forecast doubles observed cost and
charges actual twenty-job hard reservations. A different-update shortened
comparison is post hoc only.

## Approved execution and archive verification

`approval.json` authorizes only plan SHA-256
`f7dcf93ce08fa6eee9093736c5210e3c532cecfee5541e5f5475258aff91d2da`
and thirty cumulative allocated CPU-hours, including 1723 pilot seconds.
Measurement remains clean source `3828e430b13138b7458d65a840c1c833382f6c4a`,
static musl SHA-256 `a7422da3904e9b0d1a5df2152c3508bda66f27f86651ee3af929e230e9391893`.
Verify that existing file; do not rebuild the authorized cluster clone.

`prepare_study.py` snapshots clean operations, the approved plan and confirmation,
all twelve accounting roots, and mechanically partitioned pilot identity files.
`execute_batch.py` accepts indices 0--11, one registered maze/arm and twenty
seeds per batch. It requires no outstanding reservation before the next batch,
re-derives each manifest, verifies the binary and uses the shared submission path.
The node checks every seed-42 point and all three blocks against the pilot in
binary64 bits. A real failure creates STOP and blocks further work.

`verify.py` checks exact first-attempt scopes, receipts, operation hashes,
allocations, complete rows, source/hardware/preset identities and independent
recalculation of the node's pilot identity records. Admission records must cover
every accounting root exactly once. The final accounting refresh must include
all twelve arrays and no outstanding reservations. Pull the complete study and
independently checksum both copies before analysis.

The new negative tests construct a trivial copy with consistent pooled counters
and exercise the submission cap independently of manifest authorization. Removing
either refusal makes its test fail. Existing registered plans and fixtures stay
byte-identical; the source/Cargo diff from the measurement commit remains empty.

## Reproduce the registered analysis

Set `STUDY` to the verified local study root, `HER` to the approved hindsight
archive and `BASELINE` to the pair-baseline grid. Use a separate output directory
`ANALYSIS` outside every checkout and outside the raw study, preserving the raw
checksum inventory:

```bash
python3 -B experiments/hindsight-admissibility/analyze.py \
    --root "$STUDY" --her-root "$HER" --baseline-root "$BASELINE" --out "$ANALYSIS"
python3 -B experiments/hindsight-admissibility/validate_tables.py \
    --analysis "$ANALYSIS" --plan "$STUDY/approved-plan.json"
uv run --project tools python -B experiments/hindsight-admissibility/plots.py \
    --results "$ANALYSIS/results.json" --out "$ANALYSIS/figures"
```

The analysis first verifies all 240 new runs and all 98 pilot identity rows, then
checks the 480 reused reference trajectories. It exports 5880 exhaustive test
success counts, both exact seed-score views, sixty paired contrasts, all exact
thresholds, diagnostic summaries and per-run costs. Each contrast contains both
arms' diagnostics in its own registered view. Empty conditional value groups and
unmeasured historical wall counts remain null, with contributing seed counts.
Selection costs include admissibility objective queries. Historical allocation
costs are reported separately and never charged again.

`validate_tables.py` reconstructs success means, paired variances and intervals
from the integer CSV counts with independent rational arithmetic, checks all
seventy-two crossings, and reconciles new allocation CPU with the pilot. Its
tests reject unpaired intervals, strict rather than inclusive crossings, and
historical reference charges. Figures display marginal paired intervals, test
curves with SE, and registered value/population diagnostics; connecting curves
does not interpolate threshold crossings.

Run tests from the checkout root:

```bash
python3 -B -m unittest discover -s experiments/hindsight-admissibility -p 'test_*.py'
```
