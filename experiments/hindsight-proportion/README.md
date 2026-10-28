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
manifests. Test derivation recomputes the saved selection from tuning and repeats
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

Stage 1 ends with registration and the external Polish report. No tuning, control
grid or test grid may run until the user approves the exact plan bytes and a
cumulative allocated CPU-hour limit.

```bash
python3 -B -m unittest discover -s experiments/hindsight-proportion -p 'test_*.py'
python3 -B experiments/hindsight-proportion/local_check.py --out <new external directory>
python3 -B experiments/hindsight-proportion/prepare_pilot.py --out <new external study root> --destination <cluster study root> --admissibility <verified admissibility study>
python3 -B experiments/hindsight-proportion/archive.py <study root>
python3 -B experiments/hindsight-proportion/prepare_plan.py <study root>
```
