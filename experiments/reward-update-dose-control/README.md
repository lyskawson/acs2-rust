# Random reward-update dose control

`acs2her_shortened_random_rules` (R) retains shortened HER (S) at p=2/3,
three full-model draws per interaction and its policy/sampler streams. After ALP,
each post-ALP action-set classifier independently skips r and ir updates with
the maze-specific probability in `probabilities.json`, using an injected,
seed-derived ChaCha8 stream `0x726577617264736b`. The coin ignores classifier
attributes. Original draws, including original-goal fallbacks, update every rule
and consume no skip coin. ALP, GA, q, action selection, learning order and the
whole-next-match-set bootstrap retain their existing paths. The thesis preset
keeps GA disabled; step tests also exercise GA enabled.

The integer probability sums come from final G (`acs2her_shortened_goal_rules`)
rows over seeds42--61 in the original `goal-rule-copies` archive. `freeze_dose.py`
checks every source checksum, seed, final point and the card's rounded values;
a mismatch stops the study. The CLI selects the registered probability by maze;
it exposes no probability override. Validators require its exact binary64 value.
New rows record cumulative updates and skips by actual origin and goal specificity,
post-ALP macroclassifier opportunities and draws containing a goal-specific rule.
The skip RNG, parameters and counters participate in read-only evaluation snapshots.
Existing arms retain their row schemas and deterministic fields, including G counters.

`freeze_references.py` copies G from its original verified runs and S/E from the
original paths recorded by that study, checking original file hashes before
copying. References are never re-run. All six geometries, interaction budgets,
checkpoints, pair registrations and three blocks remain those of the source study.
Per-pair records cover seeds42--46. `results.py`, `dose.py` and `analyze.py`
implement the card's G-R and R-S paired contrasts, dose checks, four disjoint
outcomes and joint mechanism rule. Whole-episode success uses only the test block;
mechanisms pool all blocks with their recorded weights and equal block mass.
Goal-specific winners split tied classifier credit equally; fallbacks receive zero.
Woods14 is reported and excluded from the primary outcome.

The authorized pilot is six seed42 runs with per-pair recording and twenty-minute
single-CPU allocation limits, at most two cumulative CPU-hours including every
attempt and reservation. The shared `experiments/grid/` path mechanically derives
the manifest before each submission, verifies a pinned clean musl build and frozen
operations, checks the shared grant and queue, and runs scheduler test-only.
Node guards and shared STOP retain real failures. No seed beyond42 is authorized.
If any final relabeled skip share differs from its registered probability by more
than 0.01, pilot verification stops before plan preparation.

```bash
python3 -B -m unittest discover -s experiments/reward-update-dose-control -p 'test_*.py'
python3 -B experiments/reward-update-dose-control/freeze_references.py --out <new-reference-directory> --archive <original-G-study>
python3 -B experiments/reward-update-dose-control/prepare_pilot.py --out <new-local-study> --destination <absolute-cluster-study> --references <reference-directory>
python3 -B <frozen-operations>/experiments/grid/submit.py --root <cluster-study>/pilot --manifest <cluster-study>/pilot-manifest.json --repo <pinned-TU-clone>
python3 -B experiments/reward-update-dose-control/archive.py <local-study>
python3 -B experiments/reward-update-dose-control/prepare_plan.py <local-study>
```

Stage1 ends with the cost/dose pilot, one committed plan with a measured forecast
and the external Polish executor report. Stage2 requires approval of those plan
bytes with a cumulative CPU-hour limit. It runs seeds43--61, produces the registered
analysis and F/F_mod curves for R against G/S, and documents results in
`docs/ARCHITECTURE.md`. No ranking or mechanism conclusion follows from the pilot.
Raw rows, accounting, verification tables and reports live outside the checkout
under `~/Desktop/tu-runs/reward-update-dose-control/` and
`~/tu_runs/reward-update-dose-control/` on WCSS, with two verified copies.

## Cost, dose and continuation forecast

All six first-attempt seed42 allocations completed: 49 rows and 41,766 per-pair
records pass source, protocol, cumulative-counter, aggregate and capped-value
checks. Allocated cost is 827 CPU-seconds (0.229722 CPU-hours), with zero failures
and reservations, within the two-hour pilot cap. The largest absolute deviation
of the final relabeled skip share from registered pi is 0.000104378 (Maze7),
below 0.01 on every maze. This pilot establishes cost and dose, without ranking arms.

Measurement source is `cebf171`; the musl SHA-256 is
`734021b127f9b801ea6ef1209770c17d8089d59b5c7ec9dc7b73e5e251a33be6`.
Frozen pilot operations are `277cd7b`. The single [plan](plan.json), SHA-256
`10aafe270bd05de0822efe5dd467b8a6b35228e33ad4755c1b056f36b5fe9d53`,
fixes both registered comparisons and the outcome/mechanism rules before further
R seeds. Dose checks use equal-weight means of per-seed final cumulative shares;
pi itself retains the required pooled integer-sum definition.

The expected cumulative cost is 4.594444 CPU-hours, including the pilot.
Doubling the remaining measured cost gives 8.959167 CPU-hours; full hard
reservations plus the pilot give 13.829722. The proposed cumulative limit is
therefore 14 CPU-hours. One cost seed does not bound seed variance. These values
are a forecast and reservation bound, without approval to run seeds43--61.
