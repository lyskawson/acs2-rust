# Per-pair first-action diagnostics

This study replays five unchanged presets (`acs2`, `acs2er_episodic`, `acs2her`,
`acs2her_nontrivial`, `acs2her_shortened`, hindsight p=2/3) on the six registered
`maze-pairs-v1` configurations. It measures goal dependence of first actions,
capped optimal values and the classifiers supporting each first decision. It does
not alter learning, evaluation actions, random streams or agent parameters.

Stage 1 permits only seed42, thirty full trajectories and four allocated CPU-hours
including hard reservations. Seeds43--46 require approval of the single registered
`plan.json` and its cumulative CPU-hour limit. Stage 1 stops after that registration.

`--per-start` retains its existing pair fields and adds `first_decision` on the
population-based adapters: `random_fallback` and `winners`. Winners are every
matching change-anticipating classifier for the chosen action tied at the global
maximum `fitness * numerosity`. Each contains specified state/goal counts, q, r,
fitness, experience, numerosity and selection score. The first-action estimate
remains maximum change-anticipating fitness in the action set. Ties receive equal
fractional goal-specificity credit; the actual shuffled classifier identity is not
observable without changing the selector. No population dump is written.

`acs2-maze-truth <maze> <cap>` exports transitions and all start-goal optimal values
using public `acs2-envs` interfaces. Analysis checks the recorded shortest distance
and V* for every pair before computing anything. Values are zero when the shortest
path cannot fit the evaluation cap. When every capped action value is zero, all
eight actions are optimal. These pairs remain in the requested primary universe;
distance breakdowns expose this limitation.

The primary view pools all three blocks using their original weights, normalized
by total mass, so blocks have equal mass. C is computed separately at each start.
F and M condition on nonfallback decisions. D uses unordered goal pairs with disjoint
optimal sets, weighted by products of pair weights, and conditions on both decisions
being nonfallback. It exports both different-action and both-optimal rates; registered
D contrasts use the former. C retains all pairs as requested; eligible-set C is a
post-hoc check for selection bias when fallbacks depend on the goal. The registered winner share counts all decisions, with zero credit for fallbacks;
conditional shares among decisions with winners are also exported.

`analysis_registration()` specifies every registered contrast, t(4) interval and
post-hoc view. Undefined seed rates are retained as null, never silently dropped.
The consistency histogram uses pairwise plug-in Binomial(5,k/5) expectations and is
descriptive; fitting p to the same five outcomes cannot test independent seed
heterogeneity.

Shared grid collectors accept these records only with `per_start_diagnostics: true`
on this exact five-arm, full-pool, coordinate, three-block pair study. Compact rows
retain their earlier validators. `compare_per_pair` removes only `starts` before
using the shared bitwise identity checker and canonical archived names. Its existing
provenance and wall-clock exclusions remain explicit. No new learning accessor is
needed: measurement adapters already expose read-only populations.

```bash
python3 -B -m unittest discover -s experiments/per-pair-diagnostics -p 'test_*.py'
cargo test --release -p acs2-measure --test per_pair_diagnostics
cargo build --release --bin acs2-measure --bin acs2-maze-truth
python3 -B experiments/per-pair-diagnostics/local_check.py --out <new external directory>
python3 -B experiments/per-pair-diagnostics/prepare_pilot.py --out <new external root> --destination <absolute cluster root> --baseline <verified pair grid> --her <verified HER study> --admissibility <verified admissibility study>
python3 -B experiments/per-pair-diagnostics/archive.py <study root>
python3 -B experiments/per-pair-diagnostics/prepare_plan.py <verified local study root>
```

Freeze clean measurement source, a musl build and `measurement.json` before pilot
preparation. `prepare_pilot.py` verifies and freezes 150 historical trajectories,
exports six environment models and snapshots immutable operations. The one-shot
shared dispatcher submits a single thirty-job pilot array with eight-minute hard
limits (four CPU-hours total). Grant, queue and scheduler test-only evidence are
retained. `archive.py` independently validates raw checksums, launch identity,
accounting, every shared row bit, and every pair's distance and optimal value.
The forecast doubles measured future allocation cost and reserves actual twenty-job
array maxima. Raw rows, accounting, exports and Polish reports stay outside Git.

The completed pilot has thirty first attempts, 245 bitwise identity rows and 208830
validated pair records, at 3672 allocated CPU-seconds (1.02 hours). Both raw copies
and their 767-file inventory are verified. The fixture
`fixtures/per_pair_diagnostics_regression.json` pins 180 first-decision records and
six complete archived distance rows (5178 pairs), with shared deterministic fields
and deduplicated protocol metadata. Seven applied mutations fail their tests.

The registered `plan.json` SHA-256 is
`ec83bffbba7ca0d14315138abf03d6b485900d6f3c63dcfa0172ebe74a53d8d5`.
Expected cumulative cost is 5.10 CPU-hours, doubled future cost plus pilot 9.18,
and actual hard reservations plus pilot 14.353333. The proposed cumulative approval
is fifteen CPU-hours. Six twenty-job arrays use 5/8/10/10/4/3-minute limits on
Maze4/5/6/7, Woods14 and MazeMA. No seeds43--46 have been submitted.
`execute_batch.py` re-derives each manifest before the shared one-shot dispatcher;
replications require the approved plan, an explicit CPU-limit approval, a matching
shared scope and a frozen execution-operations snapshot.
