# Replay schedules and episode pass order

Two separate axes extend the shared episodic learner. Constant independent replay
remains the default, with unchanged historical rows and RNG consumption. The B
settings request three draws per interaction through the inclusive episode-end
environment clock T, then one, for T=30000, 100000 or 300000. The C setting makes
backward episode passes with one goal each and the same update budget as the default;
its order control shuffles each pass with independent ChaCha stream 9. B and C cannot
be combined. Acting, evaluation, stores, capacities, warmup and learning rules stay
at the registered preset values.

The base is `acs2her_shortened` at the approved proportion-study selection: Maze4
1/2, Maze5--7 and Woods14 4/5, MazeMA 1/3. The `agent` field retains that name, or
`acs2er_episodic` for controls. Arm labels append `_p1of2`, `_p4of5` or `_p1of3`
for shortened HER, followed by `_replay3to1at30000`, `_replay3to1at100000`,
`_replay3to1at300000`, `_backward` or `_pass_random`. The identity pilot's base
has no replay suffix. ER labels have no proportion suffix.

Passes select each retained episode with equal probability. Rows name this rule
`uniform_episode_passes`. A copy requests one goal
from the existing episode strategy, retaining achieved-index multiplicities and
applying admissibility at the pass start and at each transition. Empty choices fall
back to the original goal. Counterfactual passes stop at first arrival; other rules
keep their own admitted indices. The backward suffix of the last pass is selected
before any shuffle, preserving both exact volume and the order control's transition
set. The sampler stream remains 7; shuffle never consumes it. Counters distinguish
per-pass requests from realized per-update copies.

For complete original-goal passes, uniform episode choice gives every retained
transition the same expected update count. Counterfactual prefixes and the final
budget cut add goal- and position-dependent weighting. The random-pass control
shares these passes and isolates within-pass order
conditional on the same store and sampler state. Policies may diverge during training.
This is backward ACS2 replay inspired by
[Lee, Choi and Chung (2019)](https://papers.nips.cc/paper_files/paper/2019/file/e6d8545daa42d5ced125a4bf747b3688-Paper.pdf),
not the paper's temporary-Q-table target construction or diffusion coefficient.

`replay_schema.py` in the shared grid tooling admits complete explicit axis settings,
refuses their combination and requires exact parameter values and types. Switched
rows carry episode-end clock counts, from which their exact draw volume is checked.
Pass rows carry exact emitted/available steps, request/fallback counts and cuts.
Every shortened setting must have zero trivial and post-end copies. New fields are
absent from default rows. `collect.py` passes both axes to the measurement CLI;
the experiment authorizer mechanically derives each exact registered arm before
submission. It also recomputes saved validation choices for conditional tests.

Stage1 retains the original sixty seed42 pilot allocations and adds twenty-four
with uniform episode choice: shortened backward, shortened random-pass, episodic ER
backward and shortened default on six mazes, all three blocks. Both rounds share
twelve allocated CPU-hours including reservations. The first round remains immutable;
its B rows remain the schedule identity reference. The six new default allocations
also reconstruct all seventy-two B fixture rows with the frozen measurement binary,
then record a separate bitwise gate result. These short checks share those
allocations and the cumulative pilot cap. The default-setting
pilot must reproduce all selected-p proportion test fields in binary64 bits, excluding
only shared provenance/time exclusions. `prepare_pilot.py` verifies that approved
archive before freezing six test references and thirty validation-only base
trajectories. No new scientific reference arm is estimated. The local check is six
sequential short runs on Maze4 and Woods14 and ranks nothing.

After a verified pilot, `prepare_plan.py` pins the nine new settings and registers
one plan. Screening is validation-only, five seeds, four settings, 120 new runs;
the base reuses proportion tuning. Select independently per maze and axis using
the within-seed checkpoint mean then the five-seed mean; base wins within 1e-12.
Test only non-base selections on seeds42--61. B runs shortened HER and ER at the
same schedule; C runs shortened backward, ER backward and shortened random-pass.
Where base wins, that axis allocates nothing on that maze.

The plan fixes both test views, four paired contrasts per tested axis, the C order
contrast, marginal t(19) intervals and exact inclusive crossings of both archived
thresholds. `diagnostics.py` reports realized update rates from every checkpoint
interval, copy composition, value gaps and model/resource costs. All other success
comparisons, including ACS2, are post hoc. Historical references use
`grid/agent_names.json` and are reused without allocations.

One accounting scope covers fourteen completed pilot roots across the two rounds,
the failed default identity attempt root, screening and thirty possible test-arm roots. First-round accounting stays immutable while
its allocations remain charged, with zero pilot offset. The forecast doubles measured costs and reserves actual array maxima. Later
execution uses `grid_dispatch.py`, at most two streams and forty running jobs,
grant and scheduler test-only checks before each submission, immutable operations,
node guards, shared STOP and every attempt charged. Stage1 ends after the plan and
external Polish report. The user approved the exact revised plan with a cumulative
sixty-hour limit; the registered screening and conditional tests are complete.

```bash
python3 -B -m unittest discover -s experiments/replay-schedule-and-order -p 'test_*.py'
python3 -B experiments/replay-schedule-and-order/local_check.py --out <new external directory>
/usr/bin/python3 -B experiments/replay-schedule-and-order/prepare_pilot.py --out <new external study root> --destination <absolute cluster root> --proportion <verified proportion study> --previous <verified first-round root> [--failed <sealed failed-pilot root>]
python3 -B experiments/replay-schedule-and-order/archive.py <pulled study root>
python3 -B experiments/replay-schedule-and-order/prepare_plan.py <pulled study root>
```

Use matching CPython3.9 when recomputing saved choices from the earlier study.
Raw rows, launch records, accounting, verified inventories and Polish reports live
under `~/tu_runs/replay-schedule-and-order/` and
`~/Desktop/tu-runs/replay-schedule-and-order/`, outside Git. Registrations and fixtures
of preceding studies stay immutable; the proportion directory receives only tests
for inclusive ties, failed-prefix identity and distinct dose-control crossings.

## Original pilot allocation and superseded forecast

The initial length-weighted C pilot and registration below are historical. The
uniform-episode correction replaces its C rows, measurement identity and cost
forecast before screening. First-round B rows and all original artifacts remain
archived; screening and test were not yet authorized at that registration.


All sixty seed42 trajectories completed in sixty first attempts, with 490 rows and
5035 allocated CPU-seconds (1.398611 CPU-hours), no failures and no reservations.
All 49 default-setting rows reproduce the selected-proportion test archive on all
three blocks and every shared learning/knowledge field in binary64 bits. The
108-row new fixture pins the first two points of all fifty-four new combinations;
its Rust test reconstructs them through the measurement factory. Earlier fixtures
remain unchanged. Measurement source is `0a19f8e`, musl binary SHA-256
`0d0f4d368668acf7943262d1c8298db0489cff40c6eefd70ad0252a63474cb9e`.

The [superseded registration](https://github.com/lyskawson/acs2-rust/blob/d4011fcb79effaa972598ce5b6cb33470233ceea/experiments/replay-schedule-and-order/plan.json), SHA-256
`3de02461064593d2d99ad98c3dcf95ad91f11ed2c7191f3c7bb369e49619f4e6`,
expects 19.931944 cumulative CPU-hours when both axes are tested everywhere. Its
doubled future-cost forecast including the measured pilot is 38.465278 hours;
actual array hard limits total 59.065278 hours. The proposed approval is sixty
cumulative CPU-hours. Screening arrays group similar measured hard limits into
20--40 jobs, minimizing reservations without using success values. Five arrays
cover all 120 screening runs. Every conditional test arm uses its own twenty-seed
array. Where base wins, that axis runs nothing on that maze.

Every one of ten pilot admissions retained the grant, shared queue and scheduler
test-only evidence. Independent reconstruction checks all sixty allocation records,
5035 CPU-seconds, all hard bounds, at most two active streams and a twelve-running-job
hard bound. The completed pilot's maximum admission bound was 9505 CPU-seconds.
The source, static binary and preset are part of the approval identity; a changed
measurement identity or a changed screening manifest cannot authorize tests.
Screening and test were pending explicit approval of that plan and its CPU limit.

## Uniform episode pilot and revised registration

The corrected pilot source is `416637251460b91c771ffe9d8c3c503776c13986`, static musl binary
SHA-256 `ec99ead899f523e7e157e505b160a520f650ae439bee47e5cddeb7edb9ad4275`.
All twenty-four corrected trajectories completed. Default identity passes 49/49
rows across all blocks and learning/knowledge fields; the same binary passes all
72 unchanged B fixture rows in short checks within the default allocations.
Only 36 C fixture rows and the fixture header provenance change. The 31 other
fixture files retain their bytes. The exact-cycle transition exposure test rejects
a length-proportional mutation; other C tests remain in place.

The first fixture gate used an unsupported full-block CLI argument and triggered
shared STOP before C. Six failed allocations cost 222 CPU-seconds; all 37 available
rows reproduce the reference and the completed retry. The original failed root
remains sealed. `failed_pilot.py` checks its terminal accounting and retry identity.
Both rounds total ninety allocations and 8147 CPU-seconds (2.263056 hours), zero
reservations, within the cumulative twelve-hour cap. The 46-root accounting scope
includes original, failed and corrected allocations exactly once, with zero offset;
its immutable roots preserve earlier accounting files. All fifteen admissions passed
the grant and scheduler test-only checks, with at most two streams and twelve jobs.

The [revised plan](plan.json) SHA-256 is
`01d93796e56efeea2dc2b85e0264bc98dfaec4a16fe7cb08052fd0515f7b1f71`. Scientific screening,
selection, conditional tests, references and analysis are unchanged apart from the
uniform-episode limitation. Expected cumulative cost is 20.692222
CPU-hours; doubled future cost plus pilot is 39.121389; hard
reservations plus pilot total 59.263056. The proposed
cumulative approval remains sixty CPU-hours. This registration preceded all screening
and test measurements.

## Validation choices, conditional tests and analysis

The user approved the revised plan hash above with sixty cumulative allocated
CPU-hours. [approval.json](approval.json) records that authorization without changing
the registered bytes. `approved_execution.py` invokes the shared one-shot dispatcher
with exact mechanical manifests, frozen execution operations, a single 46-root
accounting scope, grant and scheduler test-only checks before every submission.
No background controller, dependency queue, lease or host pinning is used. At most
two streams and forty running jobs are admitted. The measurement clone stays at
`416637251460b91c771ffe9d8c3c503776c13986`; the approved musl binary was checked
before the first and after the last submission and was not rebuilt.

All 120 validation-only screening runs completed in five arrays without failures.
`save_selection.py` saved and independently re-derived the choices from the complete
screening rows and the thirty reused validation-only base trajectories. B selected
30000 on Maze4 and Woods14, 100000 on MazeMA, and base on Maze5--7. The tied MazeMA
100000/300000 candidates use the earlier registered setting. C selected base on all
six mazes. The resulting six twenty-seed B arms completed in 120 allocations with
all three evaluation blocks and no failures or retries. No C test or Maze5--7 B test
was allocated. Base selection does not establish a twenty-seed order effect.

Seed42 screening agrees with the appropriate pilots on 196 shared rows; seed42 tests
agree with the original B pilot on 50 rows. Shortened test seeds42--46 reproduce
125 validation screening rows. Every comparison uses binary64 field identities;
differences are zero. `verify.py` re-derives every manifest and the selection, checks
all attempts and admissions, the historical failed-prefix identity, unchanged pilot
inventories, source/binary guards and cumulative accounting. The corrected pilot's
original 690-file snapshot remains sealed beside the active study root.

`references.py` freezes 480 full trajectories: selected-proportion shortened HER,
episodic ER, ACS2 and ACS2ER on six mazes and twenty seeds. Registration, manifest,
row and completion identities, historical names and failed-attempt prefixes are
checked before copying; no reference arm is re-run. Their historical allocation
cost is reported separately and is not charged again.

`analysis.py` and the shared `grid/paired_statistics.py` reconstruct success from
integer exhaustive pair counts. `analyze.py` exports the two registered views,
paired seed differences, marginal t(19) intervals, exact inclusive thresholds,
screening means/SE, all checkpoint-interval update rates, copy/pass counters,
value gaps for all/successful/failed pairs, knowledge, population, logical memory,
RSS, elapsed/process times and allocated cost by purpose. Final-point diagnostics
use only the final point. `validate_exports.py` independently reconstructs the
statistics and diagnostics from exported counts and checkpoint metrics.

The [paired results](../../docs/ARCHITECTURE.md#validation-choices-and-paired-results-for-replay-schedules-and-episode-order)
contain no confirmed schedule gain or HER-specific interaction in either view.
The realized final update rate falls to about 1.03, 1.015 and 1.10 on Maze4,
Woods14 and MazeMA. These tests do not establish equivalence. All 330 allocations,
including ninety pilot allocations and their six historical failures, cost 21889
CPU-seconds (6.080278 hours); reservations are zero. Screening used 9930 seconds,
conditional tests 3812, and both pilots 8147. All 26 admissions passed the scope,
grant, scheduler and concurrency checks.

Recompute with matching CPython3.9 from the complete external archive:

```bash
/usr/bin/python3 -B experiments/replay-schedule-and-order/verify.py --root <study root>
/usr/bin/python3 -B experiments/replay-schedule-and-order/analyze.py --root <study root> --out <external analysis directory>
/usr/bin/python3 -B experiments/replay-schedule-and-order/validate_exports.py --root <study root> --out <external analysis directory>
uv run --project tools --locked python -B experiments/replay-schedule-and-order/plot_results.py --analysis <external analysis directory> --out <external figures directory>
```

The external `stage2/` directory contains the Polish report, thirteen CSV exports,
`results.json`, three figures in PNG/PDF, gate and mutation logs, and archive
verification evidence. Cluster and laptop copies retain the raw rows, all attempts,
accounting, frozen operations, references and completed checksum inventories.
