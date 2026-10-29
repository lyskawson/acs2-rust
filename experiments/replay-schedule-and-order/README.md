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
its B rows remain the schedule identity reference. The default-setting
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

One accounting scope covers fourteen pilot roots across the two rounds, screening
and thirty possible test-arm roots. First-round accounting stays immutable while
its allocations remain charged, with zero pilot offset. The forecast doubles measured costs and reserves actual array maxima. Later
execution uses `grid_dispatch.py`, at most two streams and forty running jobs,
grant and scheduler test-only checks before each submission, immutable operations,
node guards, shared STOP and every attempt charged. Stage1 ends after the plan and
external Polish report. Stage2 requires explicit plan and CPU-limit approval.

```bash
python3 -B -m unittest discover -s experiments/replay-schedule-and-order -p 'test_*.py'
python3 -B experiments/replay-schedule-and-order/local_check.py --out <new external directory>
/usr/bin/python3 -B experiments/replay-schedule-and-order/prepare_pilot.py --out <new external study root> --destination <absolute cluster root> --proportion <verified proportion study> --previous <verified first-round root>
python3 -B experiments/replay-schedule-and-order/archive.py <pulled study root>
python3 -B experiments/replay-schedule-and-order/prepare_plan.py <pulled study root>
```

Use matching CPython3.9 when recomputing saved choices from the earlier study.
Raw rows, launch records, accounting, verified inventories and Polish reports live
under `~/tu_runs/replay-schedule-and-order/` and
`~/Desktop/tu-runs/replay-schedule-and-order/`, outside Git. Earlier registrations,
fixtures and studies stay immutable; the proportion directory receives only tests
for inclusive ties, failed-prefix identity and distinct dose-control crossings.

## Original pilot allocation and superseded forecast

The initial length-weighted C pilot and registration below are historical. The
uniform-episode correction replaces its C rows, measurement identity and cost
forecast before screening. First-round B rows and all original artifacts remain
archived; no screening or test has been authorized.


All sixty seed42 trajectories completed in sixty first attempts, with 490 rows and
5035 allocated CPU-seconds (1.398611 CPU-hours), no failures and no reservations.
All 49 default-setting rows reproduce the selected-proportion test archive on all
three blocks and every shared learning/knowledge field in binary64 bits. The
108-row new fixture pins the first two points of all fifty-four new combinations;
its Rust test reconstructs them through the measurement factory. Earlier fixtures
remain unchanged. Measurement source is `0a19f8e`, musl binary SHA-256
`0d0f4d368668acf7943262d1c8298db0489cff40c6eefd70ad0252a63474cb9e`.

The [registered plan](plan.json), SHA-256
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
Screening and test remain pending explicit approval of this plan and its CPU limit.
