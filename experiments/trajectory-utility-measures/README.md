# Trajectory utility measures

`plan.json` registers three ACS2-TUHER arms and the original shortened HER reference
with per-goal values. It inherits 28 maze configurations and three pair blocks from
the ablation by their registered hashes. The evaluation menu remains open.

The measurement CLI selects `acs2tuher_mean_value`, `acs2tuher_trial_update`, or
`acs2tuher_random_goal`. The pilot passes `--tu-candidates 8 --tu-refresh 2000`.
The remaining menu accepts 4 candidates and refresh at 5000 steps or `none`.
Requested original, hindsight and TU routes each have probability 1/3. The route
and candidate streams are independent of the training policy and transition sampler.
The candidate sample is uniformly ordered, so retaining the first maximum breaks ties
uniformly. The random arm takes the first sampled goal.

Trial updates use a private copy-on-write column. The full trajectory state sequence,
including repeated observations and the last state, contributes to both the value gain
and the post-update level. The diagnostic averages the latter across scored candidates.
Goal assignment reach and distinct counts include both insertions and refreshes. The
per-goal write spread includes zero counts. Buffer bytes are logical size for live
episode references, assigned goals and evaluation clocks.

At every checkpoint `trajectory_utility.cost` reports cumulative thread CPU and
work. The scoring timer excludes building the shared observation masks. The cost
model treats that work as part of fixed evaluation overhead. Match sets count
goal-conditioned formations, and observation match sets count the shared masks
formed from distinct observations. Value reads count calls to Q or V. A trial value
update is one replayed prefix transition, and trial reward entries count its rule
writes. Existing match and replay counters measure learning. Runner wall times cover
their complete phases.

`pilot-cells.json` fixes 84 seed-42 cells at this menu point, their checkpoints and
reservations. The full budget is used where the archived S+1 seed-42 process CPU at
full budget was at most 150 seconds. The other cells reach 300k. Execute
`pilot.py build`, then `pilot.py submit` after the musl build completes, from the
clean pinned cluster clone. The build is also charged to the pilot. Use
`pilot.py status` for the allocation state and `pilot.py seal` for the final
accounting ledger and checksum manifest. Short-job arrays form a dependency chain
while the large cells run independently. Submission verifies the grant and runs
SLURM test-only for each array. A failed measurement writes the study STOP.

The Woods102 timeout stopped the initial wave. The recovery snapshot reruns only
incomplete cells, with the same seed, configuration, checkpoints and pinned binary.
It keeps both allocation ledgers and all stopped artifacts. The two remaining MazeE3
cells depend on completion of the shorter retries. Recovery guards count cancelled
dependencies by array element and allocation CPU by JobIDRaw. Completed retries
compare any preserved prefix in binary64 bits and exclude CPU timers. Use
`recovery.py seal` from the frozen operations snapshot to seal both waves together.

`verify_reference.py` reproduces the first checkpoint on the original six tasks
and Littman57 with S+1 and checks the original row hashes and pair records in
binary64 bits. `forecast.py` uses paired archived costs and seed-42 pilot overhead
to generate cost-only tables at all menu points. It reports conditional bounds
from the observed checkpoint ratios and paired archival seed spread. The model
projects the refresh rate from the observed insertion rate, retained steps and
the measured TU draw frequency. Its estimates hold the measured non-utility CPU
ratios fixed as K and F vary.

## Full grid

The approved plan fields fix eight candidates and a 2000-step refresh. The 48
full-budget pilot measurements supply their seed-42 cells. The grid supplies
1632 full measurements, including six MazeE2 and MazeE3 scouts. Seed-42 extensions
compare their overlap to the corresponding sealed pilot rows in binary64 bits.

`grid_limits.py` authenticates the pilot and ablation inputs and derives every
time limit. Ordinary cells use their last matched checkpoint, the maximum archived
S+1 seed cost for that maze, a 50% margin, and a two-minute allowance. On the ten
ordinary 300k prefixes the non-utility ratio has floor one. Scout reservations use
the floor-one non-utility ratio times one plus utility's process CPU share, the
S+1 seed-42 allocation, a 30% margin, and three minutes. The pending heavy arrays
reserve the scout reservation times the archived seed spread with one additional
minute. Their gate replaces that envelope with the same calculation from the
completed scout allocation CPU. The scheduler and node both verify the updated limit.

`grid_run.py prepare INPUT_SHA256` and `grid_run.py submit` run from the frozen
operations snapshot under the grid result directory. The measurement clone stays
pinned to the pilot binary. Ordinary bundles use one allocated CPU and separate
timed Slurm steps for each registered run. Bundling keeps the entire submission
within the association's array element limit. The six scouts can overlap ordinary
bundles. The two heavy array chains follow the ordinary chain and the scout gate.
Before each submission the controller re-derives the assignment, checks the pilot
plus grid budget, the shared grant, queue capacity, and SLURM test-only.

`grid_run.py status` reads cumulative accounting. A failed run or allocation
writes the study STOP and blocks downstream dependencies. `grid_run.py seal`
requires all allocations and runs to complete within the cumulative approval.
After copying and authenticating the grid snapshot, `grid_analysis.py analyze`
combines it with the full pilot runs and original S+1 files. Use
`grid_analysis.py figures` with the registered `curves.csv` for the individual
and grouped test success figures.
