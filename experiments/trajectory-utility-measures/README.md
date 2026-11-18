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

`verify_reference.py` reproduces the first checkpoint on the original six tasks
and Littman57 with S+1 and checks the original row hashes and pair records in
binary64 bits. `forecast.py` uses paired archived costs and seed-42 pilot overhead
to generate cost-only tables at all menu points. It reports conditional bounds
from the observed checkpoint ratios and paired archival seed spread. The model
projects the refresh rate from the observed insertion rate, retained steps and
the measured TU draw frequency. Its estimates hold the measured non-utility CPU
ratios fixed as K and F vary.
