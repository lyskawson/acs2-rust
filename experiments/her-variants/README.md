# Episodic replay and hindsight copy updates

This experiment compares `her_full`, `her_values` and `her_values_shortened` with
the identical original-only episodic path `er_episodic`. `acs2er_u1` is unchanged
core ACS2ER with one replay application per step. The registered ACS2 and ACS2ER
pair grid is reused. No TU, VCP or correction agent is part of this scope.

`acs2-agents` depends on core and trajectory storage; measurement constructs
agents and implements their raw-episode environment port. Research copies use
the task objective (1000/0 on these mazes) and corrected truncation. The pyalcs
mode is compiled only inside the HER unit differential. It cannot be selected
by the CLI or an experiment manifest.

All episodic learners collect without learning, insert the complete raw episode,
and check retained steps against warmup one. They draw three times the episode
length, with replacement, even when fewer transitions are retained. Skipped
warmup episodes are not repaid. Full steps use core ACS2ER ALP, RL and optional
GA, with one environment-clock timestamp at the episode's end. Values-only
copies form the action set and next match set and apply only RL to `r` and `ir`.
Original and fallback routes always get full steps. A copy equal to its original
goal still belongs to the relabeled route. `her_full` fixes requested p=2/3;
the values variants use separately selected proportions for each maze.

`protocol.py` reconstructs immutable pair configurations, the authorized seed-42
cost pilot, the cost forecast, and the final three-part registration. `plan.json`
is the resulting plan awaiting approval. Its SHA-256 is
`e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444`.
Tuning uses 1/3 and 2/3; 0.9 was removed for cost before tuning. The main grid
has 600 runs with seeds 42--61; tuning evaluates only validation with seeds
42--46. The tie rule and comparisons are registered in the plan. Nothing may
tune on test. Approval must identify these exact bytes before any tuning or
grid submission. A study plan is not an execution manifest.

The update-rule ablation uses `her_values_p2of3` minus `her_full` on every maze.
Where tuning selects 2/3, reuse the tuned values rows. Otherwise add twenty
values runs at 2/3 with identical budgets, points, pairs and caps. This fixed
control enters no other comparison; the tuned values minus full comparison
remains separate. `matched_control.py` derives the required configurations and
checks admission against aggregate spent and outstanding reservations across
all batches. Manifests carry `row_arm=her_values_p2of3`. Collection preserves
the pinned binary's `binary-rows.jsonl` and writes `rows.jsonl` with the same
measurement fields and an `arm` label; both artifacts have verified checksums.
The measurement agent stays `her_values`, and no binary change is needed.

Pilots used 1.5125 CPU-hours. Main grid and tuning hard reservations plus pilots
are 55.1792 hours; their doubled forecast is 57.2329. At most 120 extra control
runs reserve another 9.6667 hours, with expected cost 5.0684. Expected total with
every extra control is 34.4411 hours; simultaneous hard reservations would be
64.8458, and the doubled total forecast 67.3697. Submit required controls as soon
as accounting releases enough reservations, and recheck at main-grid completion.
Admission always enforces sixty hours including every attempt; if controls never
fit, STOP and report incomplete scope. Do not shrink any other arm, maze or seed.
Each job has 512 MiB. These forecasts do not guarantee new-seed population growth.

`thresholds.py` stores both baseline final-success thresholds as exact fractions
and computes crossings with rational arithmetic from integer success counts.
It reports the first registered nominal point, without interpolation, or null.
The count fixture reproduces both thresholds and all 24 baseline crossings.
`identity.py` compares all shared learning, evaluation and knowledge fields,
including IEEE 754 bits. The plan requires seed-42 pilot identity for the fixed
arms, five-seed tuning identity for selected values settings and the fixed
control, and pilot identity wherever the setting is 2/3. The Rust regression
pins all five measured arms on all six mazes at 1k and 3k from verified pilots.

Execution uses `../grid/submit.py`, `grid_job.sh`, node guards and accounting.
Experiment manifests, launches and raw artifacts live outside Git under
`~/tu_runs/5/` on Lem and `~/Desktop/tu-runs/5/` locally. No result belongs under
`reports/`. Each attempt consumes allocated CPUTimeRAW once, including failures.
`verify.py` independently checks terminal accounting, rows and artifact hashes
after pulling. Both copies must verify before analysis.

The differential instruments the unmodified reference with explicit RNG draws.
It covers 1000 random populations/steps, original and copied rewards/endings,
future/final/episode relabeling, expected/unexpected ALP, covering, marks,
subsumption, rewards and bootstrap. It compares all classifier fields, with
floating tolerance 1e-9 like P11. GA/PEE and overspecification are disabled;
theta_i=0 excludes pyalcs's known mid-iteration deletion skip. It does not
validate reference exploration, its exploitation learning, replay timing,
retention distribution or population growth under a training protocol.

```bash
baseline/.venv/bin/python -B baseline/dump_her_differential.py --out fixtures/her_differential.json
cargo test --release -p acs2-agents
cargo test --release -p acs2-measure --test her
python3 -B -m unittest discover -s experiments/her-variants -p 'test_*.py'
python3 -B experiments/her-variants/verify.py <outside-checkout batch root>
```
