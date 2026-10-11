# Episodic replay and hindsight copy updates

This experiment compares `her_full`, `her_values` and `her_values_shortened` with
the identical original-only episodic path `er_episodic`. `acs2er_u1` is unchanged
core ACS2ER with one replay application per step. The registered phase-4b ACS2
and ACS2ER grid is reused. No TU, VCP or correction agent is part of this scope.

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
is the resulting plan awaiting approval. The final grid has 600 runs with seeds
42--61; tuning evaluates only validation with seeds 42--46. The tie rule and
comparisons are registered in the plan. Nothing may tune on test. Stage one
stops after the costed registration. A study plan is not an execution manifest.

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
