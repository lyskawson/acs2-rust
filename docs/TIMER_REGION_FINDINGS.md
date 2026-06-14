# TIMER_REGION_FINDINGS

Read-only inspection of the Rust ACS2 benchmark (`acs2-bench`) timed region, to assess
whether a wall-clock comparison against the supervisor's Python `cpu_single` (ALCS
`experiment_runnerCPU3.py` / `ACS2CPU3`) is fair. The Python timed region was reported to
include a full per-episode population+knowledge metric scan (O(valid_states × population))
every episode.

**Verdict up front:** the Rust timer wraps ONLY the learner loop. No per-episode or
per-trial metric scan runs inside the Rust timed region. No "knowledge" metric is computed
anywhere in this codebase. Population metrics are computed once per repeat, **after** the
timer stops. The two timed regions are therefore **not** symmetric — Rust's timed region is
strictly the learning work; Python's additionally includes a per-episode metric sweep.

---

## 1. Timer boundaries

All wall-clock instrumentation lives in `acs2-bench/src/main.rs`, inside `run_repeat`
(one repeat = one full explore+exploit experiment on one maze). `std::time::Instant`.

Two separate timers:

- **Explore timer**
  - START — `acs2-bench/src/main.rs:109` → `let explore_start = Instant::now();`
  - STOP  — `acs2-bench/src/main.rs:114` → `let explore_seconds = explore_start.elapsed().as_secs_f64();`
- **Exploit timer**
  - START — `acs2-bench/src/main.rs:117` → `let exploit_start = Instant::now();`
  - STOP  — `acs2-bench/src/main.rs:129` → `let exploit_seconds = exploit_start.elapsed().as_secs_f64();`

`total_seconds = explore_seconds + exploit_seconds` (`main.rs:172`), accumulated across the
`n_exp` repeats per maze (`main.rs:158-159`, `170-172`). No other `Instant`/`elapsed` exists
in the bench or core (`grep` for `Instant|elapsed|Duration` returns only these sites).

## 2. Inside vs outside the timed region

**in-timer (explore, `main.rs:110-113`):** `options.explore_trials` calls to
`agent.run_explore_trial(...)`; each returns `TrialMetrics { steps, reward }`, and the loop
does `time += metrics.steps`. Per `acs2-core/src/agent.rs:53-169`, one explore trial runs
only the learner loop: `form_match_set` → `apply_alp` → `form_match_set` →
`bootstrap.estimate` → `apply_reinforcement_learning` → (optional `apply_ga` +
`form_match_set`) → `selector.select` → `form_action_set` → `env.step`, plus terminal-step
ALP/RL/GA. `TrialMetrics` is built from two O(1) scalar accumulators (`steps`, `total_reward`,
`agent.rs:165-168`). No population scan.

**in-timer (exploit, `main.rs:118-128`):** `exploit_phases × exploit_trials` calls to
`agent.run_exploit_trial(...)` (`agent.rs:171-237`): `form_match_set` →
`apply_reinforcement_learning` → `best_action.select` → `form_action_set` → `env.step`.
`phase_steps.push(metrics.steps as f64)` is an O(1) push. Only the final phase's per-trial
step vector is retained (`final_window`, `main.rs:125-127`) — a move, no computation.

**out-of-timer:** everything reporting-related —
- population metrics `population.len()` / `numerosity()` / `reliable_count(theta_r)`
  (`main.rs:131-136`) run **after** `exploit_start.elapsed()` at line 129;
- `mean(&final_window)` (`main.rs:133`);
- per-maze aggregation `mean` / `population_std` (`main.rs:165-169`);
- CSV formatting and `std::fs::write` (`main.rs:182-227`).

So: **in-timer = [explore learner loop, exploit learner loop, O(1) step/reward
accumulation].  out-of-timer = [population macro/micro/reliable scan, final-window mean,
cross-repeat aggregation, CSV write].**

## 3. Knowledge metric

**Not computed anywhere.** `grep -rln "knowledge|Knowledge"` over `acs2-core`, `acs2-bench`,
`acs2-envs` returns nothing. There is no "fraction of environment transitions a reliable
classifier (q>theta_r) correctly anticipates" computation in this repo — neither inside nor
outside the timer. (This is the single biggest asymmetry vs. the Python `cpu_single`, whose
per-episode metric scan reportedly includes exactly this kind of population×states work.)

## 4. Population metrics

`macro` = `Population::len()` (`acs2-core/src/population.rs:21`), `micro` = `numerosity()`
(sum of numerosity, `population.rs:29`), `reliable` = `reliable_count(theta_r)` (count of
classifiers with `q > theta_r`, `population.rs:33`). In the bench they are read once per
repeat at `main.rs:131-136`, i.e. **end-of-run only and OUTSIDE the timed region** (after the
exploit timer stops at line 129). They are **not** recomputed per-episode or per-trial.
Cross-repeat means are formed later at `main.rs:167-169`, also outside the timer.

## 5. (Secondary) ALCS per-phase epsilon wiring — external repo

Cloned `https://github.com/ounold/ALCS` read-only into `/tmp/alcs-inspect` (HEAD only;
upstream history is bulk "Add files via upload" commits, so per-line provenance is coarse).

- `src/experiment_runnerCPU3.py:136` sets `agent.epsilon = phase_cfg.epsilon` (and `:137`
  `agent.beta = phase_cfg.beta`).
- The agent is `ACS2CPU3` (`acs2CPU3.py`, instantiated at `experiment_runnerCPU3.py:81`).
  Its `__init__` (`src/models/acs2/acs2CPU3.py:13-18`) defines only `cfg`, `population_dict`,
  `population_by_action`, `time`, `curr_ep_idx`. There is **no `epsilon` attribute and no
  `epsilon` setter/property** — the only `@property` is `population` (`acs2CPU3.py:20`).
- Action selection reads **`self.cfg.epsilon`**, not `self.epsilon`:
  `acs2CPU3.py:51` → `elif explore and random.random() < self.cfg.epsilon:`.

Therefore `agent.epsilon = phase_cfg.epsilon` creates a **dead instance attribute** that
nothing ever reads; `cfg.epsilon` (default `0.1` in `confCPU3.py:22`) is what actually drives
exploration, unchanged across phases. Same dead-attribute issue for `agent.beta` (learning
reads `self.cfg.beta`).

**Was it ever wired correctly, then broken?** No evidence of that. Git pickaxe:
- `git log -S "agent.epsilon" -- src/experiment_runnerCPU3.py` → only `b5e16a6` (the single
  upload that introduced the whole file, already with the dead assignment).
- `git log -S "cfg.epsilon ="` (write-through) → **no commits ever**.
- `git log -S "self.epsilon" -- src/models/acs2/acs2CPU3.py` → **no commits** (CPU3 agent
  never read `self.epsilon`). The `self.epsilon` references in history belong to other files
  (`configCPU3.py`, `configGPU4.py`, `acs2GPU4.py` — the GPU variant), not the CPU3 path.

**Conclusion (part 5):** as far as the available history shows, the per-phase
`agent.epsilon` assignment in `experiment_runnerCPU3.py` has **always targeted a dead
attribute** and was never connected to the value action-selection reads (`cfg.epsilon`). It
was not "wired then broken." Caveat: upstream history is squashed bulk uploads, so a
pre-upload working state cannot be fully excluded — but within this repo's recorded history
the assignment was dead from introduction. Correct wiring would be
`agent.cfg.epsilon = phase_cfg.epsilon`.

---

### Bottom line for the timing comparison
Rust timed region = pure learner loop (explore + exploit), zero per-episode metric work, no
knowledge computation. If Python `cpu_single` includes a per-episode O(states×population)
metric/knowledge scan inside its timed region, a raw Rust-vs-Python total-time ratio
overstates Rust's learner-loop advantage by whatever fraction that scan costs in Python —
the regions are not measuring the same thing. For a fair comparison either strip the
per-episode metric scan out of the Python timed region, or add an equivalent scan inside the
Rust timer.
