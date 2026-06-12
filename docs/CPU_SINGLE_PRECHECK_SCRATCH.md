# CPU_SINGLE — Pre-check scratch (STEP 0–2 evidence)

Scratch working notes for the cpu_single-vs-Rust timing comparison. Not the deliverable.
Deliverable = `CPU_SINGLE_COMPARISON.md` (after the run).

## STEP 0 — Isolated env (RESOLVED)
- `uv venv --python 3.11 /Users/aleklyskawa/Desktop/ALCS/.venv-cpu-single`.
- Installed: numpy, pyyaml, matplotlib, **torch 2.12.0 CPU-only** (`--index-url .../whl/cpu`).
- Verified: `python -c "...; import torch; torch.cuda.is_available()"` → **False**. Python 3.11.15.
- cpu_single learner path is **torch-free** (grep `import torch|cuda|tensor` over
  acs2CPU3 / logicCPU3 / classifierCPU3 / confCPU3 / runtime_cpu3 / metricsCPU3 /
  experiment_runnerCPU3 / configCPU3 → all clean).
- Dispatch gate: `UniversalRunner._is_gpu_mode` (universal_runner.py:33) is True only for
  {"gpu","gpu_seq"}; `cpu_single` → `no_mp=True` (universal_runner.py:66) → `_run_cpu_phase`,
  never `_run_gpu_phase`. torch is imported only at module load by universal_runner's top
  imports; the tensor/CUDA path is never executed on cpu_single.

## STEP 1 — PRE-CHECK A (metric scan observational) — HOLDS → knob valid, no fallback
- **Master gate (RNG/mutation purity): HOLDS.**
  - `calculate_metricsCPU3` (metricsCPU3.py:39-117): builds read-only `pop_infos` dicts, does
    bit-comparisons; **no `random`/`np.random`**, **no write** to classifiers/population.
  - `calculate_origin_distributionCPU3` (metricsCPU3.py:120-130): counts by origin_source;
    pure read, no RNG, no mutation.
  - `specified_attribute_count` (classifierCPU3.py:108): `bit_count` arithmetic, pure.
  - The only freq-gated calls are these two (experiment_runnerCPU3.py:156-176). The 10-point
    creation-distribution snapshot (184-199) is gated by `_observation_trigger` (fixed
    `linspace`), **independent of metric_calculation_frequency**.
  - ⇒ changing `metric_calculation_frequency` cannot perturb the RNG stream or trajectory.
    freq=1 and freq=25 produce **identical** trajectories/populations/steps; only timing and
    observation cadence differ.
- **Array shape / slicing (refinement 2b): full-length, forward-filled, slicing correct.**
  - `_empty_episode_series` (experiment_runnerCPU3.py:42-67) = full-length `np.zeros(total_episodes)`.
  - experiment_runnerCPU3.py:178-179 writes `metric_cache` (last computed value) into **every**
    episode slot regardless of the gate at :156 → gaps carry the last real value, **not zeros**,
    **not append-only**.
  - Under UniversalRunner each phase is a separate `run_experimentCPU3` (others zeroed by
    `gpu_config_to_cpu3`, hybrid_utils.py:64); `merge_experiment_stats` concatenates per-phase
    arrays on axis=1 (hybrid_utils.py:123) → merged `stats_macro` = (n_exp, 1500),
    order [explore 0:500 | exploit1 500:1000 | exploit2 1000:1500].
  - `exploit2_population_stats` (run_maze_benchmarks.py:80-97) slices `[:, 1000:]`
    (`exploit2_start = 500+500`). Correct columns; full-length forward-filled ⇒ no zero dilution.
  - freq=25 ⇒ 20 real samples inside the 500-ep exploit2 window (≥10 satisfied); exploit2 has
    ALP off + GA off ⇒ macro frozen, so forward-filled mean ≈ freq=1 mean (RL q-drift only).

## STEP 1 — Aggregation parity (refinement 1) — traced
- cpu_single `total_time_s` = `summary["Total Time"]` = `total_wall_time`
  (universal_runner.py:93) = **OUTER** wall clock summed over 3 phases, each wrapping the whole
  `run_experimentCPU3` (all n_exp=10 workers run sequentially under no_mp). → **total over n_exp**,
  full 1500-ep experiment, **includes** per-phase setup (env/agent/metric-context/initial seed)
  **and** the in-loop metric scan (knob-reduced).
- cpu_single `avg_exp_time_s` = `summary["Avg Time"]` = Σ over 3 phases of mean-over-10-workers
  **INNER** duration (experiment_runnerCPU3.py:130/204; setup-excluded, metric-scan-included).
  → **per-experiment** time for the full 1500-ep experiment.
- Rust P9 `total_s` (TIMER_REGION_FINDINGS): Σ over n_exp=10 repeats of explore+exploit
  **learner-loop** time; setup-excluded, **no** metric scan. → **total over n_exp**.
- Like-for-like: cpu_single `total_time_s` vs Rust `total_s` are both total-over-n_exp, BUT
  cpu_single's includes per-phase setup that Rust excludes. Cross-check at run time:
  compare `total_time_s` vs `avg_exp_time_s × 10`; the gap = setup+inter-phase overhead.
  Report both; use the metrics-symmetric inner-total for the cleaner ratio if the gap is large.

## STEP 2 — PRE-CHECK B (maze geometry) — **PARTIAL: 2 match, 2 are different mazes**
goalState in `.cpp` = {row, col}; Rust marks goal cell with `9`. Both use 1=wall/0=corridor.

| Maze | cpu_single `.cpp` | Rust `acs2-envs/maze_data.rs` | Verdict |
|---|---|---|---|
| **Maze4** | 8×8, full border, goal (1,6), wall layout row-for-row | 8×8, goal (1,6), identical matrix | **CELL-IDENTICAL ✓** |
| **Woods100** | 9×3, 1×7 corridor, central goal (1,4) | 9×3, central goal (1,4), identical | **CELL-IDENTICAL ✓** |
| **Maze7** | **5w × 7h** (~9 free cells, narrow double-corridor), goal (5,1) | **9×9** (~30 free cells), goal (1,7) | **DIFFERENT MAZE ✗** |
| **Woods1** | **7×7** (~17 free cells, irregular), goal (3,3) | **5×5** (9-cell room), goal (1,3) | **DIFFERENT MAZE ✗** |

- Rust side = pyalcs-canonical (P9 validated Rust Maze7 6.74 ≈ pyalcs 6.61; P9 itself calls
  Woods1 "a 9-cell maze"). cpu_single `.cpp` Maze7/Woods1 are differently-shaped mazes under a
  colliding name. This is the exact risk RECON §D flagged.
- Still owed for the two survivors (Maze4, Woods100) before STEP 6, non-blocking: wrap/clamp
  (both have solid wall borders → wrap moot, state so), start-cell set, Rust action count vs
  cpu_single's 8 actions (4 cardinal + 4 diagonal, RECON B5).

## GATE 1 decision
- PRE-CHECK A: clean (knob valid, no two-run fallback).
- PRE-CHECK B: **2 of 4 mazes are not the same maze** → mandated stop. Headline comparison =
  **Maze4 + Woods100** only. Maze7/Woods1: no Rust cross-system ratio. Await user direction.
- freq-validation target after gate = largest-population maze actually executed (Maze4 ~283 if
  Maze7 dropped; Maze7 ~363 if kept as incomparable).
