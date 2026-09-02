# CPU_SINGLE vs Rust — timing & population comparison

**Cross-representation systems comparison** of the supervisor's ALCS single-core CPU backend
(`cpu_single`) against the existing Rust ACS2 P9 baseline, on the maze names that are
**cell-identical** between the two codebases: **Maze4** and **Woods100**.

> This is **NOT a language speedup.** The two sides run different perception encodings,
> different reward schemes, different episode caps/counts, and a different GA setting. The
> ratio compares two *systems solving related-but-not-identical workloads*, with a metric-scan
> correction applied to make the timed regions symmetric. Read the **Caveat Ledger** before
> quoting any number.

- **Date:** 2026-06-11. **Machine:** this host (darwin, sequential runs).
- **cpu_single artifact:** `/Users/aleklyskawa/Desktop/ALCS` HEAD, byte-identical to upstream
  `ounold/ALCS` on the measured path (per prior recon). **No measured-artifact source was
  modified** — only a YAML *copy* was pointed at via `--config`.
- **Rust baseline:** `acs2-rust` P9 (`reports/P9_comparison.md`), GA OFF, n_exp=10,
  500 explore + 3×200 exploit.
- **Raw cpu_single CSVs:** `reports/cpu_single_raw/` (`cpu_single_real_f50.csv` = reported run;
  `smoke_f1.csv` / `smoke_f25.csv` / `smoke_woods100_f1.csv` = metric-fraction & purity checks).
- **Env:** isolated `uv` venv, Python 3.11.15, `numpy/pyyaml/matplotlib` + **CPU-only torch
  2.12.0** (`cuda=False`). torch is import-graph-only; the `cpu_single` learner path is
  torch-free and never dispatches to the tensor/CUDA backend (`_is_gpu_mode` gate,
  `universal_runner.py:66`).

Mazes dropped: **Maze7** and **Woods1** — the ALCS `.cpp` definitions are *different mazes* than
the Rust `acs2-envs` ones (Maze7: 5×7 ~9 cells vs 9×9 ~30 cells; Woods1: 7×7 ~17 cells vs 5×5
9-cell), sharing only the name. **Maze5** is absent from `cpu_single` entirely. See PRE-CHECK B.

---

## 1. Reported run configuration

`cpu_single` run is **native batch_mazes.yaml** except for one measurement knob:

| Parameter | Value | Native? |
|---|---|---|
| n_exp / n_steps | 10 / 200 | native |
| phases (episodes) | 500 / 500 / 500 (explore / exploit1 / exploit2) | native |
| epsilon (effective) | **0.1 constant, all phases** (per-phase 0.8/0.2/0.0 is a dead write) | native quirk |
| u_max | 1 | native |
| GA | **on in explore**, off in exploit1/2 | native |
| ALP | on / on / off | native |
| reward | **+1000 goal, −1 per non-goal step** | native |
| perception | absolute (row,col), 2 symbols, 8 actions, random start | native |
| subsumption | disabled | native |
| **metric_calculation_frequency** | **50** (native = 1) | **knob — see timer symmetry** |

---

## 2. Per-maze comparison

`t_cpu_single` and `t_rust` are both **totals over n_exp = 10** (like-for-like aggregation:
Rust `total_s` accumulates the 10 repeats — sum of P9 per-maze totals = 2.20s; cpu_single
`total_time_s` is the outer wall-clock over 3 phases × 10 sequential workers). cpu_single's
total includes ~0.4–0.6 % per-phase setup that Rust's excludes (`total_time_s` vs
`avg_exp_time_s × 10`: Maze4 +0.39 %, Woods100 +0.59 %) — immaterial.

### Timing

| Maze | metric | cpu_single (freq=50) | Rust P9 | ratio t_cpu/t_rust |
|---|---|---|---|---|
| **Maze4** | total time (10 exp) | **11.104 s** | **0.415 s** | **26.8×** |
| | per-experiment | 1.1060 s ± 0.0938 | 0.0415 s | 26.7× |
| | per-episode (derived) | 737 µs (÷1500) | 37.7 µs (÷1100) | 19.6× *(crude)* |
| **Woods100** | total time (10 exp) | **0.9376 s** | **0.047 s** | **20.0×** |
| | per-experiment | 0.0932 s ± 0.0122 | 0.0047 s | 19.8× |
| | per-episode (derived) | 62.1 µs (÷1500) | 4.27 µs (÷1100) | 14.5× *(crude)* |

**Robustness to the metric correction:** stripping the *entire* residual metric scan (using the
metric-fully-excluded learner time `L` derived from the freq=1/freq=K smoke pairs: Maze4
L≈1.08 s/exp, Woods100 L≈0.091 s/exp) moves the total ratio only 26.8×→26.0× (Maze4) and
20.0×→19.4× (Woods100). The conclusion does not depend on the metric treatment.

### Population & steps (exploit2-averaged) — secondary, representation-dependent

| Maze | metric | cpu_single | Rust P9 |
|---|---|---|---|
| **Maze4** | macro / reliable-macro | 197.6 / 80.5 | 283 / 260 |
| | exploit steps-to-goal | 5.394 ± 4.616 | 3.584 ± 0.128 |
| **Woods100** | macro / reliable-macro | 22.7 / 15.8 | 18 / 14 |
| | exploit steps-to-goal | 2.246 ± 1.060 | 17.774 ± 3.007 |

> **Steps-to-goal and population are NOT clean comparisons** — see Ledger items (b), (e), (i).
> The Woods100 steps inversion (cpu_single 2.2 vs Rust 17.8 on the *identical* maze) is the
> sharpest illustration: (row,col) perception makes the corridor trivially solvable, while the
> 8-sensor wall perception aliases every interior cell into the same percept.

**Headline:** on these two cell-identical mazes, the optimized Rust system completes the
benchmark **~20–27× faster** (total wall-clock, metric-symmetric) than the native CPython
`cpu_single` system — a cross-representation systems gap, not a language micro-benchmark.

---

## 3. Timer-symmetry mechanism (the load-bearing methodological move)

Rust times **only** the learner loop (no per-episode metric scan; `TIMER_REGION_FINDINGS.md`).
`cpu_single`'s timed region (`experiment_runnerCPU3.py:130–204`) additionally runs a per-episode
O(states × population) metric scan, gated by `metric_calculation_frequency`. To make the regions
symmetric, the knob was raised **1 → 50** in a YAML copy so the in-timer scan is rare.

This is legitimate because the scan is **purely observational** — verified read-only:

- **Master gate (RNG/mutation purity) HOLDS.** `calculate_metricsCPU3` (metricsCPU3.py:39–117)
  and `calculate_origin_distributionCPU3` (120–130) consume **no RNG** and **mutate nothing**
  (read-only `pop_infos` + bit-comparisons; `specified_attribute_count` = pure `bit_count`).
  The only freq-gated calls are those two (experiment_runnerCPU3.py:156–176); the 10-point
  creation snapshot (184–199) is gated by a fixed `linspace`, independent of the knob.
  ⇒ changing the frequency **cannot perturb the trajectory**.
- **Empirically confirmed:** freq=1 and freq=25 on Maze4 (n_exp=2, same seeds) produced
  **byte-identical** `Exploit Avg = 5.9660 ± 7.9709` and exploit2 populations within 0.01 %
  (macro 188.758 vs 188.742; reliable-macro 77.5 vs 77.5). The knob does **not** corrupt the
  population columns (the per-episode arrays are full-length and forward-filled — last value, not
  zeros — so `exploit2_population_stats`' `[:,1000:]` slice is undamaged; exploit2 has ALP+GA off
  ⇒ macro frozen).
- **Chosen freq = 50** is the *maximum* that still yields ≥10 exploit2 samples (500/50), and it
  drives the residual in-timer metric scan to **1.9 % (Maze4)** / **2.2 % (Woods100)** of the
  per-experiment time — measured from the freq=1/freq=K pairs (Maze4: full-metric 1.06 s/exp,
  learner 1.08 s/exp; Woods100: full-metric 0.10 s/exp, learner 0.091 s/exp). The residual is
  *conservative*: it **inflates** cpu_single's time, so it works *against* cpu_single in the
  ratio, never flatters it.

No two-run fallback was needed (purity holds and forward-fill keeps the population columns valid).

---

## 4. PRE-CHECK B — maze geometry (cell-by-cell)

`goalState` in `.cpp` = {row, col}; Rust marks the goal with `9`; both use 1=wall / 0=path.

| Maze | cpu_single `.cpp` | Rust `acs2-envs/maze_data.rs` | Verdict |
|---|---|---|---|
| **Maze4** | 8×8, full border, goal (1,6), wall layout row-for-row | 8×8, goal (1,6), identical matrix | **CELL-IDENTICAL** |
| **Woods100** | 9×3, 1×7 corridor, central goal (1,4) | 9×3, central goal (1,4), identical | **CELL-IDENTICAL** |
| Maze7 | 5w×7h, ~9 free cells | 9×9, ~30 free cells | different maze — dropped |
| Woods1 | 7×7, ~17 free cells | 5×5, 9-cell room | different maze — dropped |

For the two survivors: **8 actions** both sides (Moore neighbourhood); **random start** both
(uniform over non-wall non-goal cells; Rust `maze.rs:101`, cpu_single `reset_to_random_start`);
**no toroidal wrap** both — moot here since both mazes have solid wall borders (Rust blocks on
wall `maze.rs:116`; cpu_single clamps). Geometry, action set, start regime, and boundary
behavior match; **perception and reward do not** (Ledger).

---

## 5. MANDATORY METHODOLOGY / CAVEAT LEDGER

Every item below is a real difference that the ratio does **not** normalize away:

- **(a) The ratio is a cross-representation SYSTEMS comparison, never a language speedup.**
  Different perception, reward, caps, episode counts, and GA setting are baked into both sides.
- **(b) Perception differs fundamentally.** cpu_single = absolute **(row,col) coordinates**
  (2 symbols, fully observable position); Rust = **8 surrounding wall/path sensors** (8 symbols,
  aliased in uniform regions). Same maze ⇒ different learning problem (see Woods100 steps).
- **(c) Reward differs.** cpu_single = **+1000 goal / −1 per non-goal step**; Rust = **1000 /
  0** (no step penalty; `maze.rs:124`).
- **(d) Episode cap & count differ.** cap: cpu_single **200** (both mazes) vs Rust **50**
  (Maze4) / **500** (Woods100). episodes: cpu_single **1500** (500/500/500) vs Rust **1100**
  (500 + 3×200). The per-episode proxy partially normalizes the *count* but not the cap, phase
  composition, or GA.
- **(e) GA setting differs.** cpu_single runs **GA ON in explore**; Rust P9 is **GA OFF**
  throughout. GA-on grows/reshapes the explore population, changing both timing and final macro.
  **Direction:** GA-on is a cpu_single-only cost that pushes the ratio **up** (slower) — the
  *opposite* of the ε quirk in (f), which pushes it **down** (faster). The two quirks partially
  offset, so the net bias on the timing ratio is **ambiguous**, not one-sided.
- **(f) Epsilon dead write — flatters cpu_single's speed.** The per-phase ε (0.8/0.2/0.0) is a
  dead write; cpu_single runs **effective ε = 0.1 in ALL phases** (not the described schedule).
  Suppressed exploration (vs the intended ε=0.8 explore) tends to yield a **smaller population**,
  and a smaller population means **faster** cpu_single timing — i.e. the quirk makes cpu_single
  look *faster* than its own described configuration would. exploit2 is therefore **not greedy**
  (10 % random actions), which also inflates its steps-to-goal.
- **(g) Timer symmetry knob.** cpu_single `metric_calculation_frequency` reduced **1 → 50** to
  match Rust's metrics-excluded timer. Verified **observational** (no RNG, no mutation; trajectory
  provably unchanged and empirically identical). Residual in-timer metric scan ≈ **1.9 % (Maze4)
  / 2.2 % (Woods100)**, which *inflates* cpu_single time (conservative).
- **(h) Per-episode time is a CRUDE proxy.** Phase composition (GA-on explore vs GA-off) and
  episode caps differ, so dividing by episode count does **not** cleanly isolate per-episode cost;
  reported only as a rough normalizer.
- **(i) Steps-to-goal is NOT a clean comparison** — different representation (b), reward (c),
  and a non-greedy cpu_single exploit2 (f). Reported as a **labelled secondary only**, never a
  conclusion. The Woods100 inversion (2.2 vs 17.8 on an identical maze) proves the point.
- **(j) Population columns are representation-dependent.** macro/reliable-macro reflect different
  classifier spaces ((row,col) vs 8-sensor) and different GA settings. The Maze4 reliable-macro
  gap (80.5 vs 260 — a reliable *fraction* of 41 % vs 92 %, even though cpu_single has fewer
  total classifiers, 197.6 < 283) is **confounded by several simultaneous config differences**
  (representation, GA on/off, reward, and the non-greedy ε quirk); it is **not** interpretable as
  an algorithm-quality difference and cannot be pinned on any single cause.
- **(k) Geometry caveats (PRE-CHECK B).** Only Maze4 & Woods100 are cell-identical. Maze7 &
  Woods1 are different mazes under the same name (dropped). Maze5 absent from cpu_single.

---

## 6. PASS / FAIL — STEP 0–6

| Step | Status | Evidence |
|---|---|---|
| **0** isolated env + torch-free dispatch | **RESOLVED** | uv venv Py 3.11.15; torch 2.12.0 CPU-only (`cuda=False`); cpu_single learner files torch-free; `_is_gpu_mode` routes cpu_single→`no_mp=True` (universal_runner.py:66), never GPU. |
| **1** PRE-CHECK A (scan observational) | **RESOLVED** | Master gate holds — metricsCPU3.py:39–130 no RNG/no mutation; freq-gated calls only at experiment_runnerCPU3.py:156–176; arrays full-length forward-filled (178–179); exploit2 slice `[:,1000:]` valid. Empirically: freq=1≡freq=25 trajectory (Exploit Avg identical, pop within 0.01 %). Knob valid, no fallback. |
| **2** PRE-CHECK B (geometry) | **RESOLVED (PARTIAL overlap)** | Maze4 & Woods100 cell-identical (run); Maze7 & Woods1 different mazes (dropped per user); Maze5 absent. 8 actions / random start / no-wrap match on survivors. |
| **3** configure (knob only) | **RESOLVED** | YAML copy `real_f50.yaml`: native learning config, `metric_calculation_frequency=50`. freq=50 justified: ≥10 exploit2 samples AND ~2 % metric residual (Maze4 1.9 %, Woods100 2.2 % — marginally over, but conservative: it inflates cpu_single, so it only *understates* Rust's edge). No source edit. |
| **4** run (native, n_exp=10) | **RESOLVED** | `cpu_single_real_f50.csv` — Maze4 & Woods100, freq=50, n_exp=10. Smoke pairs for purity/fraction. |
| **5** extract + aggregation parity | **RESOLVED** | All columns pulled; per-episode derived ÷1500; Rust read from P9. Both totals = total-over-n_exp=10 (Rust sum=2.20s confirms); cpu_single setup inclusion <0.6 %, quantified. |
| **6** deliver | **RESOLVED** | This document + raw CSVs + caveat ledger. |

**No ALCS measured-artifact source was modified** (only an untracked YAML copy and venv;
confirm via `git -C /Users/aleklyskawa/Desktop/ALCS status`).
