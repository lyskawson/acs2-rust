# CPU_SINGLE vs Rust — full 22-maze timing & population comparison

**Cross-representation systems comparison** of the supervisor's ALCS single-core CPU backend
(`cpu_single`) against the Rust ACS2 baseline, over the **22 supervisor mazes** ported into the
Rust repo's `unold` namespace (ids suffixed `-ounold`, source = `OunoldAlcs`, episode cap 200).
This extends the earlier 2-maze run (`CPU_SINGLE_COMPARISON.md`, Maze4 + Woods100) to the full set
with **identical methodology**.

> This is **NOT a language speedup.** The two sides run different perception encodings (cpu_single
> = absolute (row,col); Rust = 8 wall sensors), different reward (−1/step vs 0/step), different
> episode counts (1500 vs 1100) and GA setting (on-in-explore vs off). The ratio compares two
> *systems solving related-but-not-identical workloads*, with a metric-scan correction making the
> timed regions symmetric. **Read the Caveat Ledger (§7) before quoting any number.**

- **Date:** 2026-06-12. **Machine:** this host — MacBook **M1** (darwin 24.5.0), sequential runs,
  the two timed sweeps run **one-at-a-time** (never concurrently) to avoid CPU contention.
- **Time provenance:** both sides timed on **this M1**. cpu_single `total_time_s` / `avg_exp_time_s`
  from the run-driver; Rust total / per-experiment from the `acs2-bench` run on the `-ounold`
  geometries. The supervisor's supplement **Table 2 is NOT used for time** (his Windows/28-thread
  box — not comparable). Steps are machine-independent and cross-checkable against Table 1, but the
  steps reported here are the ones actually measured on this M1.
- **cpu_single artifact:** `/Users/aleklyskawa/Desktop/ALCS` HEAD (= `ounold/ALCS`). **No measured
  source modified** — only a YAML *copy* pointed at via `--config` (see §3, §8).
- **Raw CSVs:** `reports/cpu_single_raw/cpu_single_full_f50.csv` (cpu_single, 22 mazes, n_exp=10,
  freq=50); `reports/bench_rust_unold.csv` (Rust, 22 mazes, n_exp=10, GA off). YAML copy:
  `reports/cpu_single_raw/batch_mazes_f50.yaml`. Heavy-maze residual spot-check (§4):
  `reports/cpu_single_raw/mazeE3_residual_f{1,50}_nexp2.csv` (+ their `.yaml` configs).

---

## 1. Environment

Isolated `uv` venv `/Users/aleklyskawa/Desktop/ALCS/.venv-cpu-single`, **Python 3.11.15**,
`numpy / pyyaml / matplotlib` + **CPU-only torch 2.12.0** (`torch.cuda.is_available() == False`).
torch is import-graph-only: the `cpu_single` learner path is torch-free and **never dispatches to
the tensor/GPU backend** — `_is_gpu_mode` (`src/universal_runner.py:33`) returns true only for
`{"gpu","gpu_seq"}`, so `cpu_single` routes to `_run_cpu_phase(no_mp=True)` (`:66-67`). Confirmed
this run.

---

## 2. Geometry parity — ALL 22 cell-identical (independent check)

The cross-system comparison requires each Rust `-ounold` geometry to be the *same maze* as the
cpu_single catalog maze of the same name. This was **independently verified cell-by-cell** for all
22 (not assumed from the port): the ALCS catalog matrix (`environment/maze_loader.load_acs2_maze_catalog`
— the exact source `cpu_single` runs) was diffed against the Rust matrix const for every maze.
**Dimensions, full wall-set, and goal cell match on all 22:**

| maze | dims (h×w) | walls (rust==alcs) | goal | maze | dims | walls | goal |
|---|---|---|---|---|---|---|---|
| Cassandra4x4 | 6×6 | 20==20 | (4,4) | MazeE3 | 11×11 | 40==40 | (5,5) |
| Littman57 | 4×13 | 37==37 | (2,9) | MazeF2 | 6×5 | 23==23 | (1,3) |
| Littman89 | 7×9 | 40==40 | (4,7) | MazeF3 | 6×6 | 27==27 | (1,4) |
| Maze10 | 6×9 | 35==35 | (4,3) | MazeF4 | 6×7 | 31==31 | (1,5) |
| Maze4 | 8×8 | 37==37 | (1,6) | MiyazakiA | 8×8 | 34==34 | (6,3) |
| Maze7 | 7×5 | 25==25 | (5,1) | MiyazakiB | 8×8 | 36==36 | (3,6) |
| MazeA | 8×8 | 41==41 | (3,6) | Woods1 | 7×7 | 32==32 | (3,3) |
| MazeB | 8×8 | 38==38 | (2,6) | Woods100 | 3×9 | 20==20 | (1,4) |
| MazeD | 8×8 | 39==39 | (4,5) | Woods101 | 5×7 | 24==24 | (3,3) |
| MazeE1 | 9×9 | 36==36 | (4,4) | Woods101_5 | 11×7 | 55==55 | (1,3) |
| MazeE2 | 9×9 | 32==32 | (4,4) | Woods102 | 11×7 | 49==49 | (1,3) |

This is a full PRE-CHECK-B for all 22 (the prior doc hand-verified only Maze4/Woods100). The Rust
fidelity tests (`acs2-envs/tests/unold_geometry.rs`) also pass: well-formedness, goal-reachable,
and exhaustive per-cell × 8-action transition check (walls block, open cells move, goal terminates
@ reward 1000). **Perception and reward still differ by design** (Ledger §7).

---

## 3. Run configuration

`cpu_single` is **native `batch_mazes.yaml`** except for one measurement knob:

| Parameter | Value | Native? |
|---|---|---|
| n_exp / n_steps | 10 / 200 | native |
| phases (episodes) | 500 / 500 / 500 (explore / exploit1 / exploit2) = 1500 | native |
| epsilon (effective) | **0.1 constant, all phases** (per-phase 0.8/0.2/0.0 is a dead write) | native quirk |
| u_max | 1 | native |
| GA | **on in explore**, off in exploit1/2 | native |
| ALP | on / on / off | native |
| reward | **+1000 goal, −1 per non-goal step** | native |
| perception | absolute (row,col), 2 symbols, 8 actions, random start | native |
| subsumption | disabled | native |
| **metric_calculation_frequency** | **50** (native = 1) | **knob — timer symmetry (§4)** |

Rust `acs2-bench`: **GA OFF**, n_exp=10, 500 explore + 3×200 exploit = 1100 episodes, episode cap
200 (carried by the `-ounold` geometries), seed 42.

---

## 4. Timer-symmetry mechanism + heavy-maze validation

Rust times **only** the learner loop (no per-episode metric scan; `TIMER_REGION_FINDINGS.md`).
`cpu_single`'s timed region additionally runs a per-episode O(states × population) metric scan,
gated by `metric_calculation_frequency`. The knob was raised **1 → 50** in a YAML *copy* so the
in-timer scan is sparse, making the two timed regions comparable. This is legitimate because the
scan is **purely observational** — it consumes **no RNG** and **mutates nothing**
(`metricsCPU3.py`), so it cannot perturb the learning trajectory; exploit2 (ALP+GA off) has a
frozen population, so a sparse sample still yields a stable average.

**Heavy-maze validation (new this run).** The prior 2% residual was measured only on the two small
mazes. Re-checked on **MazeE3** (the heaviest population, exploit2 macro ≈ 526):

| | per-experiment time | Exploit Avg (trajectory) |
|---|---|---|
| freq=1 (1500 scans) | **12.124 s** | 20.0110 ± 33.8045 |
| freq=50 (30 scans) | **5.928 s** | 20.0110 ± 33.8045 |

- **Trajectory byte-identical** between freq=1 and freq=50 → the knob is observational on heavy
  mazes too (not just the small ones).
- Per-scan cost = (12.124 − 5.928)/(1500 − 30) = **4.22 ms**; residual at freq=50 = 30 × 4.22 ms =
  **0.127 s = 2.1 %** of the freq=50 time.
- The residual stays ≈ **2 %** across the whole population range (Maze4 1.9 %, Woods100 2.2 %,
  MazeE3 2.1 %) because both the scan **and** the learner scale with states×population, so their
  ratio is roughly population-invariant. The residual always **inflates** cpu_single's time, so it
  works **against** cpu_single in the ratio — conservative.

---

## 5. Per-maze timing (this M1, both sides)

`t_cpu` and `t_rust` are both **totals over n_exp = 10** (like-for-like). cpu_single's total
includes per-phase setup that Rust's timer excludes — quantified as ~0.2–0.6 % (e.g. Maze4
`avg_exp×10`=11.186 vs total 11.213 → +0.24 %; MazeE3 54.98 vs 55.10 → +0.23 %), immaterial.

Per-episode is a **crude** proxy: cpu_single `avg_exp_time_s / 1500`; Rust `total_time_s / (10 × 1100)`
= `/11000` (Rust `total_time_s` accumulates all 10 experiments — `acs2-bench/src/main.rs:152-172`).

| maze | t_cpu (10exp, s) | t_rust (10exp, s) | **ratio t_cpu/t_rust** | cpu /exp (s) | rust /exp (s) | cpu /ep (µs) | rust /ep (µs) |
|---|---|---|---|---|---|---|---|
| Cassandra4x4 | 5.260 | 0.244 | 21.5× | 0.5185 | 0.0244 | 345.7 | 22.2 |
| Littman57 | 6.183 | 0.186 | 33.3× | 0.6171 | 0.0186 | 411.4 | 16.9 |
| Littman89 | 11.680 | 0.299 | 39.1× | 1.1660 | 0.0299 | 777.4 | 27.2 |
| Maze10 | 40.129 | 1.127 | 35.6× | 4.0115 | 0.1127 | 2674.3 | 102.4 |
| Maze4 | 11.213 | 0.438 | 25.6× | 1.1186 | 0.0437 | 745.7 | 39.8 |
| Maze7 | 21.648 | 0.930 | 23.3× | 2.1639 | 0.0930 | 1442.6 | 84.5 |
| MazeA | 23.638 | 0.309 | **76.5×** | 2.3618 | 0.0309 | 1574.5 | 28.1 |
| MazeB | 11.498 | 0.539 | 21.3× | 1.1474 | 0.0539 | 764.9 | 49.0 |
| MazeD | 10.681 | 0.314 | 34.0× | 1.0659 | 0.0314 | 710.6 | 28.5 |
| MazeE1 | 21.829 | 0.714 | 30.6× | 2.1779 | 0.0714 | 1452.0 | 64.9 |
| MazeE2 | 21.900 | 1.716 | **12.8×** | 2.1842 | 0.1716 | 1456.1 | 156.0 |
| MazeE3 | 55.102 | 3.827 | **14.4×** | 5.4977 | 0.3827 | 3665.1 | 347.9 |
| MazeF2 | 2.076 | 0.088 | 23.5× | 0.2071 | 0.0088 | 138.0 | 8.0 |
| MazeF3 | 15.763 | 0.137 | **115.2×** | 1.5755 | 0.0137 | 1050.3 | 12.4 |
| MazeF4 | 27.118 | 1.066 | 25.4× | 2.7109 | 0.1066 | 1807.3 | 96.9 |
| MiyazakiA | 8.998 | 0.439 | 20.5× | 0.8968 | 0.0439 | 597.9 | 39.9 |
| MiyazakiB | 9.747 | 0.635 | 15.3× | 0.9720 | 0.0635 | 648.0 | 57.7 |
| Woods1 | 4.172 | 0.109 | 38.3× | 0.4159 | 0.0109 | 277.2 | 9.9 |
| Woods100 | 0.944 | 0.045 | 20.9× | 0.0939 | 0.0045 | 62.6 | 4.1 |
| Woods101 | 19.620 | 0.244 | **80.3×** | 1.9611 | 0.0244 | 1307.4 | 22.2 |
| Woods101_5 ‡ | 45.574 | 1.577 | 28.9× | 4.5558 | 0.1577 | 3037.2 | 143.3 |
| Woods102 ‡ | 60.401 | 1.509 | 40.0× | 6.0380 | 0.1509 | 4025.3 | 137.1 |

‡ geometry-dominated — see §7(k).

**Aggregate:** Σt_cpu = **435.17 s**, Σt_rust = **16.49 s** → **aggregate ratio 26.4×**
(sum/sum). Per-maze ratio: **median 27.3×**, mean 35.3×, range **12.8× – 115×**. Maze4 (25.6×)
and Woods100 (20.9×) reproduce the prior 2-maze run (26.8× / 20.0×; small drift = fresh M1 Rust
run vs the prior P9 numbers).

**Episode-count-normalized.** The total ratio bundles a fixed 1500-vs-1100 episode-count factor
(`total ≈ avg_exp×10` uniformly, so the per-episode *ratio* is just the total ratio × 1100/1500 =
× 0.733, carrying no new per-maze signal — which is why §5 shows per-episode *times*, not ratios).
Normalized: aggregate **26.4× → ~19.3× per episode**, median **27.3× → ~20×**. So the gap is
**not** an artifact of cpu_single running more episodes.

**Why the ratio swings 12.8×–115× (representation, not language).** The per-episode proxy
normalizes episode *count* but NOT *steps-per-episode*, and steps-per-episode is set by the
representation. Where the 8-sensor encoding aliases interior cells, **Rust** takes many more steps
(MazeE2 61.7, MazeE3 137.4 vs cpu_single 7.8, 15.9) → longer Rust episodes → more Rust work → ratio
*shrinks* (12.8×, 14.4×). Where the (row,col) encoding makes a maze hard for cpu_single but easy
for Rust (MazeF3 cpu 104 steps vs Rust 3.4; MazeA 40 vs 4.3; Woods101 83.5 vs 19.4) → longer
cpu_single episodes → ratio *inflates* (115×, 76×, 80×). The ratio therefore partly reflects
**which side's representation suits each maze**, not a per-operation speed constant.

---

## 6. Population & steps (exploit2-averaged) — secondary, representation-dependent

> **Steps-to-goal and population are NOT clean comparisons** (Ledger §7 b,e,f,j). Reported as a
> labelled secondary only, never a conclusion.

The Rust population/steps columns are the **final-exploit-window** measure — the exploit2-equivalent
under the pinned benchmark protocol (Rust's last 200-trial exploit phase ≡ cpu_single's exploit2),
so the phase mapping against cpu_single's `*_exploit2_avg` is like-for-like.

| maze | cpu macro / rel-macro | rust macro / reliable | cpu steps ± | rust steps ± |
|---|---|---|---|---|
| Cassandra4x4 | 117.1 / 27.0 | 258.4 / 151.5 | 5.035 ± 6.05 | 3.131 ± 0.57 |
| Littman57 | 90.0 / 48.6 | 104.7 / 93.3 | 5.702 ± 7.75 | 4.003 ± 0.20 |
| Littman89 | 154.3 / 85.3 | 206.3 / 168.6 | 5.712 ± 4.61 | 4.269 ± 0.35 |
| Maze10 | 80.9 / 59.8 | 121.0 / 99.7 | 147.995 ± 83.2 | 124.962 ± 7.61 |
| Maze4 | 197.6 / 80.5 | 281.5 / 263.9 | 5.394 ± 4.62 | 3.634 ± 0.13 |
| Maze7 | 38.9 / 37.2 | 95.8 / 89.0 | 141.092 ± 80.9 | 95.760 ± 9.37 |
| MazeA | 149.5 / 102.0 | 173.2 / 171.7 | 40.398 ± 73.4 | 4.290 ± 0.06 |
| MazeB | 190.0 / 84.5 | 301.7 / 253.7 | 5.514 ± 4.78 | 4.240 ± 0.25 |
| MazeD | 179.7 / 91.5 | 261.3 / 198.3 | 4.286 ± 3.79 | 2.911 ± 0.09 |
| MazeE1 | 297.9 / 78.9 | 496.2 / 177.5 | 9.865 ± 18.6 | 6.332 ± 1.65 |
| MazeE2 | 337.0 / 62.0 | 600.4 / 183.2 | 7.778 ± 13.2 | 61.654 ± 18.8 |
| MazeE3 | 526.5 / 92.1 | 573.9 / 259.4 | 15.930 ± 28.6 | 137.375 ± 3.89 |
| MazeF2 | 37.6 / 25.0 | 87.8 / 87.8 | 3.270 ± 1.63 | 2.483 ± 0.06 |
| MazeF3 | 40.2 / 37.5 | 97.7 / 97.7 | 103.982 ± 90.9 | 3.350 ± 0.11 |
| MazeF4 | 46.0 / 44.9 | 114.0 / 100.4 | 157.599 ± 72.9 | 101.769 ± 17.6 |
| MiyazakiA † | 222.8 / 66.1 | 334.5 / 227.2 | 4.631 ± 5.07 | 3.299 ± 0.19 |
| MiyazakiB | 209.4 / 74.7 | 327.8 / 256.6 | 5.144 ± 2.69 | 3.684 ± 0.11 |
| Woods1 | 127.7 / 44.6 | 203.1 / 118.7 | 3.094 ± 3.06 | 1.799 ± 0.07 |
| Woods100 | 22.7 / 15.8 | 18.0 / 14.0 | 2.246 ± 1.06 | 17.777 ± 3.01 |
| Woods101 | 53.9 / 35.2 | 98.8 / 85.2 | 83.505 ± 89.1 | 19.436 ± 2.42 |
| Woods101_5 ‡ | 89.0 / 72.7 | 102.8 / 64.1 | 154.242 ± 79.7 | 154.595 ± 8.73 |
| Woods102 ‡ | 133.8 / 116.4 | 136.5 / 103.9 | 152.153 ± 80.4 | 142.436 ± 7.44 |

† MiyazakiA goal at (6,3) — see §7(k). ‡ geometry-dominated — see §7(k).

---

## 7. MANDATORY CAVEAT LEDGER

Every item is a real difference the ratio does **not** normalize away:

- **(a) Cross-representation SYSTEMS comparison, not a language speedup.** Different perception,
  reward, caps, episode counts, and GA setting are baked into both sides.
- **(b) Perception differs fundamentally.** cpu_single = absolute **(row,col)** (2 symbols, fully
  observable position); Rust = **8 surrounding wall/path sensors** (aliased in uniform regions).
  Same maze ⇒ different learning problem — this is the dominant driver of the §5 ratio swings and
  the step inversions in §6 (Woods100: cpu 2.2 vs Rust 17.8; MazeE3: cpu 15.9 vs Rust 137.4).
- **(c) Reward differs.** cpu_single = **+1000 goal / −1 per non-goal step**; Rust = **1000 / 0**
  (no step penalty).
- **(d) Episode cap & count differ.** episodes: cpu_single **1500** (500/500/500) vs Rust **1100**
  (500 + 3×200). Both cap 200 here. The per-episode proxy normalizes the *count* but not the cap,
  phase composition, GA, or steps-per-episode.
- **(e) GA setting differs.** cpu_single **GA ON in explore**; Rust **GA OFF** throughout. GA-on is
  a cpu_single-only cost that pushes the ratio **up** (slower) — the *opposite* of the ε quirk (f),
  which pushes it **down**. The two partially offset; net timing bias is **ambiguous**, not
  one-sided.
- **(f) Epsilon dead write — flatters cpu_single's speed.** The per-phase ε (0.8/0.2/0.0) is a dead
  write; cpu_single runs **effective ε = 0.1 in ALL phases**. Suppressed exploration → smaller
  population → **faster** cpu_single → the ratio **understates** Rust's edge on that axis. (Also
  makes exploit2 non-greedy, inflating its steps.) *This is a bias toward cpu_single.*
- **(g) Setup-inclusion — biases against cpu_single.** cpu_single `total_time_s` includes per-phase
  setup the Rust timer excludes (quantified ~0.2–0.6 % via `total` vs `avg_exp×10`). Opposite
  direction to (f); immaterial in magnitude.
- **(h) Timer knob — conservative.** `metric_calculation_frequency` 1 → 50 to match Rust's
  metrics-excluded timer. Verified observational (no RNG, no mutation; trajectory byte-identical at
  freq=1 vs 50 on Maze4 *and* MazeE3). Residual in-timer scan ≈ **2 %** across the population range
  (§4) — **inflates** cpu_single time (conservative).
- **(i) Per-episode time is a CRUDE proxy.** It normalizes episode count only; phase composition,
  GA, and steps-per-episode differ, so it does not cleanly isolate per-episode cost.
- **(j) Steps-to-goal & population are NOT clean comparisons.** Different representation (b), reward
  (c), and non-greedy cpu_single exploit2 (f); different classifier spaces ((row,col) vs 8-sensor)
  and GA settings. Reported as labelled secondary only — never as algorithm-quality.
- **(k) Geometry / goal caveats.**
  - **Woods101_5 & Woods102 are geometry-dominated.** Each source `.cpp` has an all-wall middle row
    splitting the grid into two 8-disconnected halves; the goal is in the top half only. ~50 % of
    start cells (Woods101_5 11/22, Woods102 14/28 path cells) **cannot reach the goal** in the
    source geometry — **both sides** random-start on any path cell, so both truncate the
    unreachable half at the cap. Their high steps (cpu 154/152, Rust 155/142) are **geometry, not
    policy** — do not read as poor learning. Reproduced faithfully on both sides.
  - **MiyazakiA goal at (6,3).** `MiyazakiA.cpp` has PRIZE at (3,6) but `goalState = {6,3}` (row/col
    transposed); the ALCS runtime terminates at **(6,3)**. Both sides run (6,3) — consistent.
  - **MazeF1 & MazeMA excluded.** Their `goalState` is in a wall (degenerate at source);
    cpu_single = 200.000 ± 0.000 across all modes. Not run, not reported.

---

## 8. Measured-artifact integrity

`git -C /Users/aleklyskawa/Desktop/ALCS status` shows **no tracked source modified** — the only
` M` entries are `__pycache__/*.pyc` bytecode caches (auto-regenerated by Python on import; not
source). No `.py`, `.yaml`, or `.cpp` source file was touched. The measurement YAML copy
(`batch_mazes_f50.yaml`) lives in the **Rust** repo (`reports/cpu_single_raw/`), not the ALCS tree
— it is an input pointed at via `--config`, not a modification. Untracked in ALCS: `RECON_CPU_SINGLE.md`
(prior recon) and `__pycache__/` dirs only.

---

## 9. PASS / FAIL

| Step | Status | Evidence |
|---|---|---|
| **0** isolated env + torch-free dispatch | **PASS** | uv venv Py 3.11.15; torch 2.12.0 CPU-only (`cuda=False`); `_is_gpu_mode` routes cpu_single→`no_mp=True` (`universal_runner.py:66`), never GPU. |
| **1** geometry parity (all 22, independent) | **PASS** | Cell-by-cell diff ALCS catalog vs Rust matrices: dims + full wall-set + goal match on all 22 (§2). Rust `unold_geometry.rs` 3 tests green. |
| **2** YAML knob only | **PASS** | `batch_mazes_f50.yaml`: native learning config, `metric_calculation_frequency=50`; mazes/modes are scope only. No source edit. |
| **3** cpu_single run (native, n_exp=10) | **PASS** | `cpu_single_full_f50.csv` — 22/22 rows, no FAIL in log. Maze4/Woods100 reproduce prior run. |
| **4** Rust bench (GA off, n_exp=10) | **PASS** | `bench_rust_unold.csv` — 22/22 rows on `-ounold` geometries. |
| **5** timer symmetry + heavy-maze validation | **PASS** | Residual ≈ 2 % across macro 22→526; freq=1≡freq=50 trajectory on MazeE3 (§4). Conservative (inflates cpu_single). |
| **6** extract + deliver | **PASS** | This doc: per-maze timing + population tables, per-maze & aggregate ratio (26.4×), full ledger, raw CSVs. |
| **7** measured-artifact integrity | **PASS** | ALCS tree clean apart from `.pyc` caches + untracked YAML-copy-elsewhere/outputs (§8). |

**Headline:** across all 22 supervisor mazes, the Rust ACS2 system completes the benchmark a
**median ~27× / aggregate ~26× faster** (total wall-clock, metric-symmetric, on the same M1) than
the native CPython `cpu_single` system — a **cross-representation systems gap, not a language
micro-benchmark**. The per-maze ratio ranges 12.8×–115× and is driven largely by which side's
representation suits each maze (§5).
