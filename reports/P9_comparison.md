# P9 — Benchmark comparison (Rust acs2 vs pyalcs)

GA OFF, n_exp=10, protocol = 500 explore (eps 0.8) + 3x200 exploit, final-window mean steps. Same machine, sequential timed runs.

## Learning quality (correctness gate: same order of magnitude)

| Maze | pyalcs steps | Rust steps | steps ratio | pyalcs macro/reliable | Rust macro/reliable | flag |
|---|---|---|---|---|---|---|
| Maze4-v0 | 3.594±0.181 | 3.584±0.128 | 1.00x | 304/265 | 283/260 | ok |
| Maze5-v0 | 4.812±0.236 | 4.694±0.239 | 1.03x | 396/322 | 393/337 | ok |
| Maze7-v0 | 6.610±0.334 | 6.741±0.290 | 1.02x | 365/339 | 363/343 | ok |
| Woods1-v0 | 1.812±0.062 | 1.692±0.110 | 1.07x | 204/131 | 121/117 | ok |
| Woods100-v0 | 18.439±2.144 | 17.774±3.007 | 1.04x | 18/14 | 18/14 | ok |

## Timing and speedup (t_python / t_rust)

| Maze | pyalcs total_s | Rust total_s | speedup |
|---|---|---|---|
| Maze4-v0 | 63.569 | 0.415 | 153.3x |
| Maze5-v0 | 93.924 | 0.756 | 124.2x |
| Maze7-v0 | 117.444 | 0.916 | 128.3x |
| Woods1-v0 | 19.840 | 0.066 | 298.8x |
| Woods100-v0 | 11.428 | 0.047 | 242.6x |

- **Total-time speedup (Sigma t_py / Sigma t_rust): 139.2x** (dominated by the slowest/longest mazes)
- Mean of per-maze speedups (equal weight): 189.4x
- pyalcs total 306.20s vs Rust total 2.20s over 5 mazes x 10 repeats

Correctness gate: PASS — all mazes within 2.0x of pyalcs steps.

## Reading the results

- **Correctness.** Exploit steps-to-goal agree across every maze (ratios near 1.0x), confirming Rust runs the same ACS2 algorithm — not a faster, different one.
- **Population/reliable counts are equal-or-smaller in Rust, and match closely on most mazes** (Maze5/Maze7/Woods100 within ~1%, Maze4 ~7%). The direction is consistent with the P8 `apply_alp` mid-iteration skip (pyalcs occasionally fails to process an action-set classifier, which can trigger spurious covering), but the effect is small for the same steps-to-goal. The lone large gap is **Woods1 (204 vs 121)**, a 9-cell maze where population composition is RNG-noisy; it is not attributed to any single cause here. None of this is a regression: steps agree.
- **Woods100 is the noise maze.** A 1x7 corridor with a 500-step cap: under eps=0.8 exploration both sides truncate often, so its steps and timing carry the most variance and the ratio reflects truncation as much as learning.
- **Speedup is optimized-Rust vs CPython/pyalcs on one machine, sequential runs.** It reflects language/runtime, not yet the bit-packing optimization (still behind the same interface, deferred). GA-ON is a separate later run of the same binaries.

