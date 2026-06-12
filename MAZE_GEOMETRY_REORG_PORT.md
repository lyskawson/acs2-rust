# Maze geometry reorg + supervisor (ounold/ALCS) port

Two-part change to `acs2-envs` **geometry data/organization only** — no change to `acs2-core`
learning semantics or the `acs2-bench` timed region. ALCS stayed read-only (source of the
`.cpp` geometries).

## Part 1 — Reorganization (one-file-per-maze)

`maze_data.rs` (single const blob) → a `mazes/` module tree, one file per maze, with a typed
`source` field so a result's provenance is never ambiguous:

```
acs2-envs/src/mazes/
  mod.rs                       # MazeGeometry, MazeSource{Pyalcs,OunoldAlcs}, geometry_by_id, registries
  canonical/{maze4,maze5,maze7,woods1,woods100}.rs   # source = Pyalcs, data byte-identical
  unold/{22 mazes}.rs + mod.rs                        # source = OunoldAlcs, cap 200
acs2-envs/src/maze_data.rs     # back-compat re-export shim (MazeGeometry, geometry_by_id, MAZE_GEOMETRIES, …)
```

- `MazeGeometry` gained `source: MazeSource`; the five canonical literals are otherwise verbatim
  (matrix / goal-as-`9` / `max_episode_steps` unchanged).
- `MAZE_GEOMETRIES` still = the five canonical mazes (so the P9 default run is unchanged);
  `geometry_by_id` searches canonical **then** unold. Consumer API (`acs2-bench`, existing tests)
  unchanged via the shim — compiles without edits to those crates.

### RE-VALIDATION GATE — learning metrics byte-identical (deterministic, seed=42)

Captured a golden bench run **before** the reorg, then re-ran **after**. Diff of all learning
columns (steps mean/std, macro, micro/numerosity, reliable) — only wall-clock allowed to drift:

| maze | steps | macro | micro | reliable | golden == post-reorg |
|---|---|---|---|---|---|
| Maze4-v0 | 3.5845±0.1275 | 283.40 | 283.40 | 260.10 | ✓ identical |
| Maze5-v0 | 4.6940±0.2388 | 392.80 | 392.80 | 336.70 | ✓ identical |
| Maze7-v0 | 6.7410±0.2905 | 362.90 | 362.90 | 342.70 | ✓ identical |
| Woods1-v0 | 1.6925±0.1102 | 121.10 | 121.10 | 117.10 | ✓ identical |
| Woods100-v0 | 17.7745±3.0072 | 18.00 | 18.00 | 14.00 | ✓ identical |

`diff` over fields 1–7 of `reports/golden_pre_reorg.csv` vs `reports/revalidation_final.csv` is
**empty** (and these match the recorded `reports/P9_comparison.md` numbers). The reorg did not
alter semantics. **GATE: PASS.**

## Part 2 — Port of the ounold/ALCS `.cpp` geometries

Source: `ALCS/environment/acs2_mazes/*.cpp` (read-only). Conversion per maze: parse the
`OBSTACLE/CORRIDOR/PRIZE` matrix (→ `1/0`, PRIZE→`0`) **and** the separate `goalState = {row,col}`,
then place `9` at `goalState` — **the goal is `goalState`, not the PRIZE marker**, because the
ALCS runtime uses `goalState` literally: `maze_loader.py:67,93` (`goal_pos = goalState`) →
`runtime_cpu3.py:137` (`done = pos == goal_pos`); the PRIZE token maps to a passable corridor and
is **never read as the goal**. `max_episode_steps = 200` (his flat cap). Kept in the `unold`
namespace with ids suffixed `-ounold`, so e.g. `Maze4-ounold` never shadows canonical `Maze4-v0`.

### Outcome: 22 ported, 2 held (all holds derived, not hardcoded)

Hold logic is computed from the parsed geometry: `goalState` on a wall or isolated (BFS,
8-connectivity = his 8 actions) → hold. Multi-PRIZE mazes are **ported** single-goal at
`goalState` (the runtime ignores the second prize), with the second prize → corridor.

| maze | decision | dims (h×w) | goalState | cell | goal==PRIZE | walls | note |
|---|---|---|---|---|---|---|---|
| Cassandra4x4 | PORT | 6×6 | (4,4) | path | yes | 20 | |
| Littman57 | PORT | 4×13 | (2,9) | path | yes | 37 | |
| Littman89 | PORT | 7×9 | (4,7) | path | yes | 40 | |
| Maze10 | PORT | 6×9 | (4,3) | path | yes | 35 | |
| Maze4 | PORT | 8×8 | (1,6) | path | yes | 37 | coincides w/ canonical Maze4 |
| Maze7 | PORT | 7×5 | (5,1) | path | yes | 25 | **differs** from canonical Maze7 (9×9) |
| MazeA | PORT | 8×8 | (3,6) | path | yes | 41 | |
| MazeB | PORT | 8×8 | (2,6) | path | yes | 38 | |
| MazeD | PORT | 8×8 | (4,5) | path | yes | 39 | |
| MazeE1 | PORT | 9×9 | (4,4) | path | yes | 36 | |
| MazeE2 | PORT | 9×9 | (4,4) | path | yes | 32 | |
| MazeE3 | PORT | 11×11 | (5,5) | path | yes | 40 | |
| MazeF2 | PORT | 6×5 | (1,3) | path | yes | 23 | |
| MazeF3 | PORT | 6×6 | (1,4) | path | yes | 27 | |
| MazeF4 | PORT | 6×7 | (1,5) | path | yes | 31 | |
| MiyazakiA | PORT | 8×8 | (6,3) | path | **NO** | 34 | **goal≠PRIZE — see ledger** |
| MiyazakiB | PORT | 8×8 | (3,6) | path | yes | 36 | |
| Woods1 | PORT | 7×7 | (3,3) | path | yes | 32 | **differs** from canonical Woods1 (5×5) |
| Woods100 | PORT | 3×9 | (1,4) | path | yes | 20 | coincides w/ canonical Woods100 |
| Woods101 | PORT | 5×7 | (3,3) | path | yes | 24 | |
| Woods101_5 | PORT | 11×7 | (1,3) | path | yes | 55 | source has 2 PRIZE; single-goal@goalState; **50% partial-unreachable — ledger** |
| Woods102 | PORT | 11×7 | (1,3) | path | yes | 49 | source has 2 PRIZE; single-goal@goalState; **50% partial-unreachable — ledger** |
| **MazeF1** | **HOLD** | 6×4 | (1,3) | **wall** | no | 18 | goalState in a wall → unreachable |
| **MazeMA** | **HOLD** | 7×11 | (1,7) | **wall** | no | 62 | goalState in a wall → unreachable |

For every ported maze, parsed dimensions == `.cpp` `mazeHeight×mazeWidth`, and `goalState` is a
non-wall, reachable cell. `on_prize` is recorded for transparency (it is **not** a port condition
— the runtime goal is `goalState`). For the two multi-PRIZE mazes, `9` is placed at `goalState`
only and the second PRIZE cell becomes a passable corridor — exactly what the ALCS runtime does.

### ANOMALY LEDGER (factual — for the supervisor)

Source files: `ALCS/environment/acs2_mazes/<Name>.cpp`. Runtime goal logic:
`environment/maze_loader.py:46-50,67,93` (`goal_pos = goalState`) and
`environment/runtime_cpu3.py:137` (`done = tuple(final_pos) == self.goal_pos`).

1. **MiyazakiA — `goalState` ≠ PRIZE marker (PORTED at goalState).** `MiyazakiA.cpp`: PRIZE at
   (3,6), but `.goalState = {6,3}` (row/col transposed). The runtime terminates at **(6,3)**, a
   reachable corridor; cpu_single MiyazakiA terminates at 23.14 steps. Ported at **(6,3)** to
   match the runtime. If (3,6) was intended, edit `acs2-envs/src/mazes/unold/miyazakia.rs` (one
   line).
2. **MazeF1 — `goalState` (1,3) is a wall (HELD).** `MazeF1.cpp`: PRIZE at (1,2), but
   `.goalState = {1,3}`, which is an `OBSTACLE` cell. The runtime goal is therefore unreachable;
   no episode terminates. Measured: cpu_single MazeF1 exploit steps = **200.0000 ± 0.0000** (cap).
   Per the supplement Table 1, MazeF1 = 200.000 across all modes. Not portable to the
   `9`-in-matrix format without flipping wall→reachable-goal (would falsify the geometry).
3. **MazeMA — `goalState` (1,7) is a wall (HELD).** `MazeMA.cpp`: PRIZE at (1,9), but
   `.goalState = {1,7}` is an `OBSTACLE` cell. Same as MazeF1: cpu_single = **200.0000 ± 0.0000**;
   Table 1 = 200.000 across all modes. Held.
4. **Woods101_5 / Woods102 — 2 PRIZE cells + disconnected halves (PORTED single-goal).** Each
   `.cpp` has two PRIZE cells (goalState top, second prize bottom) and an all-`OBSTACLE` middle
   row that splits the grid into two 8-disconnected components. The runtime terminates only on
   `goalState` (top); the second prize is an ordinary corridor. Ported `9` at `goalState` only.
   Consequence (faithful to source): BFS shows only **50%** of path cells can reach the goal
   (Woods101_5 11/22, Woods102 14/28); since both ALCS and the Rust env random-start on any path
   cell, the bottom-half starts cannot reach the goal and truncate at the cap. Measured:
   cpu_single 152.08 / 170.40 steps; Rust port 106.7 / 110.7 — both consistent with ~half the
   episodes hitting the cap. This is a property of the source geometry, reproduced exactly.

> Maze5 has no `.cpp` in ALCS, so there is no `Maze5-ounold` (matches the earlier cpu_single recon).

### Fidelity tests (`acs2-envs/tests/unold_geometry.rs`, 3 tests, all green)

- `unold_geometries_are_well_formed`: 22 geometries, each `source=OunoldAlcs`, cap 200,
  rectangular, exactly one goal, closed wall border.
- `unold_goal_is_reachable`: BFS (8-conn) from the goal reaches >1 cell for all 22.
- `unold_transitions_match_geometry`: **exhaustive** over every path cell × 8 actions per maze —
  walls block (position unchanged, no termination), open cells move, stepping onto the goal
  terminates with reward 1000. This is the geometry/transition fidelity check (not perception
  parity — perception differs by design: ALCS (row,col) coords vs Rust 8-sensor).

## Files

- **Modified:** `acs2-envs/src/lib.rs`, `acs2-envs/src/maze_data.rs` (→ shim).
- **Added:** `acs2-envs/src/mazes/` (mod.rs; `canonical/` 5 mazes + mod; `unold/` 22 mazes + mod),
  `acs2-envs/tests/unold_geometry.rs`.
- **Evidence (reports/):** `golden_pre_reorg.csv`, `revalidation_final.csv` (parity),
  `unold_smoke.csv` (ported mazes run end-to-end). Generator: `/tmp/gen_unold_mazes.py` (scratch,
  not committed; geometries committed as plain `const` data — no runtime `.cpp` dependency).

## PASS / FAIL

| Item | Status | Evidence |
|---|---|---|
| Part 1 reorg (one-file-per-maze, source field, shim) | **PASS** | builds; consumer API unchanged |
| Re-validation gate (learning metrics byte-identical) | **PASS** | empty diff golden vs post-reorg/final on fields 1–7; matches P9 |
| Part 2 port (22 mazes, separate namespace, cap 200) | **PASS** | 22 `unold/*.rs`, ids `-ounold`, run end-to-end |
| Goal conversion (`9` at goalState, runtime-faithful) | **PASS** | maze_loader/runtime cited; multi-PRIZE → single-goal@goalState |
| Holds derived + justified (2) | **PASS** | BFS reachability; MazeF1/MazeMA wall-goal degeneracy empirically = 200.0 |
| Fidelity tests | **PASS** | 3/3 green; exhaustive transition check over all 22 |
| acs2-core / acs2-bench untouched | **PASS** | only acs2-envs changed |
| Anomaly ledger consolidated | **PASS** | 4 items (MiyazakiA, MazeF1, MazeMA, Woods101_5/102), factual + file:line |

**Next phase (NOT done here):** the cpu_single comparison on the ported geometries.
