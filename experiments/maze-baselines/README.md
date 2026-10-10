# Maze baselines with held-out goals

`protocol.json` fixes the six geometries, caps and disjoint goal pools before any
learning under this protocol. `register_protocol.py` reconstructs it using only the
reference graph and the exact integer random-policy calculator. The reference is
unmodified pyalcs commit `79d9f862e35385ba6fbad3a75415c96b9b116500`.

Caps are Maze4=5, Maze6=10 and Maze7=10 from the pilot. New caps use the smallest
horizon covering all distinct start-goal pairs: graph diameter, giving Maze5=8,
Woods14=18 and MazeMA=9. Every block's exact random floor is below 0.20. Maze4 has
two legacy exceptions, `(1,1)->(6,6)` and the reverse, each requiring six steps.
The registered blocks list them explicitly; no pair is silently dropped.

Goal difficulty is the sum of shortest distances from all other walkable starts.
Integer ranking avoids rounding ties. Rank by this sum, row and column; divide
into three balanced contiguous strata, giving remainders to earlier strata.
Within each stratum divide into `max(1, floor((size+2)/5))` balanced segments.
The first and last goals enter validation/test, with their order alternating by
stratum plus segment parity. Interior goals enter training. Sort each resulting
pool by row and column. No randomness or learning outcomes enter the selection.

A goal split is adequate if every block has at least three goals, forty distinct
start-goal evaluation pairs and all three difficulty strata. All six mazes pass,
including Woods14 (three held-out goals, 54 pairs per block) and MazeMA (three,
42 pairs). Consequently no pair split is used. These small sets support a
within-maze generalization check; they do not estimate variation across mazes.

Training draws a goal uniformly from the training pool and a start uniformly from
every other walkable cell, including validation/test cells and the original `9`
landmark. Evaluation uses every start times every goal of each named block,
excluding equality, with exactly `1/(goals*(cells-1))` weight. The agent receives
only the training pool. Held-out goals appear solely in immutable evaluation
action calls. Test diagnostics may be reported but must not influence tuning,
sample selection, utility or the HER/TU mixture.

The twenty-seed grid requires a separately approved plan. Stage-one checks and
the one-seed cost pilot are task diagnostics, not final thesis results. Raw data,
approval, registration commit, accounting and Polish reports stay outside Git.

```bash
python3 -B -m unittest discover -s experiments/maze-baselines -p 'test_*.py'
baseline/.venv/bin/python -B -m unittest baseline/test_random_policy.py
```
