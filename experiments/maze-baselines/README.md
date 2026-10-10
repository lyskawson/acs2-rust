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

`pilot.json` authorizes only the seed-42 cost pilot: both agents, six mazes, through
ten million interactions. `grid/submit.py` resolves source, binary and preset
identity in an external result root. Full-grid manifests require a matching
explicit approval; merely committing a proposal does not authorize submission.

`analyze.py` reads approved schema-4 runs. Its primary block is test: each seed's
equally weighted mean success over the registered checkpoints, paired ER minus
ACS2 differences and Student t(19) intervals across twenty seeds. Training and
validation receive the same summaries alongside it. The original pilot rejection
criterion applies to test: retain floor-level baselines for successors; refuse
speed ranking if ACS2 already achieves 90% of its ceiling at the first checkpoint.
Threshold time is the first registered nominal point whose across-seed mean
success reaches the plan's fraction of the ceiling, without interpolation. A
missing crossing remains null. All configurations remain in the report.

The diagnostic composition example accepts `--protocol maze-goals-v1`. It uses
only the registered training goal set as its fixed candidate set, including goals
not yet seen as desired goals. UniformReal and candidate-filtered strategies
select within that set; unfiltered achieved-goal strategies may select elsewhere.
Raw achieved cells can coincide with held-out goal cells. This does not hand the
agent a held-out desired-goal distribution or authorize utility on that set.

```bash
acs2-measure --task woods14 --cap 18 --pool full --encoding coordinates \
  --protocol maze-goals-v1 --agents acs2,acs2er --seeds 42 --targets 1000,10000 \
  --out RESULT_OUTSIDE_CHECKOUT.jsonl
```

```bash
python3 -B -m unittest discover -s experiments/maze-baselines -p 'test_*.py'
baseline/.venv/bin/python -B -m unittest baseline/test_random_policy.py
```
