# Maze baseline protocols

The main protocol is `maze-pairs-v1`, fixed before learning in commit
`eef356cfe050f5e11502e7a1e9a4a229bda29de3`. `pair-protocol.json` contains the
ordered distinct walkable start-goal pairs, including the original `9` landmark.
`register_pairs.py` reconstructs it using only geometry and exact integer
absorbing-state random-policy calculations. Do not edit the registered pairs
in response to success, knowledge or cost diagnostics.

For each goal rank starts by shortest distance, row and column. Split into three
balanced rank strata, then `max(1, floor((size+2)/5))` balanced segments per
stratum. Put endpoints in validation/test, alternating their order by
 goal-index+stratum-index+segment-index parity, and interiors in training.
Every goal has at least three starts in every block, all three strata and two
distinct distances; every block has at least forty pairs. Actual proportions,
counts, exact floors, ceilings and beyond-cap pairs are explicit in the JSON.
All goals occur in all three blocks; only the start changes within a goal.

Caps remain Maze4=5, Maze5=8, Maze6=10, Maze7=10, Woods14=18, MazeMA=9.
The two six-step Maze4 pairs remain in validation at cap five. No pair is dropped.
Training draws a uniform goal, then a uniform start from that goal's training
starts, identically for both agents. The agent receives all goal identifiers;
held-out pair sets belong to the measurement layer. Training can visit held-out
starts during an episode: this protocol holds out resets, not transitions.
Evaluation is read-only with equal exact pair weights separately in all blocks.

Schema 5 also reports `knowledge` over the complete maze's multi-goal walkable
transitions and every goal, excluding wall bumps and transitions starting at
their own goal. Both transition sides contain the same goal. The core's
`anticipation_fraction` requires quality strictly above `theta_r` and matches
the entire goal condition. `knowledge_transition_goal_pairs` pins its universe.
This corrects the goal-ignoring matching defect in the predecessors' pyalcs
figure, so their values cannot be compared directly. Knowledge runs outside
training timers and match counters, without consuming random streams.
Schemas 2–4 remain readable; full-pool learning fields reproduce the archived
regression fixture. No learner, configuration or checkpoint format changes.

`maze-goals-v1` remains unchanged in `protocol.json`, reproduced by
`register_protocol.py`. Its seeds 42–46 diagnostics through 200k and seed-42
cost pilot through 10M showed weak, inconsistent held-out-goal transfer and
unfiltered relabeling outside training goals. The numerical rationale and its
limits are recorded in [the architecture](../../docs/ARCHITECTURE.md#goal-holdout-diagnostics).
Its twenty-seed proposal is superseded; submission rejects that stage.
`pilot.json` is the historical manifest of the completed goal cost pilot.
The composition example retains its explicit `maze-goals-v1` diagnostic path.

The pair budget rule was fixed before new learning:
`2M * ceil(eight-action graph diameter / 10)` interactions, equal for both
agents. Five mazes use 2M and Woods14 4M. Checkpoints are
1k/3k/10k/30k/100k/300k/1M/2M, plus 4M for Woods14. This horizon does not
assert convergence. `pair-pilot.json` authorizes only a seed-42 cost
recalibration if the distribution/knowledge change materially affects cost.
The predeclared materiality threshold is 25% in training cost per step or
whole-body cost, compared with archived five-seed 200k diagnostics on the same
laptop. The optional pilot reserves twelve single-CPU allocations of fifteen
minutes (3 CPU-hours); the completed goal pilot already spent 4,475 CPU-seconds.

`grid-plan.json` must be a new pair proposal before a grid can be approved.
Its test primary score is the equally weighted checkpoint mean within each
seed; across seeds 42–61 report mean, SE, t(19) intervals and paired ER-minus-ACS2
intervals. Side blocks and knowledge are secondary. The threshold is the first
registered nominal point whose across-seed mean test success reaches half its
ceiling, without interpolation. Retain every configuration, including those
whose final lower intervals do not exceed the floor. If ACS2 already reaches
90% of its ceiling at the first point, omit speed ranking for that maze.

Use the single shared `experiments/grid/` execution path. It verifies clean
pinned source, musl binary/preset, CPU model, exact allocation identity, rows
and artifact hashes. Before each submission check the shared grant and
`sbatch --test-only`, and count pilot plus used, reserved and new hard limits
against 40 CPU-hours. Full-grid submission requires explicit approval of the
exact plan SHA-256. Do not submit a proposal merely because it is committed.
No host pinning, lease, monitor or laptop-dependent supervision is used.

Raw rows, resolved manifests, approval, accounting, local diagnostics and Polish
reports stay outside every checkout. After an approved grid, `analyze.py`
requires complete selected attempts, terminal allocation accounting and the
registered seed-42 cost reference; it validates each block's own distribution.

```bash
acs2-measure --task woods14 --cap 18 --pool full --encoding coordinates \
  --protocol maze-pairs-v1 --agents acs2,acs2er --seeds 42 --targets 1000,10000 \
  --out RESULT_OUTSIDE_CHECKOUT.jsonl
```

```bash
python3 -B -m unittest discover -s experiments/maze-baselines -p 'test_*.py'
python3 -B -m unittest discover -s experiments/grid -p 'test_*.py'
```
