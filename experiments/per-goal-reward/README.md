# Per-goal classifier reward values

The two new arms keep a shared transition model and separate `r` and `ir` for
every live classifier and every maze goal. `acs2er_episodic_per_goal_reward`
(E+1) extends episodic ER (E). `acs2her_shortened_per_goal_reward` (S+1) extends
shortened full-update HER (S) at p=2/3. This changes the reinforcement part of
Butz's ACS2: values do not generalize across goals. GA and restore are rejected.

Creation values follow the base constructor: covering starts `r=0.0` and
`ir=initial_ir`, general classifiers start `initial_r` and `initial_ir`, and ALP
children copy every parent's goal value. A child's update flags start false.
Subsumption retains the existing owner's values; deletion removes its entries.
The shared classifier `r` and `ir` fields do not drive these arms. The agent-owned
value table, including update flags and route counters, participates in read-only
evaluation snapshots. No shared core type or checkpoint format changes.

Every row records cumulative classifier-goal reward writes by original/relabeled
origin, currently existing classifier-goal pairs, and live entries updated at
least once in their own lifetime. The last count can decrease after deletion.
Inheritance alone is not a reward write. Policy scores, next-match bootstrap,
evaluation estimates and first-decision profiles use the decision/draw goal.

`protocol.py` fixes the two arms, six existing pair configurations, seeds 42--61,
three blocks, success contrasts, per-seed interaction and disjoint verdict rules.
`results.py` and `analyze.py` implement the registered analysis. Success uses the
held-pair test block, separately for checkpoint-mean and final success, with
marginal paired t(19) intervals. Mechanism metrics use seeds 42--46, the final
point and equal block mass, with t(4). The mechanism rule jointly requires positive
within-start slope and F_mod intervals on the same three of Maze4--7. All other
comparisons, including any with ACS2 or R, are post hoc.

The authorized cost pilot is seed42, two arms, six mazes, all checkpoints, all
three blocks and per-pair records: twelve single-CPU twenty-minute allocations,
at most four allocated CPU-hours including reservations. `freeze_references.py`
copies E/S from their original archives and G/R from their approved archives,
verifying unchanged source bytes. The older studies are never written.
`prepare_pilot.py` freezes the exact manifest, references, models and operations.
`operations.py` mechanically rederives that manifest before scheduler preview and
submission. The study-local launcher and collector adapter reuse the protected
grid guards, collector, accounting and artifact checks without editing them.
A real failure writes the shared STOP; every allocation stays in the ledger.

```bash
cargo build --release --bin acs2-measure
python3 -B -m unittest discover -s experiments/per-goal-reward -p 'test_*.py'
python3 -B experiments/per-goal-reward/archive.py /absolute/external/study
python3 -B experiments/per-goal-reward/prepare_plan.py /absolute/external/study
```

`prepare_plan.py` writes the single `plan.json` only after all twelve pilot cells
are audited. Its forecast keeps each arm-maze cost and counts array padding in
the hard reservation total. Stage1 ends with that committed plan and the external
report. Seeds43--61, final contrasts, training curves and the architecture results
section require separate approval of the exact plan bytes and a CPU-hour limit.
Stage2 was approved on 2026-11-08 with a cumulative 29 CPU-hour limit after the
strict upper-zero-boundary test remark was closed. Raw results live in
`~/tu_runs/per-goal-reward/` on the cluster and `~/Desktop/tu-runs/per-goal-reward/`
locally, with checksum verification of both copies.

The seed42 pilot completed twelve attempts without failures: 98 rows and 83,532
pair records, at 1,648 allocated CPU-seconds (0.457778 CPU-hours). The
[single registered plan](plan.json) forecast 8.697778 CPU-hours for 228 remaining
runs and 9.155556 including the pilot, using one cost seed. Full array reservations
would total 28.657778 CPU-hours including the pilot; the approved cumulative
limit is 29 CPU-hours.

The full grid completed all 240 runs on their first allocations: 1,960 rows and
417,660 new-arm pair records, including the unchanged pilot. JobIDRaw-deduplicated
accounting totals 33,384 allocated CPU-seconds (9.273333 CPU-hours), with no failures,
unmapped allocations or remaining reservations. A separate job-name accounting
query reconciles every study allocation with its scoped ledger. All 600 reference
index entries match 540 original archive files, which were not rerun or written.

Both new arms improve over their bases on all six mazes, separately for primary
and final success (A+=6, A-=0); S+1 also improves over G on all six in both measures.
The primary interaction is mixed (A+=1, A-=3). The final interaction gives
"copies gain less from per-goal values" (A+=0, A-=5). Both registered joint
mechanism rules are met on all four of Maze4--7. The
[architecture results](../../docs/ARCHITECTURE.md#per-goal-reward-values-with-a-shared-transition-model)
give all 120 registered intervals, definitions, provenance and limitations.

`approval.json` pins the approved plan and the unchanged pilot binary.
`prepare_grid.py` preserves a checksum inventory of every pilot input and artifact,
then freezes separate execution operations. `dispatch.py` derives each of the eight
forecast arrays from the approved plan before scheduler preview and submission.
Each arm has one stream: its 24-job pair array precedes its three 30-job compact
arrays. At most two arrays are active, each with twenty running slots. The shared
accounting scope includes the immutable pilot and every new allocation.

```bash
python3 -B experiments/per-goal-reward/prepare_grid.py --root STUDY_ROOT --destination CLUSTER_STUDY_ROOT
python3 -B experiments/per-goal-reward/dispatch.py --root CLUSTER_STUDY_ROOT --repo TU_CLONE --array 0
```

The forecast indices are 0--3 for episodic replay and 4--7 for shortened HER.
Submit the next array of an arm only after its predecessor is terminal and all
registered rows validate. The dispatcher refuses an existing batch, a changed
measurement source or binary, shared STOP, exhausted grant/budget or a third
active stream. No pilot job is repeated and no measurement binary is rebuilt.

`verify_grid.py` reconciles every ledger with terminal JobIDRaw-deduplicated
accounting, verifies the preserved pilot and both frozen operation snapshots,
checks complete registered rows and validates pair aggregates against capped
environment values. Analysis then reads only checksum-indexed verified rows.
Training curves are explicitly post hoc: pooled F and F_mod over seeds42--46,
with marginal t(4) intervals at each checkpoint, without curve-level inference.
The archived `analysis/output-scope.json` labels ancillary shared-exporter columns
in `mechanism-seed-metrics` as post hoc; only the six registered mechanism metrics
enter registered contrasts and decisions. Full tables are also beside the external
Polish report. The figures contain six maze panels and a six-maze overview, each
in PNG and SVG. Every archive file has two independently SHA-256-verified copies.

```bash
python3 -B experiments/per-goal-reward/verify_grid.py STUDY_ROOT
python3 -B experiments/per-goal-reward/analyze.py --root STUDY_ROOT --verification STUDY_ROOT/verification.json --out STUDY_ROOT/analysis
uv run --project tools python -B experiments/per-goal-reward/plot_results.py STUDY_ROOT/analysis
```
