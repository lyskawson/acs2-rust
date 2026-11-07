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
No stage2 submission is currently authorized. Raw results live in
`~/tu_runs/per-goal-reward/` on the cluster and `~/Desktop/tu-runs/per-goal-reward/`
locally, with checksum verification of both copies.
