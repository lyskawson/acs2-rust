# Goal-specific reward updates for hindsight copies

Stage 1 implements `acs2her_shortened_goal_rules`, a fixed p=2/3 shortened HER
preset. Only actual relabeled draws filter reward updates. ALP runs first on the
whole action set; the next match set is rebuilt and its maximum change-anticipating
fitness supplies the unchanged bootstrap. Both r and ir are then updated only for
action-set classifiers with at least one specified goal attribute. GA receives the
whole post-ALP action set. Original draws, including original-goal fallback, update
all action-set classifiers. The thesis preset keeps GA disabled, as in the archives.

The variant lives in `acs2-agents/src/acs2her/goal_rules.rs`; existing presets retain
their original learning dispatch. `reward_updates` in every new-arm row records
original and relabeled routes separately: draws, executed updates with and without
goal attributes, skipped blind-rule updates, draws whose post-ALP action set contains
a goal-specific rule, and the number of post-ALP action-set macroclassifiers. These
are cumulative counts, not numerosity-weighted counts. Executed plus skipped updates
partition action-set rules, while route draws partition replay updates. No historical
reference counter is reconstructed as an observed count.

`protocol.py` pins the archived geometry, budgets, checkpoints and three blocks.
`results.py` implements the registered paired t(19)/t(4) contrasts and outcome rules.
`experiments/per-pair-diagnostics/decision_metrics.py` supplies within-start sensitivity,
modal decomposition, overestimation above V*, and the tie-required component of D.
A zero denominator remains null. Uniform-pair sensitivity first pools all blocks,
then assigns every ordered pair equal mass. All supplementary views are post hoc.

The only authorized allocations are six seed42 cost-pilot runs with per-pair records,
up to two CPU-hours including reservations. The shared grid submission path re-derives
the exact pilot manifest, verifies a clean pinned musl build and frozen operations,
checks grant and queue reservations, and invokes scheduler test-only before submission.
The pilot uses one six-job array with twenty-minute hard limits. Node guards and a
shared STOP retain real failures. Seeds43--61 require separate approval of the single
registered plan and a cumulative CPU-hour limit. Stage 1 does not submit them.

References are reused from their original archives: shortened HER from the
hindsight-admissibility study, episodic ER from the HER study, and per-pair records
for seeds42--46 from the per-pair study. Raw data, accounting, checksums, full tables
and the Polish executor report stay outside the checkout under
`~/Desktop/tu-runs/goal-rule-copies/`, mirrored on WCSS at `~/tu_runs/goal-rule-copies/`.

```bash
python3 -B -m unittest discover -s experiments/goal-rule-copies -p 'test_*.py'
python3 -B experiments/goal-rule-copies/prepare_pilot.py --out <new-local-root> --destination <absolute-cluster-root> --references <frozen-reference-directory>
python3 -B <frozen-operations>/experiments/grid/submit.py --root <cluster-root>/pilot --manifest <cluster-root>/pilot-manifest.json --repo <pinned-TU-clone>
```

Stage 2 will run the remaining nineteen seeds, compute the registered comparisons,
plot F and F_mod over training against both references, and document the results in
`docs/ARCHITECTURE.md`. No ranking follows from the cost pilot.

## Cost pilot and registered continuation

The six seed42 allocations completed without failure: 49 rows and 41,766 pair
records passed the row, reward-counter, aggregate and capped-value checks. The
allocated cost was 865 CPU-seconds (0.240278 CPU-hours), with zero reservations,
inside the two-hour pilot limit. Measurement source is `1c36582`; frozen operations
are `b78e5e3`. The binary identity is in [measurement.json](measurement.json).

The single [registered plan](plan.json) has SHA-256
`24b71afbdccf67afd46bab36304a0f319933753f0c3a11b7b6393f2ef7f46330`.
Its expected cumulative cost is 4.805556 CPU-hours; doubling the remaining measured
cost gives 9.370833, while full hard reservations plus the pilot give 14.340278.
The proposed cumulative approval is therefore 15 CPU-hours. This is a reservation
bound, not a prediction of actual CPU use. One cost seed does not bound other seeds.
The remaining 114 runs are not authorized by stage 1.

```bash
python3 -B experiments/goal-rule-copies/archive.py <complete-pilot-root>
python3 -B experiments/goal-rule-copies/prepare_plan.py <verified-pilot-root>
python3 -B experiments/goal-rule-copies/analyze.py --root <complete-approved-study> --verification <independent-verification.json> --out <new-external-analysis-directory>
```

`prepare_plan.py` refuses to overwrite the existing registration. The analysis
requires the exact approved plan and a complete independently verified run index;
it cannot infer registered twenty-seed outcomes from the pilot.
