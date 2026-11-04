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

The cost pilot authorized six seed42 runs with per-pair records, up to two CPU-hours
including reservations. [approval.json](approval.json) now authorizes the registered
continuation within fifteen cumulative CPU-hours, including the pilot and every attempt. The shared grid submission path re-derives
the exact pilot manifest, verifies a clean pinned musl build and frozen operations,
checks grant and queue reservations, and invokes scheduler test-only before submission.
The pilot uses one six-job array with twenty-minute hard limits. Node guards and a
shared STOP retain real failures. The continuation uses the same pinned measurement commit and static binary. Two
streams share one accounting scope containing the immutable pilot; no pilot is repeated.

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

Stage 2 runs the remaining nineteen seeds, computes the registered comparisons,
plots F and F_mod over training against both references, and documents the results in
`docs/ARCHITECTURE.md`. No ranking follows from the cost pilot.

## Cost pilot and registered continuation

The six seed42 allocations completed without failure: 49 rows and 41,766 pair
records passed the row, reward-counter, aggregate and capped-value checks. The
allocated cost was 865 CPU-seconds (0.240278 CPU-hours), with zero reservations,
inside the two-hour pilot limit. Measurement source is `1c36582`; frozen operations
are `b78e5e3`. The binary identity is in [measurement.json](measurement.json).

The single [registered plan](plan.json) has SHA-256
`cb38ee22abc0a73c2411adc0a2e3ba921c36b45290353359bbe9ea95d6e57449`.
Its expected cumulative cost is 4.805556 CPU-hours; doubling the remaining measured
cost gives 9.370833, while full hard reservations plus the pilot give 14.340278.
The proposed cumulative approval is therefore 15 CPU-hours. This is a reservation
bound, not a prediction of actual CPU use. One cost seed does not bound other seeds.
The remaining 114 runs use the explicit stage2 approval in `approval.json`.

```bash
python3 -B experiments/goal-rule-copies/archive.py <complete-pilot-root>
python3 -B experiments/goal-rule-copies/prepare_plan.py <verified-pilot-root>
python3 -B experiments/goal-rule-copies/analyze.py --root <complete-approved-study> --verification <independent-verification.json> --out <new-external-analysis-directory>
```

`prepare_plan.py` refuses to overwrite the existing registration. The analysis
requires the exact approved plan and a complete independently verified run index;
it cannot infer registered twenty-seed outcomes from the pilot.

## Approved execution and verification

The three required review remarks were closed by test-only commit `b83222b`:
success blocks and both reference contrasts are distinct, origin tests preserve
all other count identities, and both evaluation contracts include the new arm.
The measurement commit, binary, registered plan and pilot remain unchanged.

`prepare_execution.py` preserves the original 668-file inventory, snapshots clean
operations, records the exact approval and creates `continuation/scope.json`.
The shared dispatcher derives one 24-job diagnostic array and three 30-job compact
arrays from the approved plan before submission. Each stream admits at most twenty
running jobs, with at most two streams. Every admission checks the shared grant,
expanded queue, scheduler test-only and cumulative spent plus reserved CPU-seconds.
Original pilot files and its admission record are immutable; new events live under
`continuation/`. No laptop controller or background process is needed.

```bash
python3 -B experiments/goal-rule-copies/prepare_execution.py --root <verified-study> --destination <absolute-cluster-study>
python3 -B <execution-operations>/experiments/goal-rule-copies/execute_batch.py --root <cluster-study> --repo <pinned-TU-clone> --purpose diagnostics
python3 -B <execution-operations>/experiments/goal-rule-copies/execute_batch.py --root <cluster-study> --repo <pinned-TU-clone> --purpose compact
python3 -B <execution-operations>/experiments/goal-rule-copies/verify.py <cluster-study> --update-accounting
python3 -B experiments/goal-rule-copies/verify.py <complete-local-study>
python3 -B experiments/goal-rule-copies/analyze.py --root <complete-local-study> --verification <study>/verification.json --out <new-analysis-directory>
python3 -B experiments/goal-rule-copies/diagnostics.py --root <complete-local-study> --verification <study>/verification.json --out <analysis-directory>
uv run --project tools python -B experiments/goal-rule-copies/plot_results.py <analysis-directory>
```

Call the compact dispatcher again only after its preceding array is terminal.
The verifier checks all attempts, JobIDRaw-deduplicated allocation accounting,
exact assignments and source identities, every row and reward counter, per-pair
aggregates and capped values, protected pilot files, immutable operations and
admission receipts. The plotting export retains all five seed slots and labels
curves and marginal t(4) bands as post hoc. PNG and SVG figures show F and F_mod
for all three arms on each maze, plus one six-maze overview.

Value-dose exports are post hoc descriptions of the recorded macroclassifier
opportunities in the new arm. The retained-blind share divides original blind
updates by original blind updates plus skipped relabeled blind opportunities.
This denominator uses the observed trajectory and post-ALP action sets; it does
not reconstruct the value-update count of either historical reference agent.
