# Experiment index

These directories contain registered protocols and analysis tools for the thesis line.
Raw rows, manifests, launch/accounting records, failed attempts, checksums and analysis
outputs live outside the repository. Learning runs use an authorized TU cluster clone;
analysis of archived rows can run locally. The multiplexer experiment/archive tooling
remains in `acs2-bench`, `slurm/`, `tools/` and `reports/`; its live state is
[`docs/AGENT_HANDOFF.md`](../docs/AGENT_HANDOFF.md).

| Directory | Question | Measurement | Result and data provenance |
|---|---|---|---|
| [`baseline-pilot/`](baseline-pilot/) | How do ACS2 and uniform-replay ACS2ER compare with the random-policy floor and with each other on ten sparse goal configurations at equal interactions? | Pilot seed 42, followed by seeds 42–61 for both agents: 400 selected grid runs and 3,800 rows. Five maze, three HandEye, one Taxi and one BitFlipping configuration; budgets and checkpoints below. | [Pilot grid: ACS2 and ACS2ER on ten goal configurations](../docs/ARCHITECTURE.md#pilot-grid-acs2-and-acs2er-on-ten-goal-configurations). Archived learning data came from clean measurement commit `af3ce49b2df0f5a764efb49be9c48da54ebe0040`; operations were versioned separately. |
| [`relabeling-composition/`](relabeling-composition/) | What goals, endings, fallback routes and objective-call costs does relabeling produce on random-policy history before a replay agent learns? | The same ten configurations, seeds 42–61, uniform actions, no learning. Finish the first episode meeting 20,000 steps per run: 200 runs, 4,003,305 steps, 4,800 rows and 240 strategy/rule/filter summary cells. | [Random-policy composition before replay agents](../docs/ARCHITECTURE.md#random-policy-composition-before-replay-agents), within [Trajectory storage and relabeling](../docs/ARCHITECTURE.md#trajectory-storage-and-relabeling). Initial schema-1 data: `a86b7db2d80f04a9aac0693bd871b7ca8d8f6785`; current schema-2 fix-round archive: `cd950fc805e6f99f86fbfe5dceb1e735955ec4f7`. |

## Baseline pilot

Under the supervisor's train/validation/test goal protocol, the maze results are a
pilot rather than the final thesis comparison. HandEye, Taxi and BitFlipping results
describe limitations outside the main discrete multi-goal maze comparison. The grid
preserves the registered thesis preset, goal pools and equal budgets for both agents.

| Configuration IDs | Goal pool | Episode cap | Maximum interactions per agent/seed |
|---|---|---:|---:|
| `maze4_c5_full`, `maze4_c5_p4` | Full / four registered cells | 5 | 10,000,000 |
| `maze6_c10_full`, `maze6_c10_p4` | Full / four registered cells | 10 | 10,000,000 |
| `maze7_c10_full` | Full | 10 | 10,000,000 |
| `handeye4_c50_full`, `handeye4_c50_p4` | Full / four registered block positions | 50 | 10,000,000 |
| `handeye5_c50_full` | Full | 50 | 2,000,000 |
| `taxi_c200_full` | All four delivery goals | 200 | 10,000,000 |
| `bitflip8_c8_full` | All binary eight-bit goals | 8 | 1,000,000 |

Checkpoints are 1k, 3k, 10k, 30k, 100k, 300k, 1M, 2M, 5M and 10M, truncated
at each configuration's budget. Evaluation occurs at the first completed episode
meeting a checkpoint; rows preserve nominal and actual interactions. The primary
score is the equally weighted mean checkpoint success within each seed, then the
mean and SE across seeds; paired agent differences use the registered t(19) analysis.

[`baseline-pilot/pilot.json`](baseline-pilot/pilot.json) is immutable (SHA-256
`8b0a4badd4b2e65025e0e37536e5f45b1323abfb8fe83aab7a43ae9dbae4e77e`).
The approved grid plan, retained beside the archived raw data, has SHA-256
`862f77785f1cf73fc58e81a88fabeac4b708ee908082f227a499d5cb2e4efb98`.
The registered restricted-pool salt `acs2-tu-phase3b-pool-v1` keeps its historical
label because changing its bytes changes the selected pools. Registered values,
row formats, field names and schema contracts are not renamed with directories.

For archived analysis, set `GRID` to the archived grid root (currently
`~/Desktop/tu-runs/3b/grid-v1/`) and `PILOT` to the pilot batch root (currently
`~/Desktop/tu-runs/3b/20260930-pilot-v1/`). From the repository root, for each
configuration ID in the table:

```bash
python3 -B experiments/baseline-pilot/analyze_grid.py \
    --root "$GRID" --pilot "$PILOT" --configuration maze4_c5_full
python3 -B -m unittest discover -s experiments/baseline-pilot -p 'test_*.py'
```

The analyzer validates registered assignments, complete attempts, raw checksums,
references and seed-42 agreement with the pilot. It writes beneath the supplied grid
root's `analysis` directory, outside a checkout. Use an independent copy of the grid
root when comparing reproductions; it requires `approved-plan.json`, `ledger.json`,
`batches/` and `attempts/`, with the attempt directories inside that copied root.
[`baseline-pilot/README.md`](baseline-pilot/README.md) documents pinned binary builds,
pilot submission, approved grid execution, accounting and recovery. Re-running the
learning grid is a new experiment scope and requires authorization; reproducing its
analysis does not.

The pilot tooling is preserved as the record of how this grid ran. The next experiment
copies it once into a shared grid directory; this directory is not refactored into
that future shared infrastructure.

## Relabeling composition

The probe measures final, future, episode and uniform-real goal strategies, three
admissibility rules and both candidate-filter settings. Relabel request proportion is
one, with original-goal fallback for an empty admissible distribution. Episode results
are step-weighted, then per-seed quantities are averaged equally. Route-conditional
shares and scoring/selection/provenance costs are reported separately. It uses the
same tasks, caps, goal pools, preset identity and random streams as the registered grid.

Set `COMPOSITION_RAW` to the archived fix-round `composition-raw.jsonl` under
`~/Desktop/tu-runs/4/` and `COMPOSITION_OUT` to a new directory outside every Git
checkout. Reproduce all three summary files without collecting new interactions:

```bash
python3 -B experiments/relabeling-composition/summarize.py \
    "$COMPOSITION_RAW" --out "$COMPOSITION_OUT"
python3 -B -m unittest discover -s experiments/relabeling-composition -p 'test_*.py'
```

The archived fix-round raw file has SHA-256
`61f1e416ae3bc4e78e27899ec30a2d28348230921010b28ae6bfc112d639951d`.
The summary validates schema 2, complete seed/configuration sets, route identities and
objective-call totals, producing `summary.json`, `summary.csv` and
`composition-report.md`. Schema-1 data remains archived and is not accepted by this
summary version. Output wording and data field names remain part of the preserved
analysis output, including historical labels in generated reports.

For a separately authorized new collection, the measurement example is
[`acs2-measure/examples/trajectory_composition.rs`](../acs2-measure/examples/trajectory_composition.rs):

```bash
cargo run --release -p acs2-measure --example trajectory_composition -- --out "$COMPOSITION_RAW"
```

Build provenance records the commit used, so a new run names its own source rather than
the archived data's commit. The full storage/relabeling semantics, exact composition,
limitations and acceptance evidence are in the linked architecture sections.
