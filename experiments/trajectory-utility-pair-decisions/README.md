# Per-pair first decisions under trajectory utility copies

Post hoc analysis of stored rows; no learning run and no cluster use. It reads
the per-pair test records of seeds 42–46 and the copy counters of seeds 42–61
for E+1, S+1, and Mean value and Random goal at λ = 1/4, 1/2, 3/4 and 1, on
the 28 mazes. Results are in
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md#first-decisions-and-value-errors-under-trajectory-utility-copies-by-start-goal-distance).

- `pair_sources.py` selects the sources from the original archives under
  `~/Desktop/tu-runs/`. The fixed-share arms go through
  `grid_results.measured_sources` (pilot, first submission and recovery). The
  references come from the ablation and trajectory utility measures indexes,
  under the checksums the fixed-share analysis uses. It also loads the unchanged
  `per-pair-diagnostics` definitions in isolation.
- `pair_register.py` wrote `plan.json`: definitions, predictions, the verdict
  criterion, the SHA-256 of all 5,600 inputs and of the 28 ground-truth models,
  and the test pair universe per maze. It checks that every run has complete
  per-pair records and refuses to overwrite the plan.
- `pair_measures.py` holds F, the value errors, overestimation, success, the
  distance bins, pair-weighted pooling and the verdict rule.
- `pair_analysis.py` requires the plan to be committed once and unchanged. It
  regenerates the ground truth and compares its checksums, verifies every input
  file, and writes the tables in CSV and JSON together with `summary.json`.
- `pair_figures.py` plots F and the value error by checkpoint for each distance
  bin and aliasing group.

```bash
cargo build --release --bin acs2-maze-truth
python3 -B experiments/trajectory-utility-pair-decisions/pair_analysis.py
python3 -B experiments/trajectory-utility-pair-decisions/pair_figures.py
python3 -B -m unittest discover -s experiments/trajectory-utility-pair-decisions -p 'test_*.py'
```

Outputs go to `~/Desktop/tu-runs/trajectory-utility-pair-decisions/analysis/`
and `figures/`. Both directories must not exist yet.
