# Shared grid execution

This is the single reusable copy of the baseline pilot operations. Scientific
registrations and analyses belong to each experiment directory. The original
`baseline-pilot/` is preserved. Result roots, resolved plans and approvals are
outside every checkout; only the dedicated TU clone builds the musl binary.

`submit.py` accepts a manifest, result root and clean pinned clone. It records
plan bytes and hash, binary/source identity, one submission intent per array,
grant usage, shared queue commitment, `sbatch --test-only` and submission reply.
It checks pilot + used + reserved + new allocation limits against the manifest's
CPU-hour limit. An existing root or unresolved intent needs investigation, never
blind resubmission. Cost pilots use seed 42. A grid requires an explicit approval
file matching the resolved plan hash. No grid is authorized by its manifest alone.

A pilot is one array; a grid has one array per configuration, chained with
`afterok`. Each array uses 20–40 concurrent single-CPU jobs, no host pinning,
no supervision lease, no laptop dependency, no automatic requeue. For a mixed
agent array the allocation limit is the larger agent limit; the budget bound
counts that limit for every element. This deliberately includes waiting/guard
overhead, rather than estimating only child CPU.

The node verifies its exact allocation identity and remaining time, clean source,
binary hash, partition, CPU model and preset. The collector retains a sixty-second
first-row watchdog, row/block validation, process timeout, partial raw tails,
process CPU/RSS and sampled CPU attribution. A real failure creates STOP, causing
later starts to refuse measurement; dependencies block later configurations.
No app heartbeat or external monitor is needed. Failed attempts are retained.

`row_schema.py` reads schemas 2 and 3 as full-pool evaluation and schema 4 as either
that legacy path or three distinct blocks. It checks each block's registered
reference and refuses ambiguous top-level success fields in split rows.
`grid_attempts.py` selects a complete attempt per key and requires duplicate
complete attempts to agree on all learning fields. Pull every file without
deletion after completion, retain allocation-only `sacct` accounting, and verify
raw rows before analysis. Training/evaluation CPU attribution is an estimate;
allocation CPU and process CPU are reported separately.

```bash
python3 -B -m unittest discover -s experiments/grid -p 'test_*.py'
python3 -B experiments/grid/submit.py --root RESULT_ROOT --manifest MANIFEST --repo TU_CLONE
```
