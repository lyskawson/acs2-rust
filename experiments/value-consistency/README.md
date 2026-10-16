# Value consistency selection with independent hindsight heads

The approved `plan-draft.json` bytes have SHA-256
`9428773fa7861f6102fdccb165ae711b597d1fd61a9e644f8becb17399ebf4eb`.
`approval.json` confirms the fixed four-arm, six-maze, twenty-seed grid and its
110 allocated CPU-hour ceiling, including 16041 pilot seconds. Prioritized K=2/4
uses T=24; both uniform controls use T=0. Hindsight is full/future/every-transition,
p=2/3. There is no tuning. The approved hindsight and episodic replay references,
and both pair baselines, are reused from verified archives.

`prepare_study.py` freezes tracked experiment operations, the exact approval,
registration, pilot rows and a scope covering all twenty-four batch roots.
`execute_batch.py` derives one twenty-run manifest and refuses another batch
while reservations remain outstanding. Shared grid tooling checks authorization,
the grant, test-only scheduling and joint allocation admission before submission.
The dedicated clone stays clean at `82c08b5a89a63878fddfb76be8d5e25d0ae17a41`.
Copy the approved static binary, SHA-256
`0c8dcb9c539462f9b84800519237a24e34c49e4d81b70a79a17cae0d7a98816d`,
to its target path; never rebuild it on the cluster.

`pilot_identity.py` is the node-side comparison adapter. It requires every
registered overlapping validation point at seed 42 before accepting a complete
run. `pilot_equivalence.py` preserves the pinned original pilot contract and
permits only the documented control-counter changes, added diagnostics and known
native header layouts. Maze6 at T=24 has no pilot oracle. The complete grid must
reproduce 180 overlapping rows before analysis.

`recovery.json` and `recovery-approval.json` record the separately approved
operational exception after the first Maze4/K=2 process timeout: seven incomplete
seeds, one recovery array, ten minutes each. `retry_batch.py` verifies this exact
exception, preserves the original STOP and failed artifacts, checks the whole
remaining hard bound, and records grant/test-only evidence before resolving STOP.
It cannot authorize another retry or change the scientific registration. Normal
batches retain their registered sizes and limits.

`verify.py` checks operations hashes, raw artifact hashes, every attempted
allocation, selected complete rows, launch identities, exact attempt scope and
node identity records. It maps declared remote references into the local copy
without modifying archived data. STOP, incomplete runs, changed identities,
unapproved attempts or a budget excess block analysis. `references.py` verifies
the two approved reference registrations, raw hashes, all twenty seeds, all
points and all three evaluation blocks before reusing their current canonical
names.

`analyze.py` first verifies all 480 runs and the 180 identity rows. It then writes
both registered test-success views, all nine paired comparisons with marginal
t(19) intervals, exact rational thresholds and baseline crossing checks. Separate
tables retain seed scores, block curves, per-head and aggregate diagnostics,
pooled variance bins, and run costs. Seeds are the independent units; heads are
averaged within a seed. Reference-set quantiles are summarized across seeds and
are not labelled as pooled quantiles. Histogram pooling uses raw counts. Uniform
arms have no measured training proposal variance; no zero is imputed. Process
CPU, sampled training CPU, allocation CPU, wall time and memory lower bounds have
separate fields.
Run-cost rows explicitly distinguish selected new runs from reused historical
references. The study allocation total instead includes every new attempt and
the pilot, including failed attempts; historical reference allocations are not
charged to the new ceiling.

`checkpoint-scores.csv` retains integer summed head successes, head/pair counts and
exact test fractions. `validate_tables.py` independently reconstructs seed scores,
primary summaries, paired intervals and first threshold crossings from these counts
with rational arithmetic. It also checks table scope, zero-query controls and allocation
scope. Tests deliberately substitute unpaired intervals, strict threshold crossing and
charges for historical references. This checker does not reuse the analysis arithmetic.

`plots.py` accepts only checksum-verified output declaring all 480 runs and 180
pilot identity rows. It exports static SVG/PNG test curves, common head disagreement,
exact-zero shares and both sets of nine paired comparisons. Matplotlib is a plotting
dependency only and is recorded beside the figures; it does not enter measurement.

```bash
python3 -B -m unittest discover -s experiments/value-consistency -p 'test_*.py'
python3 -B experiments/value-consistency/analyze.py --study STUDY --her HER_ARCHIVE --baseline PAIR_ARCHIVE --out NEW_ANALYSIS_DIRECTORY
python3 -B experiments/value-consistency/validate_tables.py ANALYSIS_DIRECTORY
python3 -B experiments/value-consistency/plots.py ANALYSIS_DIRECTORY NEW_FIGURE_DIRECTORY
```
