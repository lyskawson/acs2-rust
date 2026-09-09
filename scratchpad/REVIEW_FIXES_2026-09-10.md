# Independent review and fix report — ACS2 Rust

Prepared on 2026-09-10 for comparison with the independent verification.
Branch: `feature/mpx264`. Original reviewed commit: `d5370e7`.
Original fixes: `94f74fa` (A), `c683c3f` (B), `373c748` (C).

This is the complete report, committed in the repository's scratchpad so another
agent can retrieve it. It is outside the measurement archive in `reports/`.

## My answers to the three disagreements

### Finding 10 — preserve the historical structural metric

**I agree with the verification: redefining `structurally_correct` in place would
be the wrong fix.** My counterexample establishes a limitation of the predicate,
not permission to change the meaning of an existing column. Approximately 63,000
archived diagnostic rows already use the complete-address definition. Silently
expanding it would make subsequent counts look comparable when they are not.

In MPX-6, address `0#` and data `00##` imply answer 0 with specificity 3 (`a+1`),
despite one unspecified address bit. Thus a complete address is sufficient for a
standard compact rule but is not necessary for every correct minimal rule. Zero
`correct` or `addr_full` does not prove that no correct candidate exists.

I documented this limitation in `reports/MPX_final.md` and `docs/AGENT_HANDOFF.md`
and preserved the predicate and every archived `correct` value. If a more complete
metric is needed later, it should have a new name and an explicit definition, with
tests against the multiplexer truth function. Historical logs alone cannot supply
that new metric where classifier populations were not saved.

### Finding 2 — diagnostic overhead versus a resource-limit defect

**I agree that my original wall-clock framing was too broad.** The diagnostic
invariant concerns the population and learning RNG: the same seed/configuration
produces identical learning state at equal trial counts. More evaluation work can
make a wall-clock cap stop the run at a different trial without violating that
invariant. The matching 17,880,000-trial reproductions support the stated invariant;
they do not promise identical stopping trials under every time budget.

I corrected the wording rather than changing the diagnostic algorithms. The
separate accepted defect was that an evaluation could finish beyond a resource
limit and still grant SUCCESS without another check. That check now exists under
`--strict-resource-limits`, default off to retain the historical protocol. It
refreshes peak RSS and checks elapsed time after evaluation/diagnostics, before
granting SUCCESS. A strict run can therefore record knowledge 1.0 and still be
TIME-LIMITED because evaluation exceeded its budget.

### Finding 7 — encoding provenance is not concealment

**I agree: “concealment” overstated a column working as designed.** The archived
verdict logs predate the header's encoding field. `encoding_source` explicitly
distinguishes recorded values from filename, wrapper-default, or submission-record
inference. The existence of inferred values is not a failure of that mechanism.

I preserved the encoding values and their provenance. A downstream table should
make the provenance discoverable, but not displaying every provenance field is a
different issue from silently losing the information. The readable tables point
to `encoding_source`. I would not describe the archive as concealing encoding.
The accepted ER half was an actual data-loss defect: recorded `do_ga`, replay
capacity, warmup, and replay count were discarded. Those fields now survive in
the CSVs, and m=1 and m=3 can be selected and distinguished explicitly.

## What changed, grouped as in the fix brief

### Group A — false claims in documents

Files: `reports/MPX_final.md`, `docs/AGENT_HANDOFF.md`, `README.md`,
`docs/PROJECT_CONTEXT.md`, and `CLAUDE.md`.

- **Candidate discovery versus reliability, finding 9.** With seed 42 and
  `u_max=11`, the 4,865 qdetail points contain 16 zero, 2,600 partial, and 2,249
  printed-full candidate-coverage readings. Best recorded quality is 0.823,
  below `theta_r=0.9`. I withdrew the claim that no classifier of any quality
  exists in the starved class. The handoff now directs replay research toward
  the candidate-to-reliable gap: quality, retention and reliable coverage at
  matched learning applications. These aggregate, finite-precision measurements
  do not establish that the same candidates persist between evaluations, or
  identify a unique cause such as insufficient reinforcement or replacement.
- **Accuracy, finding 11.** The first reading is 0.4989. Only 769 of 2,715 points
  (28.3%) print 1.0000. The last lower reading is at 299,160,000 trials; the final
  uninterrupted printed-full sequence starts at 299,280,000. The log spans
  325.8 M trials; there was no accuracy-1 plateau of that duration. At four
  decimals, 49,999/50,000 also prints 1.0000, so this is rounded sampled accuracy,
  not an exact zero-error or exhaustive result.
- **Handoff consistency, finding 12.** Outcome encoding reached SUCCESS on four
  seeds, including seed 45 at 55,560,000 trials. Seed 44 was cancelled before
  success; permanent failure and eventual convergence are both unestablished.
  Finite-run endpoints are no longer called permanent ceilings. Canonical
  epsilon-1 seed 43 reached sampled knowledge 1.0 at 427,920,000 trials; seed
  42's first run stopped at 301.44 M, before a comparable second class opened
  on seed 43. The separate rerun is not evidence of eventual success. Budget
  and queue figures are identified as dated snapshots.
- **Defaults and onboarding, findings 14 and 18.** Documented the actual direct
  runner default `u_max=100000`, distinct from the wrapper's explicit derived
  setting. Fixed the nested `baseline/` path error. Located `Environment` in
  `acs2-core`, including the architecture tree. Documented the cluster checkout
  and musl-binary prerequisite. Corrected the branch to `feature/mpx264`, the
  MPX-70 ideal specificity to 7, and the ledger statement excluding the later
  `u_max` sweep from the report. Distinguished the baseline protocol from later
  encoding and epsilon overrides.
- **Claims deliberately narrowed.** Preserved the historical structural metric,
  stated diagnostic invariance at equal trial counts, and limited reproducibility
  claims to the tested 64-bit Apple M1 and x86_64 Bem2 platforms.

### Group B — silent data loss

Files: `tools/parse_mpx_logs.py`, `tools/summarize_mpx.py`, `tools/sync_runs.sh`,
`README.md`, `tools/README.md`, the three derived archive CSVs, and per-size tables.

- **Finding 6:** added a one-based header `block` to run identity and retained
  source, size, repeat and configuration. The table visibly includes variant.
  Seven previously hidden Butz verdicts return: three at k=37, three at k=70,
  and one at k=135. MPX-37 contains all 18 successes.
- **Findings 7 and 13:** retained `do_ga` and the three replay parameters;
  merged all eight qdetail coverage/quality fields into the trajectory point at
  the same trial. README regeneration now calls the full rebuild path. Subset
  parsing must redirect all three outputs away from the archive.
- **Finding 8:** commit detection uses porcelain status including untracked
  files. A new header-only log can trigger a commit even when it adds no CSV
  rows. `sync_runs.sh --local` rebuilds the complete existing archive without
  contacting the cluster.
- Added archive regressions for identity, metadata retention and untracked-file
  handling. Generated tables use English and display replay settings and GA.

### Group C — code and analysis defects

| Finding | Change and location | Compatibility |
|---|---|---|
| 3 | `acs2-bench/src/bin/mpx_reach.rs` tracks the knowledge evaluation trial. Terminal knowledge without a measurement of that exact final population prints `unmeasured`. Parser/table timing metadata distinguishes current, stale and unverified historical values. | No extra evaluation is started after hitting a cap. Existing SUCCESS values and trials are unchanged. |
| 1 | `tools/mpx_selection.py` supplies shared agent, GA, ER, encoding, epsilon and u_max filters to plotting and summarization. Signal plots use them too. Curves and SUCCESS markers are tied to the same source/header/repeat identity. | Independent runs sharing a seed must be selected explicitly with `--source` and, where needed, `--block`. |
| 5 | `acs2-core/src/alp.rs`, configuration and benchmark argument handling add `butz-checked`. It counts actual specified attributes and selects a nonempty generalization target. | Opt-in through `--alp-gen-variant butz-checked`; historical `butz` and default `pyalcs` remain unchanged. |
| 4 | `mpx_reach.rs` adds `--isolate-repeats`, launching a fresh process for each size/repeat. Child headers record the actual seed and `rss_scope=repeat-process`. | Default retains process-lifetime RSS behavior. Historical multi-repeat peaks cannot be reconstructed from logs. |
| 17 | Replay construction rejects zero capacity; GA deletion rejects an offspring batch that cannot fit `theta_as`. | Valid configurations retain their behavior. Invalid/nonterminating boundary cases fail clearly. |
| 2, accepted sub-point | `--strict-resource-limits` rechecks time and RSS after evaluation/diagnostics before SUCCESS. | Default off; the new header records the chosen policy. |
| 16 | Parser flushes pending trajectory and diagnostic records before replacing header context and at EOF. Table rebuilding discovers sizes from trajectories as well as verdicts. | Header-only logs remain files without fabricated measurement rows. |

`tools/rebuild_figures.py` regenerated the seven committed PDF/PNG figure pairs
from explicit sources, recorded in `reports/figures/manifest.json`. Unique PNGs
were inspected visually. MPX-70 curves represent the five canonical ACS2 runs;
the epsilon-1 MPX-135 figure uses seed 42's first time-limited run and seed 43's
successful run, without splicing in the later seed-42 rerun.

**The finding-3 reporting correction does not move any headline SUCCESS number.**
All 45 archived SUCCESS rows retain their values and trial counts. Historical
knowledge values were preserved and annotated: 45 at-verdict, 26 stale with a
known earlier evaluation trial, and 8 legacy values whose timing is unverified.
The separate opt-in strict-resource policy can change a future resource verdict;
no historical verdict was relabelled to simulate it.

## This follow-up — report delivery and mandatory tests

The previous report was not literally absent: I reopened
`/private/tmp/acs2-review-fixes-2026-09-09.md`, a 19,133-byte file containing the
grouped changes, disagreement answers, limitations and gate results. However,
putting the only copy in a temporary directory did not make it reliably
discoverable from the repository or another agent's checkout. This committed
scratchpad report replaces that delivery arrangement. The earlier temporary copy
is not required to understand this report.

Likewise, I disagree with the literal claim that the three Rust tests had never
been compiled or run: the former `tools/check_reach_protocol.py` invoked
`rustc --test`, and the retained execution log shows all three passed. **I agree
with the substantive gate criticism.** They were outside Cargo discovery and
absent from the standing gate in `CLAUDE.md`. A future agent could pass that gate
while never executing these checks. A separate successful invocation did not
close that regression-protection gap.

I made these changes:

1. Moved the source to `acs2-bench/tests/reach_regressions.rs`, updated its include
   path and explicitly registered the `reach_regressions` target in
   `acs2-bench/Cargo.toml`.
2. Removed the redundant Python compiler driver and the Rust file from `tools/`.
3. Added the Python suite and an explicit release benchmark build to `CLAUDE.md`.
   Updated the live gate count in CLAUDE, README, tools documentation and handoff
   from 73 to **76**. Documented the single-target Cargo command as well.
4. Established `scratchpad/` as the discoverable home for this report and corrected
   the remaining handoff footer that still named `feature/mpx` and ambiguously
   put all reports in the measurement directory.

### Why retaining `include!` is acceptable here

The integration target includes the actual binary source, not a copied version
of the loop. Its tests exercise private runner functions with frozen synthetic
populations. Cargo's test harness supplies the test entry point; the included
CLI `main` is not run. Compilation and execution through the normal workspace
command succeeded. This is a bounded test-wiring change and requires no production
API or measured-path refactor.

I did not lift the runner into a library simply to expose private functions.
That may be useful if it acquires other callers, but it is unnecessary to make
these checks mandatory. The source, assertions and test names are unchanged apart
from the include path. No production Rust source changed in this follow-up.

The checks cover verdict snapshot timing, post-evaluation time/resource decisions,
and the OS property that fresh processes do not share the previous process's RSS
peak. The RSS check is not an end-to-end test of every CLI child-launch option,
and none of these tests establishes large-MPX convergence or a new speed result.

## Decisions not to change

- **Structural diagnostic definition:** preserved for archive comparability;
  a broader measure needs a separate name and definition.
- **Diagnostic learning behavior and RNG:** unchanged. Resource overhead is
  described precisely; no cross-pointer-width guarantee is made.
- **Encoding inference/provenance:** preserved as the intended historical
  reconstruction mechanism. Missing recorded parameters are not invented.
- **Historical Butz behavior:** retained as a reproducibility variant, including
  its known counter defect. The corrected behavior has the distinct opt-in name.
- **Default resource policies:** retained as requested; isolation and the strict
  post-evaluation policy are opt-in and logged.
- **Old measurements:** no synthetic final knowledge, peak RSS, new structural
  metric, or strict-policy verdict was retroactively substituted into raw logs.
- **Production architecture in this follow-up:** no library extraction, learning
  change or new flag was needed to wire tests and deliver the report.
- **Experiments and external communications:** no cluster experiments were
  submitted, no jobs cancelled, and no supervisor messages sent.

## Findings that were wrong or narrower than initially stated

- The three disagreements above narrow my original interpretation: missing
  complete-address structure is not absence of correct rules; wall-clock
  overhead is not diagnostic mutation; recorded inference is not concealment.
- Finding 3 concerns non-success terminal reports. SUCCESS values are measured
  at the terminal trial. Old non-success logs sometimes lack enough evidence to
  identify even the stale value's evaluation trial.
- Finding 16 was a real reset-before-flush control-flow defect, confirmed by a
  regression fixture, but fixing it recovered no additional rows from the
  then-current archive. Header-only files have no measurement rows to recover.
- The status-script sub-point of finding 18 overstated a default-path exit bug:
  local `set -euo pipefail` is not automatically inherited by the separate remote
  `bash -s` used by its SSH heredoc. Under that invocation, a no-match grep can
  still reach awk's `(starting)` fallback. I did not change that script or claim
  a reproduced default-path exit. Its confirmed documentation problems were fixed.
- The original 73-test workspace result was accurate but incomplete as the only
  standing gate. The three separate runner checks now belong to that gate,
  increasing its actual executed count to 76.

## Additional problems discovered during implementation

1. One experimental arm still contained multiple independent same-seed runs.
   Filtering only learning configuration did not prevent source splicing or a
   SUCCESS marker being borrowed from another run. Selection now checks run
   identity, and regression tests cover both cases.
2. Summarization attached earlier accuracy/coverage diagnostics to a later
   terminal population. Those cells are now shown on final rows only when their
   evaluation trial equals the verdict trial; older trajectory data remains
   available. This cleared stale diagnostic cells in 11 existing table rows.
3. A source can repeat exactly the same header configuration. Variant alone is
   insufficient as an identity key. Header blocks are explicit, and duplicate
   log basenames or duplicate/out-of-order repeat labels are rejected.
4. Signal plotting ignored filters applied elsewhere, and reach titles hard-coded
   GA on. The signal now uses the shared selection and titles reflect the arm.
   Trajectory points without knowledge are not drawn as measured zero knowledge.
5. The architecture tree and a later handoff footer retained stale information
   after nearby prose had been corrected. Those remaining entries were fixed.
6. The separate runner harness and temporary report location were discoverability
   failures in my own delivery. Both now have ordinary repository locations, and
   the tests are mandatory in the documented Cargo gate.

## Archive accounting

Every rebuild used the complete `sync_runs.sh` input set:

```text
reports/slurm_*.out
reports/*.cancelled
reports/mpx_m2b_reach*.log
reports/mpx_m3_e1_traj70_*.log
```

No rebuild used a reduced set. Schema changes in groups B/C removed no rows and
changed no previous measurement-column values. They added recorded configuration,
qdetail and timing/protocol provenance. The qdetail additions populated 15,048
existing trajectory rows; ER fields populated 8 verdict, 837 trajectory and 837
diagnostic rows. The seven hidden Butz verdicts were a table-key loss, not missing
raw verdicts.

| Snapshot | Verdict rows | Trajectory rows | Diagnostic rows |
|---|---:|---:|---:|
| Original reviewed archive | 79 | 68,216 | 63,334 |
| After the first required sync and groups A–C | 79 | 68,340 | 63,364 |
| After this follow-up's required sync, `3ae1302` | 79 | 68,447 | 63,392 |

The 2026-09-09 sync (`42742fd`) added 94 trajectory points to the epsilon-1 rerun
at trials 360,000–11,520,000 and 30 trajectory/30 diagnostic points to outcome-u9
at 13,560,000–17,040,000. It added no verdict.

The 2026-09-10 sync added only these points from existing jobs:

| Source | New trajectory rows | New diagnostic rows | Trial range |
|---|---:|---:|---|
| `slurm_mpx135_s42_eps1b_u11.out` | 79 | 0 | 11,640,000–21,000,000 |
| `slurm_mpx135_s42_outcome_u9.out` | 28 | 28 | 17,160,000–20,400,000 |

Multiset comparisons show every previous row survives unchanged, and the entire
verdict CSV remains unchanged. All 68 verdict-bearing sources remain. Across all
three CSV views there are 76 represented sources; the complete input list contains
83 logs, including header-only files. These are different counting scopes.
Neither updated source is selected by a committed figure recipe, so no plotted
data changed and no figure regeneration was needed in this follow-up.

## Gate results I observed

| Gate | Original fix pass | This follow-up |
|---|---|---|
| `cargo test --workspace --release` | 73 passed, 0 failed | Before wiring: 73. After wiring: **76 passed, 0 failed**, including the named integration target. |
| Python archive/plot suite | 13 passed | **13 passed**, run through `uv run --project tools python -B -m unittest discover -s tools -p 'test_*.py'` with offline/cache environment settings. |
| P9 maze learning columns 1–7 | Byte-identical before and after the original changes | **Byte-identical** to the committed CSV; the release binary was rebuilt first. |
| Separate runner harness | 3 passed via the former Python/rustc driver | Removed; the same three tests now execute automatically inside Cargo's 76. |

The new workspace output explicitly contained:

```text
Running tests/reach_regressions.rs
test regression::verdict_requires_a_measurement_at_the_terminal_trial ... ok
test regression::fresh_processes_do_not_share_a_previous_repeat_peak ... ok
test regression::strict_limits_check_evaluation_cost_before_success ... ok
```

For this follow-up I ran `./target/release/acs2-bench` against the normal report
path, compared the first seven columns, and restored `reports/bench_rust.csv`
byte for byte in a `finally` block. The learning-column projection has SHA-256
`f6303eb025cbe099bbd177a6182a74b73623ae0b0e3de28f8ba661756cac0b85`, matching the
previous pass. No timing noise is included in the fixes.

The Butz and invalid replay/GA boundary assertions added to existing tests still
pass. This follow-up changes only test wiring, documentation and report delivery,
apart from the required retrieval of existing cluster output. It does not add
large-MPX experimental evidence, certify current queue/budget state, or reconstruct
unlogged historical populations.
