# Archive — closed phases

These documents record work that is finished. They are kept because the thesis
requires every measurement to stay traceable, not because they describe the
current state of the project. For that, read `README.md`, `docs/PROJECT_CONTEXT.md`
and `docs/ARCHITECTURE.md`.

| Document | What it records | Status |
|---|---|---|
| `BUILD_PLAN.md` | The phased prompts used to build the implementation, P1–P9 | Complete; all gates passed |
| `AUDIT_PYALCS_REPORT.md` | Audit of four claimed defects in the pyalcs reference | Findings folded into `ARCHITECTURE.md` as deliberate deviations |
| `ALCS_ANOMALY_VERIFICATION.md` | Three data/config inconsistencies found in the supervisor's ALCS repository | Reported to the supervisor 2026-06 |
| `TIMER_REGION_FINDINGS.md` | Why the Rust and Python timed regions are comparable | Underpins the P9 timing claim |
| `MAZE_GEOMETRY_REORG_PORT.md` | One-file-per-maze reorganisation and the port of 22 ALCS geometries | Landed; geometry only, no learning semantics touched |
| `CPU_SINGLE_COMPARISON.md` | Timing against the supervisor's `cpu_single` on the two cell-identical mazes | Superseded in scope by the 22-maze run below, kept because that one cites it |
| `CPU_SINGLE_FULL_COMPARISON.md` | The same comparison over all 22 supervisor mazes | Final version of that comparison |

Two files were removed rather than archived: `CPU_SINGLE_PRECHECK_SCRATCH.md`
(self-described scratch notes, not a deliverable) and `reguly_notatka.md`
(an earlier draft that `ACS2_RULE_DUMPS_GUIDE.md` supersedes). Both are in git
history if ever needed.
