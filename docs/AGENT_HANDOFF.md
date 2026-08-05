# Agent onboarding — ACS2 Rust / MPX scaling (continuation)

You are a research + implementation agent for a master's-thesis ACS2 project
(supervisor: prof. Olgierd Unold, PWr). This document is the complete handoff from the
previous session. Read it fully, then check experiment results (section 4) and follow
the decision tree (section 5).

## 1. Project state

Rust port of canonical ACS2 (pyalcs semantics), differentially validated against pyalcs
(761/761 cases). Working dir: `~/Desktop/acs2-rust`, branch `feature/mpx`, provenance
commits: `2f8111c` (code: canonical u_max ALP-generalization + bench trajectory logging),
`3454ddf` (docs/reports). Read `docs/ARCHITECTURE.md` first; supervisor-facing narrative
in `reports/MPX_final.md`; literature survey in `reports/MPX_literature_review.md`.

The MPX arc so far:
- M1 specialize-only: caps ~k=11. M2a GA-on: knowledge=1.0 up to k=37, freeze at k>=70.
- M2b: canonical u_max branch (dormant in pyalcs at u_max=100000) activated; breaks the
  freeze. Two variants of the underspecified expected-case generalization: `pyalcs`
  (parent, unchanging-count, correct threshold u_max=a+2) and `butz` (child, full-count,
  needs a+3). Freeze-break is variant-independent.
- M3/E1 headline: **k=70 SOLVED at knowledge=1.0** — seed 42, pyalcs @ derived u_max=8,
  GA on, 17,880,000 trials, 6105 s on M1, 277 reliable rules at spec 7.04 (ideal).
  Log: `reports/mpx_m3_e1_traj70_pyalcs.log`. This exceeds every published ACS/ACS2
  result (literature max ~37-bit; only ExSTraCS 2.0 ever solved 135-bit directly —
  see the literature review). Butz variant is ~3x slower per wall-clock (bigger
  populations); **standard config is pyalcs @ --u-max derived** from now on.
- The trajectory is an S-curve: slow accumulation -> apparent plateau ~0.6 -> population
  condensation (~5k -> ~1.2k rules) -> sprint to 1.0. Never diagnose a plateau from a
  short run.

## 2. Hard invariants (do not violate)

- Maze path untouched: maze config keeps u_max=100000 (ALP-gen branch dead). Before any
  acs2-core change lands: `cargo test --workspace` green (59 tests) and the P8/P9 maze
  gates must stay byte-identical.
- Determinism from seed, injected RNG. Verified cross-architecture (M1 vs Xeon produce
  identical trajectories) — so "trials to SUCCESS" is the machine-independent metric;
  wall-clock is machine-specific color.
- Knowledge metric: exhaustive for k<=20, sampled (50,000 inputs, fixed eval seed) for
  k>=37 — always state this caveat in reports.
- Thesis provenance: every claim cited, every measurement reproducible (params + seeds
  + commit hash). No overclaiming: k=70 was 1 seed until seeds 43/44 confirm.
- Repetition policy: gates high-n, boundary probes low-n (1-3 seeds). The user's M1
  overheats — anything >30 min runs on WCSS, not the laptop.
- Supervisor emails: Polish, plain language (he dislikes AI-jargon), user sends them.

## 3. WCSS cluster (Bem2) — access and rules

- SSH: `ssh -i ~/.ssh/id_rsa_wcss alelys2099@ui.wcss.pl`. Scheduler: SLURM.
- **Critical**: login node has newer glibc than compute nodes — always build
  `cargo build --release --target x86_64-unknown-linux-musl` (static). A glibc build
  fails on compute nodes with `GLIBC_2.30 not found`.
- Partitions: `bem2-cpu-short` (3-day limit), `bem2-cpu-normal` (21-day). Jobs are
  1 core / 8G / single task; sbatch scripts in `~/acs2-rust/slurm/` on the cluster.
- Two checkouts on the cluster: `~/acs2-rust` (rsynced tree the current jobs run from;
  verified byte-identical to commit 3454ddf) and `~/acs2-rust-repo` (proper git clone
  of feature/mpx) — **run future work from the clone via git pull**.
- Rust toolchain installed via rustup in `~/.cargo`.

## 4. Experiments in flight (check these first)

**All jobs finished 2026-08-05. Nothing is in flight.** Results (logs pulled into
`reports/`, reduced into `reports/mpx_{trajectory,verdicts}.csv`):

| Job | What | Verdict | Trials | Wall |
|---|---|---|---|---|
| 5552394 | k=70, seed 43 | **SUCCESS** | 17,820,000 | 13,692 s |
| 5552395 | k=70, seed 44 | **SUCCESS** | 44,580,000 | 28,988 s |
| 5552396 | k=70, seed 45 | **SUCCESS** | 21,300,000 | 11,529 s |
| 5552397 | k=70, seed 46 | **SUCCESS** | 66,420,000 | 46,266 s |
| 5551654 | k=135, seed 42 | TIME-LIMITED | 105,621,000 | 250,000 s |

k=70 is confirmed on 5/5 seeds at knowledge = 1.0, all converging on 268–277 reliable
rules at the ideal specificity 7. k=135 produced **zero** reliable rules across all 1,760
evaluation points with a stationary population — a mechanism boundary, not a budget one.
Full analysis in `reports/MPX_final.md` §"Act IV".

Status of any future runs: `./slurm/mpx_status.sh`. Submit with
`sbatch --job-name=<n> ~/acs2-rust-repo/slurm/mpx_reach.sh <size> <seed> <time_cap_secs>`.

### 4a. The budget correction, and what it got wrong (read before trusting any projection)

The k=70 seeds were first submitted with a 40,000 s cap calibrated on seed 42 (17.88 M
trials, 6105 s on the M1), then cancelled ~1.5 h in and resubmitted at 600,000 s, because
seeds 43/44 sat at knowledge 0.26 at 4.2 M trials where seed 42 had been at 0.62 — an
apparent ~2.5x lag projecting to ~45 M trials, against a cap estimated to buy only
~27-36 M.

**The outcome contradicted that projection, and the error is worth keeping.** Actuals:
seed 43 finished at 17.82 M trials / 13,692 s and seed 44 at 44.58 M / 28,988 s — *both
inside the original 40,000 s cap*. The restart was insurance that the two original seeds
did not need. Only seed 46 (66.42 M trials, 46,266 s), which was added later, actually
required the larger cap.

Two specific lessons:

- **Early trajectory position does not predict trials-to-success.** Seed 43 was far
  behind seed 42 at 4.2 M trials and still finished at essentially the same total
  (17.82 M vs 17.88 M). The S-curve's sprint phase compresses; a lag measured mid-climb
  extrapolates badly. Do not size budgets from a lag ratio.
- **Throughput projections must account for condensation speeding the run up.** The
  estimate assumed the observed ~650 trials/s would hold; actual whole-run averages were
  1,301-1,847 trials/s, because a condensed population makes trials cheap. The measured
  in-run slowdown (1395 -> 648 trials/s) reverses once condensation starts.

What does hold: nodes run fully packed (`CPUAlloc=48/48`), so cluster wall-clock is
contention-polluted and ~2.7x below the M1 pre-condensation, and **trials-to-success
varies 3.73x across seeds** (17.82 M-66.42 M, median 21.30 M). Budget from the observed
maximum, not from a projection, and prefer one over-generous cap to a second attempt.

Methodological point that stands: a seed truncated by a cap tuned on the fastest seed is
a measurement artifact, not a failed confirmation. Diagnose by specificity and
reliable-count, not knowledge alone. Partial 40 k-cap logs kept as
`reports/slurm_mpx70_s4{3,4}.cap40k.cancelled.out`.

Bench binary flags (`mpx_reach`): `--sizes 70,135` `--n-exp` `--seed`
`--time-cap-secs` `--u-max derived|<int>` `--alp-gen-variant pyalcs|butz`
`--log-trajectory` `--eval-interval <trials>` (evals are pure — no agent-RNG impact).

## 5. Decision tree after results

1. **k=70 seeds 43/44 SUCCESS** -> reach claim confirmed on 3 seeds. Update
   `reports/MPX_final.md` with an M3 section (trials-to-success per seed, trajectory
   shape), commit, and draft the supervisor update (Polish).
2. **k=135 SUCCESS** -> first direct 135-bit solution by any anticipatory LCS (verify
   phrasing against the literature review before claiming). Queue 2 confirmation seeds
   on WCSS. This likely closes the "beat the MPX boundary" phase.
3. **k=135 TIME-LIMITED** -> read the trajectory: if still pre-condensation (population
   large, knowledge climbing), resubmit on `bem2-cpu-normal` with a longer cap —
   budget, not mechanism, is the first lever (that is the k=70 lesson). If genuinely
   plateaued (knowledge flat >10M trials with condensed population), investigate churn
   (reliable-count oscillation) before touching parameters.
4. **A k=70 confirmation seed fails** -> treat as real: analyze its trajectory vs seed
   42, report honestly (seed variance is a finding, not an embarrassment).
5. **After reach closes**: (a) u_max sweep (a+1..a+4, k=20/37, low-n) — the honesty
   ledger flags that derived u_max bakes in solution knowledge; a sweep de-fangs that
   critique; (b) then the thesis core: prioritized experience replay (supervisor's
   original topic). The empirical motivation is already written: ALP-gen remains
   action-set-revisitation-bound; ER/prioritization attacks exactly that. Start from
   supervisor's ER papers (ACS2ER: dl.acm.org/doi/10.1145/3520304.3533996, ACS2HER:
   arxiv.org/abs/2601.09400, diversity-based ER: arxiv.org/abs/2410.20487).

## 6. Standing rules

Idiomatic Rust, SOLID, no code comments, English identifiers/commits, injected RNG.
Anything touching the measured core path goes behind a flag with defaults preserving
current behavior. Commit + push to `feature/mpx` after each completed step (user has
approved this workflow); reports live in `reports/`, implementation record in
`docs/ARCHITECTURE.md`. Ask the user only for scope decisions (new experiment phases,
supervisor communication) — execution decisions are yours.
