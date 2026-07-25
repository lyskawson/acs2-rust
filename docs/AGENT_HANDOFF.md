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

Jobs submitted 2026-07-25, logs on the cluster in `~/acs2-rust/reports/`:

| Job | What | Internal cap | Log |
|---|---|---|---|
| 5551654 | k=135, seed 42, pyalcs @ u_max=9 | 250,000 s (~69 h) | `slurm_mpx135_s42.out` |
| 5551655 | k=70, seed 43 (confirmation) | 40,000 s | `slurm_mpx70_s43.out` |
| 5551656 | k=70, seed 44 (confirmation) | 40,000 s | `slurm_mpx70_s44.out` |

Check: `ssh -i ~/.ssh/id_rsa_wcss alelys2099@ui.wcss.pl 'squeue -u alelys2099; tail -3 acs2-rust/reports/slurm_*.out'`
A finished run ends with `repeat 0: SUCCESS|TIME-LIMITED|... trials=... knowledge=...`.

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
