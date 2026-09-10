# Agent onboarding — ACS2 Rust, multiplexer scaling and experience replay

Handoff for a fresh session. `CLAUDE.md` at the repo root carries the operational rules
and loads automatically — read it, then this. `docs/PROJECT_CONTEXT.md` says what the
project is, `docs/ARCHITECTURE.md` how it is built, `reports/MPX_final.md` is the
scientific narrative, `reports/MPX<k>_runs.md` every run at a size in one table. This
file carries only the **live state**.

**Your task, in order, is §8.** Checkpointing is **implemented, reviewed and gated**
(2026-09-10), and the archive reads a chained run as one run. What remains is step 3:
once the new WCSS grant lands, k=264, plus watching the two jobs already running.

**Branch: `feature/checkpointing`**, cut from `main` on 2026-09-10. `main` is level with
it. `develop` was retired: it never differed from `main` in a solo workflow. `feature/mpx`,
`feature/mpx264`, `feature/acs2er` and `feature/cpuSingleComp` were merged and deleted.
The cluster clone `~/acs2-rust-repo` tracks `feature/checkpointing` too.

**Git history was rewritten on 2026-09-09** to drop agent co-author trailers; every commit
is authored solely by the user, and GitHub lists one contributor. Check before any merge
to `main`, because repairing it later means another force-push:

```
git log --oneline --grep="Co-Authored-By" --all | wc -l     # must be 0
```

**The repository is public and shared with the supervisor's other students** — he asked
for it as a base for their work and the user agreed (correspondence 28/29). That raises
the bar on README and `reports/MPX_final.md`, both written for that audience, and it is
why one-off working reports are not committed. `docs/SUPERVISOR_CORRESPONDENCE.md` and
`docs/SUPERVISOR_NOTES.md` are gitignored and must stay that way.

## 1. Where the research stands

**MPX-70 is solved** at knowledge = 1.0 on all five seeds tried (17.8 M–66.4 M
trials, 268–277 reliable rules at the ideal specificity 7). Published ACS/ACS2
results stop at 20–37 bits.

**MPX-135 is solved.** Two results, and the distinction between them is the whole
story:

- **Task performance and anticipatory knowledge differ.** The canonical seed-42
  `u_max=11` accuracy log contains 2,715 points over 325.8 M trials, beginning at
  0.4989. Only 769 points (28.3%) print 1.0000; the last lower reading is at
  299,160,000 trials. Later readings print 1.0000 while knowledge is near 0.7499.
  This is rounded sampled accuracy: 49,999/50,000 also prints 1.0000, so it does
  not establish exact zero error.
- **Under `--encoding outcome`, four of five seeds reached sampled knowledge 1.0:**
  s42 at 43.2 M, s43 at 46.8 M, s45 at **55,560,000**, and s46 at 30.2 M trials,
  with 539 / 539 / 532 / 535 reliable rules at specificity 8.00–8.01. Seed 44 was
  cancelled before success. Its earlier bloated population is not evidence that
  it could never converge.

At canonical encoding and epsilon 0.8, four time-limited runs ended at last sampled
knowledge 0.7499 (s42) and 0.4980 / 0.4980 / 0.4918 (s43–45). These are finite-run
readings, not ceilings. The class-level diagnosis is in §3.

**`epsilon = 1` reaches sampled knowledge 1.0 at k=135 under canonical encoding.**
Seed 43 reached it at **427,920,000 trials**, with 532 reliable rules at specificity
8.02 and all four sampled coverage classes at 1.0000, in 152.3 h. This result rests
on one seed in the committed archive. Seed 42's first run hit its wall-clock cap at
301.4 M trials; the longer rerun is listed in the dated queue snapshot (§6).

**The mechanism, measured on both seeds — one class opens early, the other waits.**
Under `epsilon = 1` both seeds fill one wrong-answer class while the other stays
without reliable coverage for hundreds of millions of trials. Only seed 43 is observed
to open the second class and close within ~23 M further trials:

| seed | first class opens | second class opens | verdict |
|---|---|---|---|
| 43 | `a0_nochange` at 20.2 M | `a1_nochange` at **404.5 M** | SUCCESS at 427.9 M |
| 42 | `a1_nochange` at 80.6 M | `a0_nochange` still 0 at 301.4 M | cut by the wall limit |

Seed 42 is **not** a seed that fails to respond — it was stopped 100 M trials before the
point where seed 43's second class opened. A run sitting at 0.7442 with one class at
exactly zero is mid-climb, not converged. Re-running seed 42 with a cap past ~430 M
trials is the highest-probability route to a second full closure, well above a fresh
seed of unknown cost.

**ACS2ER exists and is validated** differentially against pyalcs (`p11_acs2er`).
First comparisons say uniform replay trades compute for episodes rather than
improving efficiency; at matched learning applications no advantage is measurable.

## 2. Hard invariants (do not violate)

- Maze path untouched: `u_max = 100000` on the maze config keeps the ALP-gen branch
  dead. Before any core change lands: `cargo test --workspace --release` green
  (**98 tests**, including reach regressions) and the P9 maze learning columns byte-identical to
  `reports/bench_rust.csv`.
- Determinism from an injected RNG, verified on 64-bit Apple M1 and x86_64 Bem2.
  No equivalence is claimed across 32-bit and 64-bit pointer widths. **Trials-to-success
  is the machine-independent metric**; wall-clock is machine-specific colour.
- Every diagnostic is read-only over the population and off by default. Learning
  is identical at equal trial counts: with and without instrumentation, seed 42
  solves k=70 at 17,880,000 trials. Evaluation overhead can change the trial count
  reached before a wall-clock cap.
- Knowledge is exhaustive for k ≤ 20 and **sampled** (50,000 inputs, fixed eval seed)
  for k ≥ 37 — state this caveat in reports.
- The user's laptop overheats; anything longer than a few minutes runs on WCSS.
- Supervisor emails: Polish, plain language, no AI jargon, **the user sends them**.

## 3. The k=135 diagnosis, in the order it was established

1. Population specificity sits at the ideal (~8.0), so "specialization outruns
   generalization" does **not** explain it. Mark density does not either — the
   solvable k=70 run ends at 0.914 against k=135's 0.935.
2. Splitting knowledge by action × correctness: the **wrong-answer** classes starve.
   Seed 42 at `u_max` = 11 filled one of the two (hence 3/4); seeds 43, 44 and 45
   have **both** at exactly zero (heading for 1/2). It is not tied to an action
   index, so swapping action labels would test nothing.
3. Scanning all classifiers at `u_max=11` contradicts the former discovery-failure
   claim. Of **4,865** qdetail points, `a0nc_any` is zero at **16**, partial at
   **2,600**, and prints **1.0000 at 2,249**. Candidate rules often cover the
   sampled class, but their best recorded quality never exceeds **0.823**, below
   `theta_r=0.9`. The deficit is reliable coverage, not absence of candidates.
   These aggregates do not identify whether candidates persist, receive conflicting
   updates, generalize incorrectly, or disappear before becoming reliable.
4. **Confirmed at k=70.** The hypothesis was that under the canonical encoding a
   wrong answer leaves the perception unchanged, so its rule must anticipate
   identity — every classifier's default effect, which has to be *narrowed*, whereas
   correct-answer rules are built directly by ALP's unexpected case. Running k=70
   under `--encoding outcome`, where both answers change the perception, **removes
   the starvation entirely and solves the problem 3.8x faster**: 4,680,000 trials
   against 17,880,000, at the same final structure (276 vs 277 reliable rules, spec
   7.01 vs 7.04). Under the canonical encoding `a1_wrong` sits at exactly 0.0000
   from 120 k to 5.88 M trials; under `outcome` all four classes climb together from
   the start. The starved class is an **artifact of the encoding**, not a limit of
   the learning mechanism. Results under `outcome` are **not comparable to the
   multiplexer literature** — it is a different problem.
5. Under `outcome`, k=135 reaches sampled knowledge 1.0 on four seeds (42, 43,
   45, 46), in 30.2–55.6 M trials; seed 44 was cancelled. At k=70 the cost is
   mixed: seed 43 goes from 17.8 M to 62.3 M. Encoding changes the cost distribution;
   canonical epsilon-1 success shows it is not required for k=135 closure.
6. `epsilon=1` removes the greedy preference for change-anticipating rules and
   reaches knowledge 1.0 on seed 43 at 427.92 M trials. Seed 42 also fills one
   previously starved class, but stops at 301.4 M, before the comparable second
   class opens on seed 43. This supports an exploration dependence; it does not
   show a seed-specific inability to respond or guarantee eventual success.

The historical diagnostic `correct` counts standard complete-address rules only.
It is kept unchanged for archive comparability. Correct specificity-`a+1` rules
can omit an address bit when both possible selected data bits agree (MPX-6:
address `0#`, data `00##`, answer 0). Zero `correct` or `addr_full` does not prove
that no correct candidate exists.

## 4. Instrumentation available (all off by default)

| Flag | Emits | Answers |
|---|---|---|
| `--log-trajectory` | `traj:` | the S-curve: knowledge, reliable count, specificity, population |
| `--log-diagnostics` | `diag:` | population-wide specificity, quality spread, mark density, experience, address-bit enrichment against a blind-choice baseline |
| `--log-coverage` | `cover:` | knowledge split into four action × correctness classes, plus matched-but-mispredicted |
| `--log-quadrant-detail` | `qdetail:` | per class, the share covered by **any** classifier and the best quality among them — separates "never created" from "never reliable" |
| `--log-accuracy` | `acc:` | how often greedy choice answers correctly — the metric the literature reports |

Experiment knobs: `--u-max derived|<int>`, `--alp-gen-variant pyalcs|butz`,
`--agent acs2|acs2er`, `--er-{buffer-size,min-samples,samples-number}`,
`--encoding flip|outcome`, `--epsilon <f64>`, `--eval-interval`,
`--rss-cap-gb <f64>`, `--checkpoint-path <p>` / `--checkpoint-every <n>`.

### The archive — where results live, and what makes a row reproducible

`reports/mpx_{verdicts,trajectory,diagnostics}.csv` is the machine-readable archive
(79 verdict rows, 68,340 trajectory points after the 2026-09-09 archive sync), rebuilt by
`tools/parse_mpx_logs.py`. `reports/MPX{70,135,264}_runs.md` is the same data rendered
to be read, by `tools/summarize_mpx.py --size <k>` — grouped by arm, in-flight runs
included, coverage classes beside knowledge.

Rebuild after pulling from the cluster; `tools/README.md` carries the exact commands.

Three traps the archive now guards, all of which had already cost something:

- **A log did not record its encoding.** Two runs differing only in `--encoding` were
  indistinguishable. Fixed: the header records `encoding` and `eval_interval`, and the
  SLURM wrapper prepends commit, job id, tag and argv. For older logs the value is
  reconstructed and `encoding_source` says how — only `header` is a record, everything
  else is inference. It caught `probe264`, which every filename rule would have
  mislabelled `flip`.
- **`peak_rss_gb = 0` on a cluster log means unmeasured**, not zero — the `ru_maxrss`
  bug, fixed in `f93b71e`.
- **Plots spliced arms.** `plot_mpx.py` grouped by seed alone; at k=135 seed 42 spans
  fourteen arms, so a "seed 42" curve was several unrelated runs concatenated. It now
  takes encoding, epsilon, `u_max`, agent, GA and replay filters, and refuses both
  mixed arms and independent runs sharing a seed. Use `--source` and `--block`.

`epsilon` is backfilled to 0.8 where absent, and that one is safe: the flag and the
header field landed in the same commit (`0bc6bd0`), so those runs had no other
reachable value.

**`--rss-cap-gb` matters more than it looks** (`bd88cc2`). The RSS ceiling used to be a
compile-time 5.6 GB constant. It never fired on the cluster while `ru_maxrss` was
misread, so it was invisible; now that `f93b71e` reads it correctly the cap is live and
would abort a k=264 run long before its memory curve is measurable. The default is
still 5.6 GB. New `--isolate-repeats` gives each repeat a fresh process and RSS peak;
`--strict-resource-limits` checks again after evaluation. Both default off for legacy
compatibility. Raise the cap explicitly for
anything at 264 bits.

**`accuracy` vs `knowledge` matters for reporting.** Knowledge demands anticipating
every transition, including the null ones a wrong answer produces; choosing correctly
needs only the change-anticipating side. The literature (ExSTraCS accuracy, ACS2ER
reward) scores task performance, so knowledge is a strictly harder criterion and the
numbers are not comparable without accuracy alongside.

## 5. Cluster

`ssh -i ~/.ssh/id_rsa_wcss alelys2099@ui.wcss.pl`, SLURM, partitions
`bem2-cpu-short` (3 d) / `bem2-cpu-normal` (21 d), MaxJobs 150. **Build
`--target x86_64-unknown-linux-musl`** — the login node's glibc is newer than the
compute nodes'. Run from the git clone `~/acs2-rust-repo`.

Submit: `sbatch [--mem=32G] --export=ALL,TAG=<tag>,U_MAX=..,AGENT=..,ENCODING=..,EPSILON=..,EVAL_INTERVAL=.. slurm/mpx_reach.sh <size> <seed> <time_cap_secs> [extra flags]`

Status: `./slurm/mpx_status.sh`. Output lands in `~/mpx_runs/`, **outside** the
checkout — writing into the tracked `reports/` made every `git pull` collide with a
running job.

### Budget — read this before submitting anything

Grant: 5000 CPU-hours, 2026-07-23 to 2027-07-24, 200 GB disk (27 MB used; disk is a
non-issue).

**Measured 2026-09-09: 4457 h consumed, 543 h left, 476 h of it already committed to
running jobs — roughly 67 h genuinely free.** A new grant application is being filed;
until it lands, submit nothing that is not already planned. The authoritative reading is

```
sshare -U -u alelys2099 -o RawUsage -n     # CPU-seconds; /3600 for hours
```

which is the **undecayed lifetime total** here — the QOS carries `NoDecay`, confirmed by
`scontrol show config` (`PriorityDecayHalfLife = 00:00:00`). Confirm that before trusting
the number again.

`sacct` **does** work, contrary to what this file said before. It rejects wide date
ranges with `Too wide of a date range in query`; query a month at a time. Retention
starts **2026-08-18**, so it cannot see the earliest runs — summing `CPUTimeRAW` from
that date gives 4264 h, which corroborates the 4367 h lifetime figure.

Two consequences that bite:

- **Burn rate is the whole story.** Jobs are 1 CPU each, so ten concurrent jobs spend
  10 CPU-hours per wall-clock hour. 633 h remaining is **under three days** at that rate,
  not months.
- **The queue was over budget on 2026-09-08.** Summing `squeue`'s `TIME_LEFT` on
  2026-09-08 gives **732 h of remaining commitment against 633 h left.** This prompted
  the cancellations below; use the 2026-09-09 snapshot above for the later balance.

Check both before a batch:

```
sshare -U -u alelys2099 -o RawUsage -n
squeue -u alelys2099 -h -o "%i|%j|%t|%L|%C"
```

**Resolved on 2026-09-08.** Four jobs were cancelled with the user's approval —
`encU9_s43` (pending), `enc135_s44`, `acc135u11`, `acc135u12` — freeing 341 h.
Immediately after those cancellations, commitment was **390 h against 630 h**,
leaving ~240 h. This is a historical snapshot, superseded by the dated reading above. Their final readings are preserved in §6; the logs are archived
on the cluster as `*.cancelled`, which the status script deliberately hides.

### Limits — read them from SLURM, not from the documentation

The user could not find these documented anywhere. They are not; SLURM is the source.

```
sinfo -o "%20P %10l"                       # TIMELIMIT per partition
scontrol show partition bem2-cpu-normal    # MaxTime=21-00:00:00
sacctmgr -n -P show qos name=hpc-alelys2099-1784823245 \
  format=Name,GrpTRESMins,GrpTRESRunMins,MaxWall,Flags
```

- `bem2-cpu-short` 3 d, **`bem2-cpu-normal` 21 d**, `bem2-cpu-interactive` 6 h.
- The 5000 h **is SLURM-enforced**: `GrpTRESMins=cpu=300000` with flags
  `DenyOnLimit,NoDecay`. `NoDecay` is why `RawUsage` is the lifetime total.
  `DenyOnLimit` means exhaustion makes **`sbatch` reject new jobs**; running jobs are
  not killed, because `GrpTRESRunMins` is unset.
- **`slurm/mpx_reach.sh` self-limits to 7.5 days** (`#SBATCH --time=7-12:00:00`, plus
  the 600,000 s internal cap = 6.94 d). The "167 h per long run" figure is our own
  choice, not a cluster constraint — the queue allows **504 h per job**. Three times
  the trials in one job, no checkpointing needed, at three times the budget per job.
- **Lem is available to us and is far larger**: `lem-cpu-normal` has 17,920 CPUs
  against Bem2's 2,304, same 21-day limit. Verified with
  `sbatch --test-only --partition=lem-cpu-normal` — accepted, planned start six days
  out (Lem is contended; Bem2 starts next day). Never used. A one-hour benchmark would
  say whether its cores are faster.

Three things that cost days before:
- **Budget wall-clock generously.** Nodes run packed, so throughput is ~2.7x below the
  M1 and degrades within a run. A run without `CHECKPOINT=on` restarts from zero when it
  is cut off — checkpointing exists since 2026-09-10, but it is opt-in.
- **ACS2ER is memory-bound.** m = 13 at k=70 died OUT_OF_MEMORY at 8.4 GB. Give ER runs
  `--mem=32G`.
- **RSS reporting was broken until 2026-09.** `ru_maxrss` is bytes on macOS and
  kilobytes on Linux; the code assumed bytes, so cluster runs printed `0.00GB` and the
  internal RSS cap never fired. Fixed in `f93b71e`. Peak-memory figures logged before
  that commit are meaningless.

## 6. Experiments in flight

Verified live on 2026-09-10. `./slurm/mpx_status.sh`; logs in `~/mpx_runs/`. **Pull them
into the repo with `./tools/sync_runs.sh --commit`** — nothing does it automatically.

**Two jobs running. Grant: 4529 h of 5000 spent (`RawUsage` 16,305,527 CPU-s), 471 h
left, of which 415 h is already committed to these two.** Roughly 56 h genuinely free,
so submit nothing new until the extension lands.

| Job | State, late 2026-09-10 | Why it matters |
|---|---|---|
| `eps135_s42b` (5856652) | 45.48 M trials, knowledge **0.1800**, 269 reliable, spec 11.14, pop 15 883; 11 d 23 h of wall left (300 h allocation, 1 d used) | **The one that matters.** Seed 42 restarted from zero. At 1 841 trials/s over 24.7 h the allocation projects to ~550 M trials against the 428 M seed 43 needed — and throughput rises as the population condenses. If it closes, the canonical k=135 result is two seeds instead of one. Specificity 11.1 against the ideal 8, falling from 12.5 earlier in the day, is the bloat phase resolving, not a warning sign. |
| `encU9_s42` (5828411) | 29.88 M trials, knowledge **0.2320**, 124 reliable, spec **8.15**; 5 d 7 h left | Climbing steadily — 0.0023 on 09-08, 0.1502 on 09-09, 0.2075 and then 0.2320 on 09-10 — with specificity settled at the ideal. Under the canonical encoding this configuration produced a hard zero across 105.6 M trials. So the encoding, not `u_max`, was the binding constraint at 135 bits. |

Neither job is checkpointed: both predate the feature and restarting them to gain
resumability would throw away a month of trials. The first chained run is k=264.

### The replay-volume question is unanswered, and the naive approach is unaffordable

`er70_m8`, `er70m8b` and `er70m13b` all ended **TIMEOUT at 7 d 12 h**, not out of memory,
and all three logs contain **only the header**: not one evaluation point in ~180 h each.
At `--eval-interval 60000` they never reached their first measurement, so throughput was
under 0.09 trials/s against ~6.9 for m=3. **They cost roughly 540 CPU-hours and produced
no data.**

That is itself worth stating in the thesis: scaling replay by *volume* is computationally
prohibitive well before it becomes informative, which is a direct empirical argument for
prioritising *which* samples are replayed rather than *how many*. Before retrying, drop
`--eval-interval` by an order of magnitude so something is recorded, add
`--log-coverage`, and consider k=37 where m=8 is tractable.

Finished and archived: `eps135_s43` (SUCCESS, the headline), `enc135_s45` (SUCCESS, 4th
`outcome` seed), `probe264` (the k=264 measurement). Cancelled with readings preserved:
`acc135u11`, `acc135u12`, `enc135_s44`, `encU9_s43`.

## 7. Claims corrected during the session — do not re-inherit them

These corrections include overgeneralising finite runs and misreading diagnostic
columns. Check the raw evidence and distinguish candidates from reliable rules.

- **Mark density is not the discriminator.** It looked decisive at 390 k trials; by
  10.35 M the solvable k=70 run had risen to 0.914 against k=135's 0.935.
- **The 40,000 s cap on k=70 seeds 43/44 was not too small.** Both would have
  finished inside it (13,692 s and 28,988 s). The restart was insurance those two
  did not need; only seed 46 (46,266 s) required the larger budget. A mid-climb lag
  ratio does not extrapolate, and condensation makes trials cheaper as a run
  proceeds.
- **ACS2ER with m=1 is not measurably slower than ACS2.** One k=37 comparison said
  it was; with more data k=70 seed 42 has ER *faster* (12.12 M vs 17.88 M) and seed
  43 slower (22.86 M vs 17.82 M). At the measured 3.73x seed variance, two seeds
  settle nothing. **This wrong claim is in the email already sent** — see
  `SUPERVISOR_NOTES.md`.
- **It is not one starved class but two.** Seed 42 at `u_max`=11 filled one of the
  two wrong-answer classes, which is why it reads 0.75; seeds 43–45 have both at
  zero. Seed 42 is the outlier. The sent email says "the fourth class", which
  understates it.
- **`epsilon = 1` does not "work on seed 43 and do nothing for seed 42".** Claimed
  during this session from seed 42's final coverage line (`a0_nochange` = 0.0000) read
  as a property of the seed. It is a mid-run state: seed 43 sat at exactly 0.0000 on
  *its* second class until 404.5 M trials and closed at 427.9 M, while seed 42 was cut
  at 301.4 M. Similar class-level pattern, different stopping point; eventual closure is not guaranteed. The §1 table has the numbers.
  Third time a mid-run reading has been reported as a ceiling in this project.
- **A single seed at k=135 under `outcome` is not "the encoding failing".** Seed 44 is
  in a bloated regime, but seeds 42, 43, 45 and 46 all close. Report it as 4 of 5.
- **The grant was not two-thirds spent, it was seven-eighths spent.** This file carried
  "~3250 h of 5000, `sacct` returns no accounting". Both halves were wrong: `sacct`
  works on a narrow date range, and the real figure on 2026-09-08 is **4367 h**. The
  estimate-from-wall-times method was low by a third. Measure, do not extrapolate —
  §5 has the commands.
- **The `outcome` encoding does not guarantee closure on every seed.** Four seeds
  reached SUCCESS, including seed 45 at 55,560,000 trials; seed 44 was cancelled
  before success. Neither inevitable convergence nor permanent failure is established.
- **`epsilon = 1` at k=135 seed 43 does not cap at 0.7481.** It reached sampled
  knowledge 1.0000 at 427,920,000 trials. The earlier 0.9685 in-flight status is obsolete.
- **Replay volume was never measured, despite three jobs and ~540 CPU-hours.** m=8 and
  m=13 at k=70 timed out before their first evaluation point, leaving header-only logs.
  Any statement about whether more replay reaches the starved class is unsupported.
- **The starved class is not devoid of classifiers of any quality.** At seed 42,
  `u_max=11`, only 16/4,865 qdetail points show zero candidate coverage; 2,600 are
  partial and 2,249 print 1.0000. Best recorded quality is 0.823, below 0.9.
  Withdraw the discovery-failure claim and target the candidate-to-reliable gap.
- **Accuracy did not stay at 1.0000 for 325.8 M trials.** Only 769/2,715 logged
  points print 1.0000, and four-decimal rounding can hide one error in 50,000.
- **A complete address is not necessary for every correct minimal MPX rule.** The
  historical `correct` diagnostic is a narrower structural proxy, not a completeness
  test for correct candidates.
- **The k=135 ACS2ER runs were cancelled too early.** They looked dead at one
  evaluation point per day, but the eval interval had been sized for ACS2's
  throughput. The one point they did produce showed ER touching the starved class
  (`a0_nochange` = 0.0136 where ACS2 sits at exactly zero) — with
  `matched_but_wrong` = 4549, so possibly with bad rules. Relaunched as `erfine_*`.

## 8. What to do next

The supervisor approved going after k=264 ("Koniecznie!") and asked to share the
implementation as a base for his other students, which the user agreed to — so **the
repository is a public teaching artefact as well as a thesis codebase**. Keep it that
way: README and `reports/MPX_final.md` are what an outsider reads first.

The work is sequenced, and step 1 does not need the cluster. That matters, because only
~56 h of grant are genuinely free until the extension lands.

### Step 1 — checkpointing — DONE (2026-09-10)

The blocker it removes, measured rather than argued: one k=264 seed is 700-4500
CPU-hours against a **504 h** hard queue limit, so no single job can finish one.

`--checkpoint-path` saves the population, both RNG streams (agent **and**
environment), the trial and ALP clocks, `trials_since_eval`, the accumulated
wall-clock and the peak trackers; `--checkpoint-every` adds periodic saves.
`slurm/mpx_reach.sh` takes `CHECKPOINT=on` and derives the path from size, seed and
tag. `docs/ARCHITECTURE.md` carries the design, the file format and the two decisions
that are not obvious from the code (wall-clock has two readings; the identity gate
excludes the stopping limits).

**Reviewed and corrected (2026-09-10).** The independent review of §8 step 2 found five
real defects; all are fixed and pinned by regressions. The one that mattered: a resource
cap is checked *before* the evaluation block, so a job could stop on the very batch a
measurement was due and the resumed run shifted that measurement and every later one.
Reproduced at k=20 — evaluations at 1500/2500/3500 against 1000/2000/3000 and **SUCCESS
reported at 66,500 trials instead of 67,000**. A checkpoint moved the headline metric.
`docs/ARCHITECTURE.md` lists the rest and the two challenges accepted.

**The acceptance test is determinism across the cycle** and it is in
`acs2-bench/tests/reach_regressions.rs`: three processes — whole, first half, resumed
half — for both ACS2 and ACS2ER, asserting identical trajectories *and* a byte-identical
final checkpoint. It was verified by sabotage rather than trusted
because it is green: sixteen mutations of the saved state, fifteen caught — the
sixteenth is `ee`, and that one *cannot* be caught, see below. Gates: **98 Rust tests**,
26 Python tests, P9 maze learning columns byte-identical, and `mpx_reach` output without
the flag compared line for line against the pre-checkpointing binary at k=20 over 102
learning lines.

Verified end to end outside the test harness too: a k=20 run split across **twelve**
processes by a 1 s wall cap reproduces the uninterrupted run's 134 measurements exactly
and closes at the same 67,000 trials, with the tail segments restating the verdict;
a k=264 checkpoint round-trips at 12.5 MB for 8,107 classifiers.

**Submit a chain with `--dependency=afterany`.** One checkpoint is one learning state and
nothing locks it; two segments running at once corrupt each other. `README.md` has the
loop.

Two things it deliberately does **not** do, both recorded in `ARCHITECTURE.md`:

- `ee` is serialised but cannot be covered by the test — it is written and never read,
  because PEE is not implemented.
- The archive stitching that a chained run needs was found and built with it;
  `ARCHITECTURE.md` has the rules and the mistake that reads as correct.

### Step 2 — independent review of the checkpointing — DONE (2026-09-10)

Run on a second model, read-only, against a self-contained brief and a code bundle. It
ran three times: five defects, then five, then two — **twelve in total and not one false
positive**. Every finding was verified against the code before acting and the worst was
reproduced by measurement first. Rounds two and three found most of their defects in code
written *between* rounds, which is the argument for reviewing the fixes and not only the
original change. One thing was **not** acted on as asked: it wanted an ownership lock on
the checkpoint, and a stale lock left by a killed job would block exactly the disaster
recovery that `a_periodic_checkpoint_outlives_a_killed_process` proves works. Job
dependencies sequence the chain instead.

The procedure below is what was followed and is worth following again.

Hand it to a **second agent** before it is trusted. This process ran twice on this
repository and both times found real defects, so it is established practice, not
ceremony:

1. Give the reviewer a **read-only** brief: change nothing, run nothing, write a report.
   Tell it what is deliberate so it does not report settled decisions as findings.
2. **Verify its findings yourself** against the code and the archive before acting.
   The last review was right about nine of ten checked claims and wrong about one (it
   claimed `slurm/mpx_status.sh` would exit on a header-only log; `set -euo pipefail`
   is not inherited by the remote `bash -s` in its heredoc).
3. Send back a fix brief that says which findings you confirmed, **and where you
   disagree with it and why** — that is where the value is. Ask for a report at the end.
4. Verify the fixes too. Last time the reviewer's own regression tests were not wired
   into any `Cargo.toml` and never ran.

For checkpointing specifically, point the reviewer at the determinism property: a
save/restore that is subtly non-identical will pass a casual reading.

### Step 3 — when the new WCSS grant lands

Two things at once:

- **Watch what is already running** (§6). `eps135_s42b` is the one that matters; if it
  closes, the canonical k=135 result is two seeds instead of one. `./tools/sync_runs.sh
  --commit` after anything finishes.
- **Start k=264** under `--encoding outcome`, `u_max = 12` (the `a + 4` analogue of the
  11 that works at 135), with `CHECKPOINT=on` and a **small `--eval-interval`**. The
  large-`m` replay jobs died having recorded nothing because their first evaluation point
  was never reached (§6) — do not repeat that at a size where a job costs 504 h. Size
  `--checkpoint-every` so writes stay rare: at k=264 a checkpoint is ~1.5 KB per
  classifier, 12.5 MB at 8,107 of them.

The grant application asks for 10,000 CPU-hours: ~700 h to finish current work, ~1000 h
to close k=135 canonically on three seeds, ~4000–5000 h for k=264 on three seeds with
controls, ~2000 h for replay. The original application's RAM figure needs correcting —
it declared `< 1 GB`, ACS2ER measured 8–32 GB, but the k=264 probe measured only
**0.86 GB**, so the driver is ACS2ER, not problem size. Wall-time was declared `>= 48h`
against 21 days actually needed.

### Step 4 — the thesis core: prioritised experience replay

The contribution is not replay itself — ACS2ER exists and its limits are measured — but a
**prioritisation criterion aimed at the measured gap**: candidates cover a transition
class yet stay below the reliability threshold (§3). Measure candidate quality and
retention as well as reliable coverage, and compare at matched learning applications.

Two things the data already says, and both shape this:

- **Volume does not work.** Three jobs at m = 8 and 13 cost ~540 CPU-hours and recorded
  nothing (§6). Scaling replay by quantity becomes prohibitive before it becomes
  informative — which is itself the empirical argument for prioritising *which* samples.
- **The mechanism is not established.** Whether candidates fail to become reliable
  through insufficient reinforcement, conflicting updates, or replacement is unknown.
  Do not pick a criterion before measuring which of those it is.

## 9. Working with the user

- Polish in conversation; code, comments, commits and docs in English.
- He reads the code and catches real things — the `port`-vs-`implementation` framing,
  the ambiguity in "quadrant", `specyficzność` vs "condition length", the need for an
  accuracy metric. Treat his questions as signal, not as requests for reassurance.
- He stopped a run that was about to execute an MPX experiment on his laptop. Bench
  gates (tests, maze P9) are fine locally; **experiments go to the cluster**.
- Cancelling jobs, new experiment phases and anything sent to the supervisor are his
  calls. Execution inside an agreed phase is yours.
- The supervisor thread is kept locally in `docs/SUPERVISOR_CORRESPONDENCE.md`,
  **gitignored on purpose** — the repository is public. Do not commit it, quote it
  into tracked files, or recreate it. Its operative directives are already reflected
  in §8 here and in `PROJECT_CONTEXT.md`.

## 10. Escalating to a stronger model

The user has access to a stronger model and will relay a question to it. Ask for that
when you are genuinely stuck — but the bar matters in both directions.

**Escalate when:**
- Evidence contradicts itself and you cannot reconcile it.
- A decision is expensive to get wrong — days of cluster time, a core change that would
  break a gate, or anything going to the supervisor — and the codebase does not settle it.
- You are choosing between designs with real trade-offs and no measurement can decide,
  because the trade-off is about what the thesis should *argue*, not about what the code
  *does*.

**Do not escalate when:**
- A measurement would answer it. This project's whole method is measure-then-conclude;
  nearly every wrong turn in §7 came from reasoning where a run would have settled it.
  Run the experiment.
- Reading the source would answer it. The code is the authority on what the code does.
- You have a defensible recommendation and want it validated. Give the recommendation,
  say what you are unsure about, and let the user decide.

**The escalation prompt must stand alone.** The other model has none of this context.
Include: the specific question, the numbers behind it, what you already tried and ruled
out, and what you think the answer is and why you are not confident. Without the
"already ruled out" part you will get back suggestions this project has spent days
eliminating. Write it to the scratchpad and hand the user the file.

## 11. Standing rules

**`CLAUDE.md` at the repo root carries the operational rules** and is loaded into
every session automatically — the cluster sync, the gates, and the mid-run-reading
failure mode. This file carries the research state; that one carries the habits.


Idiomatic Rust, SOLID, no code comments, English identifiers and commit messages,
injected RNG. Anything touching the measured path goes behind a flag with defaults
preserving current behaviour. Commit and push to `feature/checkpointing` after each completed
group. Measurements live in `reports/`, review/fix reports in `scratchpad/`, and the
implementation record in `docs/ARCHITECTURE.md`.
Ask the user only for scope decisions — new experiment phases, supervisor
communication, cancelling running jobs; execution decisions are yours.
