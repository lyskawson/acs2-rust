# Agent onboarding — ACS2 Rust, multiplexer scaling and experience replay

Handoff for a fresh session. Read `docs/PROJECT_CONTEXT.md` for what the project is,
`docs/ARCHITECTURE.md` for how it is built, and `reports/MPX_final.md` for the
scientific narrative. This file carries only the **live state**.

## 1. Where the research stands

**MPX-70 is solved** at knowledge = 1.0 on all five seeds tried (17.8 M–66.4 M
trials, 268–277 reliable rules at the ideal specificity 7). Published ACS/ACS2
results stop at 20–37 bits.

**MPX-135 is solved.** Two results, and the distinction between them is the whole
story:

- **By the criterion the literature uses, ACS2 solves it under the canonical
  encoding.** At `u_max` = 11 the agent reaches **answer accuracy 1.0000** while
  `knowledge` sits at its 0.7499 ceiling. Knowledge additionally demands
  anticipating the *null* transitions a wrong answer produces, which is a strictly
  harder, anticipation-specific criterion — ExSTraCS scores classification
  accuracy, ACS2ER scores reward. Report accuracy alongside knowledge or the result
  reads as weaker than it is.
- **By the anticipatory criterion, the encoding change closes it on most seeds.**
  Under `--encoding outcome`, knowledge reaches **1.0000 on three of five seeds** —
  43.2 M (s42), 46.8 M (s43) and 30.2 M (s46) trials, ending at 539 / 539 / 535
  reliable rules and specificity 8.00–8.01 (ideal `a+1`). Seed 45 sits at 0.9981 and
  is closing. **Seed 44 is not**: it is in a bloated, over-specialised regime
  (pop 64 579, spec 10.92, ~9x slower) and will not converge in its budget. Do not
  repeat the earlier "the encoding makes it reproducible" claim — see §7.

Canonical-encoding ceilings, four seeds, all TIME-LIMITED: 0.7499 (seed 42, which
filled one wrong-answer class) and 0.4980 / 0.4980 / 0.4918 (seeds 43–45, which
filled none). Seed 42 is the outlier — see §3.

**`epsilon = 1` may close k=135 on the canonical encoding — this is the live headline.**
Seed 43 caps at 0.4980 canonically; with the greedy bias removed it has reached
**0.9685 at 414 M trials and is still climbing**, with both wrong-answer classes
filling (0.9942 and 0.8796). Canonical-encoding results are the ones comparable to
the multiplexer literature, so closing this beats the `outcome` result scientifically.
The run dies on its wall limit within ~30 h and there is no checkpointing. Seed 42
under the same setting is stuck at 0.7350, so it is not a universal fix.

**ACS2ER exists and is validated** differentially against pyalcs (`p11_acs2er`).
First comparisons say uniform replay trades compute for episodes rather than
improving efficiency; at matched learning applications no advantage is measurable.

## 2. Hard invariants (do not violate)

- Maze path untouched: `u_max = 100000` on the maze config keeps the ALP-gen branch
  dead. Before any core change lands: `cargo test --workspace --release` green
  (**73 tests**) and the P9 maze learning columns byte-identical to
  `reports/bench_rust.csv`.
- Determinism from an injected RNG, verified cross-architecture. **Trials-to-success
  is the machine-independent metric**; wall-clock is machine-specific colour.
- Every diagnostic is read-only over the population and off by default. Proof that
  this holds: with and without instrumentation, seed 42 solves k=70 at exactly
  17,880,000 trials.
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
3. Scanning the whole population, not just reliable rules: at `u_max` = 11 the
   starved class has **no classifier of any quality** — rules are never created
   there. At `u_max` = 12 both failure modes appear side by side: one wrong class
   fully covered but stuck at best quality 0.670, the other empty.
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
5. **Confirmed at k=135 as well**, on three seeds: 43.2 M, 46.8 M and 30.2 M trials
   to knowledge 1.0, ending at 539/539/535 reliable rules and specificity 8.00–8.01.
   Note the k=70 picture is more mixed — the encoding is not uniformly faster there
   (seed 43 goes 17.8 M -> 62.3 M) — so at 70 bits it changes the cost distribution
   while at 135 it changes whether the problem closes at all.
6. `epsilon = 1` lifts a seed clean out of the ceiling (43: 0.4980 -> 0.9685 and
   rising), which was not expected: the greedy branch selects among change-anticipating
   classifiers, i.e. correct answers under this encoding, so it under-visits the
   starving class. The starvation is therefore **partly an exploration artifact as well
   as an encoding artifact** — two independent interventions each relieve it. Seed 42
   does not respond the same way (0.7350), so the two are not interchangeable.

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
`--rss-cap-gb <f64>`.

**`--rss-cap-gb` matters more than it looks** (`bd88cc2`). The RSS ceiling used to be a
compile-time 5.6 GB constant. It never fired on the cluster while `ru_maxrss` was
misread, so it was invisible; now that `f93b71e` reads it correctly the cap is live and
would abort a k=264 run long before its memory curve is measurable. The default is
still 5.6 GB, so every earlier run's behaviour is unchanged — raise it explicitly for
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

**Measured on 2026-09-08: 4367 h consumed, 633 h left.** The authoritative reading is

```
sshare -U -u alelys2099 -o RawUsage -n     # CPU-seconds; /3600 for hours
```

which is the **undecayed lifetime total** here — `scontrol show config` reports
`PriorityDecayHalfLife = 00:00:00` and `PriorityUsageResetPeriod = NONE`, so nothing
ages out of it. Confirm the decay settings before trusting the number again.

`sacct` **does** work, contrary to what this file said before. It rejects wide date
ranges with `Too wide of a date range in query`; query a month at a time. Retention
starts **2026-08-18**, so it cannot see the earliest runs — summing `CPUTimeRAW` from
that date gives 4264 h, which corroborates the 4367 h lifetime figure.

Two consequences that bite:

- **Burn rate is the whole story.** Jobs are 1 CPU each, so ten concurrent jobs spend
  10 CPU-hours per wall-clock hour. 633 h remaining is **under three days** at that rate,
  not months.
- **The live queue is already over budget.** Summing `squeue`'s `TIME_LEFT` on
  2026-09-08 gives **732 h of remaining commitment against 633 h left.** Something has to
  be cancelled or the grant runs dry mid-queue.

Check both before a batch:

```
sshare -U -u alelys2099 -o RawUsage -n
squeue -u alelys2099 -h -o "%i|%j|%t|%L|%C"
```

**Resolved on 2026-09-08.** Four jobs were cancelled with the user's approval —
`encU9_s43` (pending), `enc135_s44`, `acc135u11`, `acc135u12` — freeing 341 h.
Commitment is now **390 h against 630 h**, so there is ~240 h of headroom and no
submission deadline. Their final readings are preserved in §6; the logs are archived
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
  M1 and degrades within a run. There is no checkpointing; a cut-off run restarts from
  zero.
- **ACS2ER is memory-bound.** m = 13 at k=70 died OUT_OF_MEMORY at 8.4 GB. Give ER runs
  `--mem=32G`.
- **RSS reporting was broken until 2026-09.** `ru_maxrss` is bytes on macOS and
  kilobytes on Linux; the code assumed bytes, so cluster runs printed `0.00GB` and the
  internal RSS cap never fired. Fixed in `f93b71e`. Peak-memory figures logged before
  that commit are meaningless.

## 6. Experiments in flight

Eleven jobs, `bem2-cpu-normal`, 600,000 s internal cap. `./slurm/mpx_status.sh`; logs in
`~/mpx_runs/` as `slurm_mpx<size>_s<seed>[_<TAG>].out`.

Readings taken 2026-09-08 — the two marked **new** move claims made elsewhere in this file.

| Job | Live reading | Verdict so far |
|---|---|---|
| `eps135_s43` | knowledge **0.9685** at 414.0 M trials, 518 reliable, spec 8.06; classes 0.9942 / 1.0 / 0.8796 / 1.0 | **new — the most valuable job in the queue.** `epsilon = 1` under the **canonical** encoding is filling *both* wrong-answer classes and still climbing. If it closes, k=135 is solved on the literature-comparable encoding. ~30 h of wall-time left and no checkpointing, so it will most likely die short of 1.0. |
| `eps135_s42` | 0.7350 at 252.2 M, `a0_nochange` still 0.0000 | `epsilon = 1` does not lift every seed. Seed 42 is stuck at the same ceiling. |
| `acc135u11` | knowledge 0.7499, **accuracy 1.0000** held over 325.1 M trials | Settled. Nothing more to learn from it. |
| `acc135u12` | knowledge 0.4822, accuracy 0.9854 | Accuracy 1.0 is a property of `u_max` = 11, **not** of the canonical ceiling generally. |
| `enc135_s45` | 0.9981, 528 reliable, spec 8.01, accuracy 1.0000 | Essentially solved; 4th `outcome` seed. |
| `enc135_s44` | 0.1583 at 4.8 M trials, **pop 64 579, spec 10.92**, 17 trials/s | **new — a counterexample.** Under `outcome` this seed is in a bloated, over-specialised regime, ~9x slower than seed 45. It will not converge in its remaining 100 h. |
| `encU9_s42` | 0.0023, **5 reliable**, spec 11.80, pop 24 281 at 3.84 M trials | The control. Canonical encoding at `u_max` = 9 gave a hard zero across 105.6 M; `outcome` does create rules, but it is not converging. Preliminary read: **the encoding alone does not rescue `u_max` = 9**, so the `u_max` sweep was not merely treating a symptom. |
| `encU9_s43` | PENDING | Second seed of the control above. |
| `er70_m8`, `er70m8b`, `er70m13b` | running | Does replay *volume* help the starved class or only the easy ones? |
| `probe264` (5851940) | submitted 2026-09-08, `bem2-cpu-short`, 12 h internal cap, 64 GB, `--rss-cap-gb 56`, `--encoding outcome --u-max 12` | **The measurement the WCSS application is waiting on.** Real throughput and peak RSS at k=264, replacing the extrapolation from a 500-trial probe. Costs at most 13 h. |

**Cancelled 2026-09-08, final readings preserved here** (logs archived as `*.cancelled`):

| Job | Last reading |
|---|---|
| `acc135u11` | 325.8 M trials, knowledge 0.7499, **accuracy 1.0000**, 392 reliable, spec 8.00; classes 0.0000 / 1.0 / 1.0 / 1.0 |
| `acc135u12` | 169.6 M trials, knowledge 0.4821, accuracy 0.9821, 255 reliable, spec 8.16; both `nochange` classes 0.0000 |
| `enc135_s44` | 4.8 M trials, knowledge 0.1583, accuracy 0.6495, 423 reliable, **spec 10.92, pop 64 579**, `matched_but_wrong` 144 |
| `encU9_s43` | never started |

## 7. Claims corrected during the session — do not re-inherit them

Every one of these came from generalising a single seed or a short window. The
pattern is the failure mode to watch for.

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
- **The grant was not two-thirds spent, it was seven-eighths spent.** This file carried
  "~3250 h of 5000, `sacct` returns no accounting". Both halves were wrong: `sacct`
  works on a narrow date range, and the real figure on 2026-09-08 is **4367 h**. The
  estimate-from-wall-times method was low by a third. Measure, do not extrapolate —
  §5 has the commands.
- **The `outcome` encoding does not make k=135 reproducible.** §1 said the seeds end at
  "exactly 539 reliable rules and specificity 8.00" and that the encoding "makes it
  reproducible". That held for seeds 42, 43 and 46, and seed 45 is joining them — but
  **seed 44 is in a bloated regime** (pop 64 579, spec 10.92, 17 trials/s) and will not
  converge. Four seeds agreeing is not five. The honest claim is that `outcome` closes
  the problem on most seeds, with one seed so far behaving differently.
- **`epsilon = 1` at k=135 seed 43 does not cap at 0.7481.** That was a mid-run reading
  reported as a ceiling — the same mistake as the 0.75 ceiling at k=70. It has since
  passed **0.9685** under the canonical encoding and is still rising. It is now the
  strongest live result in the project, because canonical-encoding numbers are the ones
  comparable to the literature.
- **The k=135 ACS2ER runs were cancelled too early.** They looked dead at one
  evaluation point per day, but the eval interval had been sized for ACS2's
  throughput. The one point they did produce showed ER touching the starved class
  (`a0_nochange` = 0.0136 where ACS2 sits at exactly zero) — with
  `matched_but_wrong` = 4549, so possibly with bad rules. Relaunched as `erfine_*`.

## 8. What to do next

The supervisor wants to attack a larger multiplexer and asked whether to reserve
cluster resources. Answering that responsibly needs three things first, and this is
the live decision:

1. **Sizes are not continuous.** `k = a + 2^a`, so after 135 the next is **264**, then
   521. At 264 a classifier is 7504 B against 3896 B, and a complete solution needs
   1024 reliable rules against 512. `264` is now wired into the dispatch (`f93b71e`);
   it previously panicked. A 500-trial probe on the M1 gives ~14 trials/s, 0.35 GB and
   a population of 8107.
2. **Measuring is done — `probe264` is running** (§6), so the application's numbers
   stop being extrapolations from 500 trials. This was decoupled from checkpointing:
   a 12-hour measurement needs no save/restore, only the real run does.
3. **Checkpointing is still the blocker for the real run.** Extrapolating, one k=264
   seed is 700–1200 CPU-hours — beyond the 21-day queue limit, so a cut-off run loses
   everything. Save/restore of the population is the enabling change. It touches the
   core, so it goes behind a flag with the P8/P9 gates intact. Note the queue allows
   504 h per job against the 167 h we currently take (§5), which shortens how far
   checkpointing has to stretch.
4. **The WCSS application.** Drafted in the scratchpad, waiting on `probe264`. The ask
   is 10,000 CPU-hours against ~8000 h of costed work. Four declarations in the
   original application are now demonstrably false and must be corrected, not just the
   hours: RAM (`< 1 GB` against 8–32 GB measured and one OOM at 8.4 GB), per-run time
   (`2–48 h` against 167 h routinely), queue (`>= 48 h` against 21 days needed), and
   SLURM Job Arrays (declared, never used). Unknown and worth one question to WCSS
   support: whether this is an increase to the existing service, which runs to
   2027-07-24, or a fresh application.

After that, the thesis core: prioritised experience replay. The contribution is not ER
itself (ACS2ER exists and its limits are measured) but a **prioritisation criterion
targeted at the measured gap** rather than a generic TD-error rule from deep RL.

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

Idiomatic Rust, SOLID, no code comments, English identifiers and commit messages,
injected RNG. Anything touching the measured path goes behind a flag with defaults
preserving current behaviour. Commit and push to `feature/mpx` after each completed
step. Reports live in `reports/`, implementation record in `docs/ARCHITECTURE.md`.
Ask the user only for scope decisions — new experiment phases, supervisor
communication, cancelling running jobs; execution decisions are yours.
