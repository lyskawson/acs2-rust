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
- **By the anticipatory criterion, it needs the encoding fixed.** Under
  `--encoding outcome`, knowledge reaches **1.0000 on both seeds tried** —
  43.2 M and 46.8 M trials, both ending at **exactly 539 reliable rules and
  specificity 8.00** (ideal `a+1`). Trials-to-success differ by 8%, against the
  3.73x spread seen at k=70 under the canonical encoding: the encoding does not
  merely enable the solution, it makes it reproducible.

Canonical-encoding ceilings, four seeds, all TIME-LIMITED: 0.7499 (seed 42, which
filled one wrong-answer class) and 0.4980 / 0.4980 / 0.4918 (seeds 43–45, which
filled none). Seed 42 is the outlier — see §3.

`epsilon = 1` also lifts a seed a whole class: seed 43 caps at 0.4980 canonically
but reaches 0.7481 with the greedy bias removed.

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
6. `epsilon = 1` also lifts a seed a whole class (43: 0.4980 -> 0.7481), which was not
   expected: the greedy branch selects among change-anticipating classifiers, i.e.
   correct answers under this encoding, so it under-visits the starving class by
   about 10 points.

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
`--encoding flip|outcome`, `--epsilon <f64>`, `--eval-interval`.

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

**The grant is finite and roughly two thirds spent.** 5000 CPU-hours, granted
2026-07-23 to 2027-07-24, 200 GB disk. Estimated consumption is ~3250 h (from run
wall-times; `sacct` returns no accounting for this user, so this is not an official
figure). Long runs cost 167 h each at the 600,000 s cap. **Check the budget before
launching a batch of long jobs.**

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

| Jobs | Question |
|---|---|
| `enc135_s44`, `enc135_s45` | Two more seeds for the k=135 `outcome` result (three already solved). |
| `encU9_s42`, `encU9_s43` | **The most interesting one.** Does the encoding alone rescue the *canonical* `u_max` = 9? At 1.32 M trials it already had 4 reliable rules where canonical encoding gave zero across 105.6 M. If it succeeds, the `u_max` sweep treated a symptom and the encoding was the cause — which would change how the whole `u_max` story is reported. |
| `acc135u11`, `acc135u12` | Accuracy against knowledge at the canonical ceilings. `u11` already reads accuracy 1.0000 at knowledge 0.7499. |
| `eps135_s42`, `eps135_s43` | `epsilon = 1` at k=135. |
| `er70_m8`, `er70m8b`, `er70m13b` | Does replay *volume* help the starved class or only the easy ones? The `b` pair is the `--mem=32G` rerun after m=13 was OOM-killed. |

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
   it previously panicked. A 500-trial probe on the M1 gave ~10 trials/s and 0.42 GB.
2. **Checkpointing is the blocker.** Extrapolating, one k=264 seed is 700–1200 CPU-hours
   — beyond the 21-day queue limit, with no checkpointing, so a cut-off run loses
   everything. Save/restore of the population is the enabling change. It touches the
   core, so it goes behind a flag with the P8/P9 gates intact.
3. **Then measure, then apply.** A few-hour k=264 run gives real throughput and memory,
   which is what the extension request to WCSS should be sized on. The original
   application understates RAM (`< 1 GB`) and wall-time (`>= 48h`) badly; both need
   correcting alongside the hours.

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

## 10. Standing rules

Idiomatic Rust, SOLID, no code comments, English identifiers and commit messages,
injected RNG. Anything touching the measured path goes behind a flag with defaults
preserving current behaviour. Commit and push to `feature/mpx` after each completed
step. Reports live in `reports/`, implementation record in `docs/ARCHITECTURE.md`.
Ask the user only for scope decisions — new experiment phases, supervisor
communication, cancelling running jobs; execution decisions are yours.
