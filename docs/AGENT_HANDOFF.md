# Agent onboarding — ACS2 Rust, multiplexer scaling and experience replay

Handoff for a fresh session. Read `docs/PROJECT_CONTEXT.md` for what the project is,
`docs/ARCHITECTURE.md` for how it is built, and `reports/MPX_final.md` for the
scientific narrative. This file carries only the **live state**.

## 1. Where the research stands

**MPX-70 is solved** at knowledge = 1.0 on all five seeds tried (17.8 M–66.4 M
trials, 268–277 reliable rules at the ideal specificity 7). Published ACS/ACS2
results stop at 20–37 bits.

**MPX-135 is partially solved and diagnosed.** At the canonical derived
`u_max` = a+2 = 9 it produces no reliable rule in 105.6 M trials. Loosening the
limit breaks that: `u_max` = 11 reaches 0.7499, `u_max` = 12 reaches 0.50 on two
seeds. The ceilings are exact fractions because whole **wrong-answer classes** are
never covered — see §3.

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
4. Working hypothesis, **being tested now**: under the canonical encoding a wrong
   answer leaves the perception unchanged, so its rule must anticipate identity —
   every classifier's default effect, which has to be *narrowed*, whereas
   correct-answer rules are built directly by ALP's unexpected case.

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

Two things that cost days before:
- **Budget generously.** Nodes run packed, so throughput is ~2.7x below the M1 and
  degrades within a run. There is no checkpointing; a cut-off run restarts from zero.
- **ACS2ER is memory-bound.** m = 13 at k=70 died OUT_OF_MEMORY at 8.4 GB. Give ER
  runs `--mem=32G`. Its population grows with the replay count and each rule carries
  an N-slot mark.

## 6. Experiments in flight

Nineteen jobs, all `bem2-cpu-normal`, all with a 600,000 s internal cap. Check with
`./slurm/mpx_status.sh`; logs are in `~/mpx_runs/`, named
`slurm_mpx<size>_s<seed>[_<TAG>].out`.

| Tag / jobs | Question it answers |
|---|---|
| `u11cover` seeds 43–45, `cov135u11` | Does the 0.75 ceiling reproduce, or do these seeds cap at 0.50? Three of them already show **both** wrong-answer classes at zero. |
| `u10long` | Is the 10-vs-11 boundary real or a budget artifact? At 94.7 M trials it has 2 reliable rules, so `u_max`=10 is very slow rather than dead. |
| `qdetail_u11`, `qdetail_u12` | Never-created vs never-reliable, per class. Already answered at `u_max`=11: nothing is created. |
| `outcome_u11` seeds 42/43, `outcome` at k=70 | **The supervisor's suggestion.** Does the starvation vanish when a wrong answer also changes the perception? |
| `eps1_u11` seeds 42/43 | Does it vanish when the greedy bias that avoids wrong answers is removed? |
| `acc_u11`, `acc_u12` | Is the task already solved at knowledge 0.75, by the literature's criterion? |
| `erfine_u11`, `erfine_u12` | ACS2ER at k=135, eval interval sized to ER throughput rather than ACS2's. |
| `er70_m8b`, `er70_m13b` | Does replay *volume* help the starved class or only the easy ones? Rerun at `--mem=32G` after m=13 died OUT_OF_MEMORY. |
| `er70`, `er1_*` | Uniform replay against ACS2 at matched learning work. |

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

1. **Read the running experiments** (§6). The `outcome` encoding and `epsilon = 1`
   runs at k=135 test the two competing explanations for the starved class — that
   wrong answers produce no perceptual change, versus that the greedy branch
   under-visits them (it selects among change-anticipating classifiers, which under
   this encoding means correct answers, costing the wrong classes 10 points of
   visits). Accuracy runs say whether the task is already solved at knowledge 0.75.
   The four outcomes are all informative: encoding only, epsilon only, both, neither.
2. **If the encoding fixes it**, the starvation is an artifact of the canonical
   encoding and belongs in the thesis as a mechanism finding — but results under the
   alternative encoding are **not comparable to the multiplexer literature**.
3. **Then the thesis core**: prioritised experience replay. The contribution is not
   ER itself (ACS2ER exists and its limits are known) but a **prioritisation
   criterion targeted at the measured gap** — the starved transition class — rather
   than a generic TD-error rule imported from deep RL.

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
