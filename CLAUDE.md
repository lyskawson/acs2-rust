# ACS2 Rust — instructions for agents

Read `docs/AGENT_HANDOFF.md` first. It is the live state of the research: results,
what is running on the cluster, which claims were corrected, and how the user works.
`docs/PROJECT_CONTEXT.md` says what the project is, `docs/ARCHITECTURE.md` how it is
built, `reports/MPX_final.md` the scientific narrative.

**Every document in this repository is written for an agent working on the code**, in
English. Personal study notes, one-off reports and anything written for the user rather
than for the work are kept outside the checkout. Do not add a document here that only a
human learner would read.

## Sync the cluster runs — do this without being asked

```bash
./tools/sync_runs.sh --commit
```

**At the start of every session, and again after any cluster batch finishes.**

A SLURM job writes only to `~/mpx_runs/` on the cluster, outside any checkout, and
nothing pulls it. Until this runs, every result exists in exactly one copy, on an
account whose grant expires 2027-07-24. This has already cost real data: 32 logs
lived only on the cluster, and five were committed as zero-byte stubs — worse than
missing, because the repo looked like it had them.

The script pulls, rebuilds `reports/mpx_{verdicts,trajectory,diagnostics}.csv` and
every `reports/MPX<k>_runs.md`, and lists the solved runs. A run the user remembers
solving that is absent from that list never left the cluster.

Do not skip it because a run is still going: partial trajectories are the live state
and are worth archiving.

## Gates — before any change to the core

```bash
cargo test --workspace --release          # 76 tests, including reach regressions
uv run --project tools python -B -m unittest discover -s tools -p 'test_*.py'  # 13 tests
cargo build --release --bin acs2-bench
./target/release/acs2-bench               # P9 maze: learning columns byte-identical
```

Compare columns 1-7 of `reports/bench_rust.csv` against the committed version. The
last three are wall-clock and are expected to differ between machines; restore the
file afterwards so timing noise is not committed.

Anything touching the measured path goes behind a flag whose default preserves
current behaviour.

One-off reports — code reviews, fix summaries, handover notes for a single task — are
not committed. `reports/` holds measurements; a document *about* the repository is not
one, and this repository is public. Write them outside the checkout and hand the user
the file. The commits and the docs they corrected are the durable record.

## Standing rules

- Conversation in Polish. Code, comments, commits, docs and identifiers in English.
- No comments in code.
- Idiomatic Rust, SOLID, injected RNG, determinism from the seed.
- Commit and push to `feature/checkpointing` after each completed group.
- Experiments run on the cluster, never on the user's laptop — it overheats. Test
  gates locally are fine.
- **Check the grant before submitting anything.** `sshare -U -u alelys2099 -o
  RawUsage -n` gives CPU-seconds used against the 5000-hour grant, and it is
  SLURM-enforced with `DenyOnLimit`: once spent, `sbatch` refuses. §5 of the handoff
  has the commands and the traps.
- Emails to the supervisor are Polish, plain, free of AI phrasing. **The user sends
  them, never an agent.**
- Ask the user only about scope: new experiment phases, cancelling jobs, anything
  going to the supervisor. Execution decisions inside an agreed phase are yours.
- `docs/SUPERVISOR_CORRESPONDENCE.md` is gitignored on purpose — the repo is public.
  Do not commit it, quote it into tracked files, or recreate it.

## The failure mode this project keeps hitting

Reporting a mid-run reading as a ceiling. It has happened three times: the 0.75
plateau at k=70, the "seed 43 caps at 0.7481" claim, and "epsilon = 1 does nothing
for seed 42" — the last two in a single session. A knowledge value that has been
flat for 100 M trials is not converged; at k=135 one seed sat at exactly 0.0000 on a
coverage class until 404.5 M trials and then closed within 23 M.

Before writing that a run has plateaued, check how long the comparable run stayed
flat before it moved. §7 of the handoff lists every claim that had to be withdrawn.
