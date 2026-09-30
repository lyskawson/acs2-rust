# Phase 3b operations

This directory belongs to trajectory utility. It does not submit or inspect MPX jobs.
`pilot.json` is a pilot manifest, not an approved twenty-seed grid. The grid needs a
separate recorded approval after the pilot report. Pilot rows must never be silently
promoted to grid rows.

The pilot uses the unmodified `acs2-measure` executable, the thesis preset, seed 42,
and no per-start records. Each array element is one configuration and agent, ordered
as configuration index times two plus agent index. The manifest fixes ten candidate
configurations, maximum step budgets, evaluation points and restricted pools before
submission. Pool coordinates use the CLI's row:column convention; HandEye rows encode
the corresponding x,y goals. Both agents have identical points. Timeout results remain
part of the pilot and can justify proposing a smaller grid budget on measured cost.
They do not establish learning limits.

## Build and submit

Use only the dedicated cluster clone `~/acs2-tu`, detached at the pushed TU commit.
Build `cargo build --locked --release --target x86_64-unknown-linux-musl --bin acs2-measure`
on the login node. The clone must be clean before building and running. Capture the
commit, compiler version, build command and binary SHA-256 outside the clone. Never
rebuild another line's clone or binaries. Local checks are short tests only.

Before every submission, record `sshare -U -u alelys2099 -o RawUsage -n`, current queue
commitments and `sbatch --test-only`. The phase budget is 200 allocated CPU-hours,
including failed attempts and pilot runs. Twenty one-CPU pilot elements at a one-hour
limit reserve at most 20 hours; submit no replacements without accounting for those
already spent. Use only `lem-cpu-normal`, one CPU, 2048 MiB, one hour, no automatic
requeue. The collector stops its child after 3300 seconds or if no first row appears
within 180 seconds. Submission passes the repo, commit and batch directory to
`pilot_job.sh`; `SLURM_ARRAY_TASK_ID` selects the element. Set explicit stdout/stderr
paths under the batch directory. Store `binary.sha256` there for the script's check.

Each batch gets a new directory under `~/tu_runs/3b/`. Each attempt gets a fresh run
directory; existing directories cause refusal, so neither retries nor accidental
requeue can overwrite raw rows. Keep scripts and manifests in the pinned source;
also save a manifest copy and the exact submission command beside results.

Inspect every batch within one hour of starting, and investigate missing first rows
before submitting more. A first-row deadline is a fault to diagnose, not evidence that
the agent cannot learn. Pull the entire batch after completion, including incomplete
rows, stderr, launch/completion records and CPU traces, into
`~/Desktop/tu-runs/3b/`. Never use a deleting synchronization. Store `sacct -P` output
for the specific submitted IDs with JobIDRaw, State, ExitCode, AllocCPUS, ElapsedRaw,
CPUTimeRAW, TotalCPU, MaxRSS, NodeList and Partition. Sum CPUTimeRAW only over allocation
rows, not again over `.batch`, `.extern` or `srun` steps. Report TotalCPU separately.

## Measurement and limits

`collect.py` launches the original CLI and preserves its JSONL bytes. It checks every
complete row against the manifest, pinned commit, clean source, seed, point, goal pool,
policy, replay volume, episode overshoot and hardware identifiers. It stops its own
child on a validation failure. `complete_rows` discards and reports an unterminated
last line without rewriting the raw file; malformed complete lines are errors.

Linux `wait4` accounting (through Python's child resource usage) supplies exact process
user/system CPU seconds and peak RSS in KiB after exit. The collector also samples
`/proc/PID/schedstat` and VmHWM every 20 ms. This gives a CPU trace and surviving RSS
measurements even if the allocation is killed. Sampling and JSON serialization add
some overhead to the allocation, which `sacct` includes.

Training/evaluation CPU attribution is an **estimate**, not a new runner measurement:
wall intervals from successive rows are aligned to their observed flush time and CPU
is interpolated from the trace. Polling delay, scheduling gaps and serialization affect
alignment. `resources.jsonl` records the actual largest sampling gap for each interval.
The unchanged runner supplies exact definitions of training/evaluation wall intervals;
initialization, reference construction and output are outside those intervals. Report
wall costs and process CPU totals alongside the phase estimates; do not label wall
seconds as CPU seconds. Very short intervals need this timing qualification. Population
growth is observed at evaluation points; RSS is a process lifetime peak.

One pilot seed cannot estimate between-seed cost tails. Grid time/memory limits and
expected CPU-hours need a stated safety factor, using measured costs through the chosen
budget, plus explicit allowance for evaluation frequency, overhead and reruns. Keep the
overall bound below the remaining 200-hour phase allowance. Hardware names must be
checked across the pilot; sharing a partition alone does not prove one CPU model.

## Checks

`python3 -B -m unittest discover -s experiments/phase3b -p 'test_*.py'`

Run the repository gates when changing this code. The scripts neither modify the core
nor change what `acs2-measure` measures. A discovered runner defect stops the phase for
reporting and a separately gated fix.
