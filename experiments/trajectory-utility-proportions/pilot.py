import argparse
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import time

from proportions import (CONCURRENCY, HERE, MEASURES, PILOT_SECONDS, REPO, STUDY,
                         checksum, pilot_cells, read)
from grid_protocol import TERMINAL, accounting, allocation_rows, atomic_json, load_plan, require_budget
from grid_scope import SACCT_FIELDS
from submit import execute, shared_reservation, submission_command


CLONE = (Path.home() / 'acs2-tu').resolve()
ROOT = (Path.home() / 'tu_runs' / STUDY).resolve()
BINARY = CLONE / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
PHASE_START = '2026-11-23'


def clean_head():
    git = ['/usr/bin/git', '-C', str(REPO)]
    if REPO != CLONE or execute(git + ['status', '--porcelain']).strip():
        raise ValueError('a clean dedicated cluster checkout is required')
    return execute(git + ['rev-parse', 'HEAD']).strip()


def grant(reservation):
    used = int(execute(['sshare', '-U', '-u', 'alelys2099', '-o', 'RawUsage', '-n']).strip())
    queue = execute(['squeue', '--array', '-u', 'alelys2099', '-h', '-o', '%i|%j|%t|%L|%C'])
    if used + shared_reservation(queue) + reservation > 15_000 * 3600:
        raise ValueError('shared grant cannot cover the queue and remaining reservations')
    return dict(used_cpu_seconds=used, queue=queue, new_and_planned_seconds=reservation)


def evidence_submit(command, path, reservation, **fields):
    event = dict(fields, array_id=None, command=command, grant=grant(reservation), unix=time.time())
    event['test_only'] = execute(command[:1] + ['--test-only'] + command[1:])
    atomic_json(path, event)
    response = execute(command).strip()
    event.update(array_id=int(response.split(';')[0]), response=response)
    atomic_json(path, event)
    print(path.name, event['array_id'], flush=True)


def build(expected_commit):
    if clean_head() != expected_commit:
        raise ValueError('checkout is not at the pinned commit')
    settings, _, _, runs = pilot_cells()
    limit = settings['build_limit_seconds']
    reservation = sum(run['limit_seconds'] for run in runs) + limit
    require_budget(0, dict(spent_seconds=0, reserved_seconds=0), reservation, PILOT_SECONDS / 3600)
    folder = ROOT / 'build'
    folder.mkdir(parents=True, exist_ok=False)
    command = ['sbatch', '--parsable', '--partition=lem-cpu-normal', '--cpus-per-task=1',
               '--mem=8192M', f'--time={limit // 60}', '--no-requeue', '--job-name=tu-proportions-build',
               '--output=' + str(folder / 'build.out'), '--error=' + str(folder / 'build.err'),
               str(MEASURES / 'build.sh'), str(CLONE), expected_commit]
    evidence_submit(command, folder / 'submission.json', reservation, commit=expected_commit,
                    limit_seconds=limit, registration_sha256=checksum(HERE / 'plan.json'),
                    cells_sha256=checksum(HERE / 'pilot-cells.json'))


def sacct(ids):
    if not ids:
        return '', {}
    text = execute(['sacct', '--array', '-S', PHASE_START, '-P', '--format=' + SACCT_FIELDS,
                    '-j', ','.join(str(job) for job in ids)])
    records = {row['JobIDRaw']: row for row in allocation_rows(text)}
    return text, {row['JobID']: row for row in records.values()}


def state(row):
    return row['State'].split()[0].split('+')[0] if row else 'awaiting_accounting'


def measurement_identity():
    folder = ROOT / 'build'
    build_event = read(folder / 'submission.json')
    job = build_event['array_id']
    raw, records = sacct([job])
    row = records.get(str(job))
    if state(row) != 'COMPLETED' or row['ExitCode'] != '0:0' or row['AllocCPUS'] != '1':
        raise ValueError('pinned build has not completed successfully')
    commit = clean_head()
    digest = checksum(BINARY)
    cpu = int(row['CPUTimeRAW'])
    if (commit != build_event['commit'] or not 0 < cpu <= build_event['limit_seconds']
            or digest not in (folder / 'build.out').read_text()
            or build_event['registration_sha256'] != checksum(HERE / 'plan.json')
            or build_event['cells_sha256'] != checksum(HERE / 'pilot-cells.json')):
        raise ValueError('build provenance differs from the pilot registration')
    (folder / 'accounting.psv').write_text(raw)
    return commit, digest, cpu


def batches(runs):
    long, short = defaultdict(list), defaultdict(list)
    for run in runs:
        limit = run['limit_seconds']
        (long if limit >= 1800 else short)[limit].append(run)
    short_slots = CONCURRENCY - sum(map(len, long.values()))
    if short_slots < 1:
        raise ValueError('long measurements leave no concurrency for ordinary runs')
    return ([(group, len(group), False) for _, group in sorted(long.items(), reverse=True)] +
            [(group, min(len(group), short_slots), True) for _, group in sorted(short.items(), reverse=True)])


def wave_plan(number, digest, commit, build_cpu, previous):
    settings, preset, configs, registered = pilot_cells()
    completed = previous['completed']
    runs = []
    for base in registered:
        key = base['run_id']
        if key in completed:
            continue
        run = dict(base, attempt_id=f'{key}_a{number + 1:02d}')
        if number:
            earlier = previous['attempts'].get(key)
            if earlier and earlier['state'] == 'TIMEOUT' or (
                    earlier and earlier.get('completion', {}).get('failure', '') == 'TimeoutError: process deadline exceeded'):
                run['limit_seconds'] = ((base['limit_seconds'] * 3 + 119) // 120) * 60
        run['directory'] = f"attempts/{run['attempt_id']}"
        runs.append(run)
    if not runs:
        raise ValueError('no pilot measurements to submit')
    return dict(study=STUDY, wave=number, commit=commit, binary_sha256=digest, build_cpu_seconds=build_cpu,
                registration_sha256=checksum(HERE / 'plan.json'), cells_sha256=checksum(HERE / 'pilot-cells.json'),
                cpu_model=settings['cpu_model'], memory_mib=settings['memory_mib'], configurations=configs,
                preset=preset, runs=runs, earlier=previous)


def wave_dir(number):
    return ROOT / f'wave-{number:02d}'


def run_directory(folder, run, configs):
    return folder / run['directory'] / 'measurement' / f"{configs[run['task']]['id']}_{run['agent']}_s42"


def wave_status(folder, plan):
    ledger = [read(path) for path in sorted(folder.glob('submission-*.json'))]
    if any(entry['array_id'] is None for entry in ledger):
        raise ValueError('submission intent requires reconciliation before proceeding')
    raw, records = sacct([entry['array_id'] for entry in ledger])
    balance = accounting(ledger, list(records.values()))
    counts = Counter()
    attempts, completed = {}, {}
    for entry in ledger:
        assignment = folder / entry['assignment']
        if checksum(assignment) != entry['assignment_sha256']:
            raise ValueError('submitted assignment changed')
        for index, run in enumerate(entry['runs']):
            key = run['run_id']
            job = f"{entry['array_id']}_{index}"
            row = records.get(job)
            status = state(row)
            counts[status] += 1
            directory = run_directory(folder, run, plan['configurations'])
            completion_path = directory / 'completion.json'
            item = dict(state=status, job=job, relative_directory=str(directory.relative_to(ROOT)),
                        limit_seconds=run['limit_seconds'], job_raw=row['JobIDRaw'] if row else None,
                        charged_cpu_seconds=int(row['CPUTimeRAW']) if row else 0)
            if completion_path.is_file():
                item['completion'] = read(completion_path)
            attempts[key] = item
            if status == 'COMPLETED':
                result = item['completion']
                config = plan['configurations'][run['task']]
                wanted = sum(target <= run['last_target'] for target in config['targets'])
                if (row['ExitCode'] != '0:0' or row['AllocCPUS'] != '1'
                        or not 0 < item['charged_cpu_seconds'] <= run['limit_seconds']
                        or result['failure'] or result['exit_code'] or result['incomplete_final_line']
                        or result['rows_seen'] != wanted
                        or any(checksum(directory / name) != digest for name, digest in result['sha256'].items())):
                    raise ValueError(f'invalid completed pilot measurement {key}')
                completed[key] = item
    return raw, dict(states=dict(counts), attempts=attempts, completed=completed,
                     stopped=(folder / 'STOP.json').exists(), **balance)


def earlier_waves(number):
    combined = dict(spent_seconds=0, completed={}, attempts={})
    for prior in range(number):
        path = wave_dir(prior) / 'sealed.json'
        if not path.is_file():
            raise ValueError('seal the previous pilot wave first')
        result = read(path)
        combined['spent_seconds'] += result['spent_seconds']
        combined['completed'].update(result['completed'])
        combined['attempts'].update(result['attempts'])
    return combined


def submit(number):
    commit, digest, build_cpu = measurement_identity()
    previous = earlier_waves(number)
    plan = wave_plan(number, digest, commit, build_cpu, previous)
    remaining = sum(run['limit_seconds'] for run in plan['runs'])
    require_budget(build_cpu + previous['spent_seconds'], dict(spent_seconds=0, reserved_seconds=0),
                   remaining, PILOT_SECONDS / 3600)
    folder = wave_dir(number)
    folder.mkdir(exist_ok=False)
    (folder / 'logs').mkdir()
    plan_path = folder / 'plan.json'
    atomic_json(plan_path, plan)
    frozen = checksum(plan_path)
    (folder / 'plan.sha256').write_text(frozen + '\n')
    dependency = []
    for index, (runs, throttle, chained) in enumerate(batches(plan['runs'])):
        if (folder / 'STOP.json').exists():
            raise ValueError('pilot STOP prevents further submission')
        if load_plan(plan_path, frozen) != wave_plan(number, checksum(BINARY), clean_head(), build_cpu, previous):
            raise ValueError('pilot registration or binary changed')
        _, balance = wave_status(folder, plan)
        bound = require_budget(build_cpu + previous['spent_seconds'], balance, remaining, PILOT_SECONDS / 3600)
        name = f'assignment-{index:02d}.json'
        assignment = folder / name
        atomic_json(assignment, dict(plan_sha256=frozen, batch=index, runs=runs))
        command = submission_command(folder, assignment, runs[0]['limit_seconds'] // 60, len(runs),
                                     throttle, plan['memory_mib'], dependency if chained else [],
                                     'tu-proportions-pilot')
        command[-4:] = [str(HERE / 'job.sh'), str(HERE), str(folder), str(assignment)]
        command.insert(1, '--kill-on-invalid-dep=yes')
        path = folder / f'submission-{index:02d}.json'
        evidence_submit(command, path, remaining, runs=runs, assignment=name,
                        assignment_sha256=checksum(assignment), bound_cpu_seconds=bound)
        if chained:
            dependency = [read(path)['array_id']]
        remaining -= sum(run['limit_seconds'] for run in runs)


def summarize(number, seal):
    folder = wave_dir(number)
    plan = load_plan(folder / 'plan.json', (folder / 'plan.sha256').read_text().strip())
    raw, result = wave_status(folder, plan)
    total = result['spent_seconds'] + plan['build_cpu_seconds'] + plan['earlier']['spent_seconds']
    print(dict(wave=number, total_charged_cpu_seconds=total, **{key: result[key]
          for key in ('states', 'stopped', 'reserved_seconds')}), flush=True)
    if seal:
        if set(result['states']) - TERMINAL or result['reserved_seconds']:
            raise ValueError('only terminal waves can be sealed')
        if total > PILOT_SECONDS:
            raise ValueError('pilot accounting exceeds authorization')
        (folder / 'accounting.psv').write_text(raw)
        atomic_json(folder / 'sealed.json', result)
        completed = plan['earlier']['completed'] | result['completed']
        if len(completed) == 168:
            atomic_json(ROOT / 'pilot-summary.json', dict(completed=completed, total_charged_cpu_seconds=total,
                                                         build_cpu_seconds=plan['build_cpu_seconds'],
                                                         waves=number + 1, measurement_commit=plan['commit'],
                                                         binary_sha256=plan['binary_sha256']))
            manifest = ROOT / 'files.sha256.json'
            atomic_json(manifest, {str(path.relative_to(ROOT)): checksum(path) for path in sorted(ROOT.rglob('*'))
                                   if path.is_file() and path != manifest})
            print('sealed', checksum(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'submit', 'status', 'seal'))
    parser.add_argument('--commit')
    parser.add_argument('--wave', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.commit)
    elif args.command == 'submit':
        submit(args.wave)
    else:
        summarize(args.wave, args.command == 'seal')
