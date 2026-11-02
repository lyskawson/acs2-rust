import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def array_units(ledger, recovered=()):
    if recovered or any(entry.get('retry_of') is not None for entry in ledger):
        raise ValueError('recovery requires explicit cross-array prefix verification')
    return [[entry] for entry in ledger]


def verify_arrays(operations, root, plan, ledger, recovered=()):
    for unit in array_units(ledger, recovered):
        payload = dict(root=str(root), plan=plan, ledger=unit)
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--verify-array', str(operations)],
                       input=json.dumps(payload), text=True, check=True)


def verify_array(operations):
    operations = Path(operations)
    sys.path.insert(0, str(operations / 'experiments/per-pair-diagnostics'))
    sys.path.insert(1, str(operations / 'experiments/grid'))
    import protocol
    from grid_dispatch import verify_completed
    resource.setrlimit(resource.RLIMIT_AS, (850 * 1024 * 1024, 850 * 1024 * 1024))
    payload = json.load(sys.stdin)
    verify_completed(Path(payload['root']), payload['plan'], payload['ledger'])
    print(json.dumps(dict(verified_array=payload['ledger'][0]['array_id'], peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)


def dispatch(root, repository):
    root, repository = Path(root), Path(repository)
    operations = root / 'execution-operations'
    if (root / 'approval.json').read_bytes() != (operations / 'experiments/per-pair-diagnostics/approval.json').read_bytes():
        raise ValueError('user approval differs from its immutable copy')
    if json.loads((root / 'approval.json').read_text())['phase_cpu_hour_limit'] != 15:
        raise ValueError('cumulative limit differs from explicit approval')
    sys.path.insert(0, str(operations / 'experiments/per-pair-diagnostics'))
    sys.path.insert(1, str(operations / 'experiments/grid'))
    import grid_dispatch
    from execute_batch import execute
    if not Path(grid_dispatch.__file__).resolve().is_relative_to(operations.resolve()):
        raise ValueError('dispatcher must come from pinned operations')
    original = grid_dispatch.verify_completed
    grid_dispatch.verify_completed = lambda root, plan, ledger, recovered=(): verify_arrays(operations, root, plan, ledger, recovered)
    started = time.monotonic()
    try:
        result = execute(root, repository, 'replication')
    finally:
        grid_dispatch.verify_completed = original
    evidence = dict(result=result, controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_dispatcher_sha256=hashlib.sha256(Path(grid_dispatch.__file__).read_bytes()).hexdigest(),
        verification='Original pinned verifier, one complete array per foreground child; retries refused.',
        elapsed_seconds=time.monotonic() - started, allocated_controller_cpu_seconds=0)
    with (root / 'controller-invocations.jsonl').open('a') as stream:
        stream.write(json.dumps(evidence) + '\n')
    return result


if __name__ == '__main__':
    if sys.argv[1:2] == ['--verify-array']:
        verify_array(sys.argv[2])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', required=True)
        parser.add_argument('--repo', required=True)
        args = parser.parse_args()
        print(json.dumps(dispatch(args.root, args.repo)), flush=True)
