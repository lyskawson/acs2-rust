import hashlib
import json
from pathlib import Path
import subprocess
import sys

from protocol import CORRECTED_PILOT_INDICES, pilot_roots


def verify_previous(root):
    root = Path(root)
    inventory = root / 'checksums.json'
    expected = json.loads(inventory.read_text())['files']
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('previous pilot artifact changed')
    verifier = root / 'operations/experiments/replay-schedule-and-order/archive.py'
    completed = subprocess.run([sys.executable, '-B', str(verifier), str(root)], check=True,
                               capture_output=True, text=True)
    checked = json.loads(completed.stdout)
    summary = json.loads((root / 'pilot-summary.json').read_text())
    if any(summary[key] != value for key, value in checked.items()):
        raise ValueError('previous pilot summary differs from independent verification')
    if (summary['allocations'] != 60 or summary['rows'] != 490 or summary['identity_rows'] != 49
            or summary['allocation_cpu_seconds'] != 5035):
        raise ValueError('previous pilot is not the completed first round')
    return dict(registration_sha256=hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest(),
        inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        **{key: summary[key] for key in ('measurement_commit', 'binary_sha256', 'allocations', 'rows',
                                         'identity_rows', 'allocation_cpu_seconds')})


def previous_path(root):
    root = Path(root)
    registration = json.loads((root / 'pilot-registration.json').read_text())
    previous = registration['previous_pilot']
    return Path(previous['root']) if root.resolve() == Path(registration['destination']).resolve() else (root / previous['mirror_relative']).resolve()


def pilot_batch(root, index):
    root = Path(root)
    return (root if index in CORRECTED_PILOT_INDICES else previous_path(root)) / pilot_roots()[index]


def reference_path(root, declared, source):
    root, declared, source = Path(root).resolve(), Path(declared), Path(source).resolve()
    if source.is_relative_to(root):
        return declared / source.relative_to(root)
    previous = previous_path(root)
    if not source.is_relative_to(previous):
        raise ValueError('pilot identity reference is outside the two registered rounds')
    registration = json.loads((root / 'pilot-registration.json').read_text())
    return Path(registration['previous_pilot']['root']) / source.relative_to(previous)
