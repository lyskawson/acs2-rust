import argparse
import hashlib
import json
from pathlib import Path

from approved_execution import APPROVED_SHA256, approved
from archive import catalog
from execution import choose, registered_references
from grid_authorization import require_authorization
from grid_identity import field_bits
from grid_protocol import atomic_json, load_plan


def save(root):
    root = Path(root)
    plan = approved(root / 'approved-plan.json')
    tuning = root / 'tuning'
    digest = (tuning / 'plan.sha256').read_text().strip()
    manifest = load_plan(tuning / 'plan.json', digest)
    require_authorization(tuning, manifest, digest, root / 'approval.json')
    entries, state = catalog(tuning)
    if state['reserved_seconds'] or len(entries) != plan['tuning']['runs']:
        raise ValueError('selection requires complete terminal tuning')
    selection = choose(plan, entries, registered_references(plan, root))
    path = root / 'selections.json'
    if path.exists() and field_bits(json.loads(path.read_text())) != field_bits(selection):
        raise ValueError('saved selection differs from complete registered tuning')
    atomic_json(path, selection)
    proof = dict(plan_sha256=APPROVED_SHA256, selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 tuning_runs=len(entries), two_thirds_identity_rows=sum(len(config['targets']) * 5 for config in plan['configurations']),
                 input_rows_sha256={'|'.join(map(str, key)): entry['sha256'] for key, entry in entries.items()},
                 evaluation_blocks=['validation'], test_access='none')
    atomic_json(root / 'selection-verification.json', proof)
    return dict(selection=selection, verification=proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(save(args.root)))
