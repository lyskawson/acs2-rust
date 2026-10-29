import argparse
import hashlib
import json
from pathlib import Path

from approved_execution import APPROVED_SHA256, approved, catalog_cache
from archive import catalog
from execution import choose, read_index, saved_selection, verified_screening
from protocol import SHORTENED, pilot_roots
from grid_identity import field_bits
from grid_protocol import atomic_json


def save(root):
    root = Path(root)
    plan = approved(root / 'approved-plan.json')
    verified_screening(root)
    entries, state = catalog(root / 'screening')
    if state['reserved_seconds'] or len(entries) != 120:
        raise ValueError('selection requires all 120 complete terminal screening trajectories')
    base = root / pilot_roots()[-1]
    defaults = catalog(base)[0]
    default_rows = {(config['registered_id'], 42): defaults[config['id'], SHORTENED, 42]['rows']
        for config in json.loads((base / 'plan.json').read_text())['configurations']}
    selection = choose(plan, entries, read_index(root), default_rows)
    path = root / 'selections.json'
    if path.exists() and field_bits(json.loads(path.read_text())) != field_bits(selection):
        raise ValueError('saved choice differs from complete registered validation rows')
    atomic_json(path, selection)
    with catalog_cache():
        if field_bits(saved_selection(root, plan)) != field_bits(selection):
            raise ValueError('independent saved selection re-derivation differs')
    proof = dict(plan_sha256=APPROVED_SHA256, selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        screening_runs=len(entries), evaluation_blocks=['validation'], test_access='none',
        input_rows_sha256={'|'.join(map(str, key)): entry['sha256'] for key, entry in entries.items()})
    atomic_json(root / 'selection-verification.json', proof)
    return dict(selection=selection, verification=proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(save(args.root)))
