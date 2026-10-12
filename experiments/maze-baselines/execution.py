import json
from pathlib import Path
import sys


def authorization_manifest(manifest, registration_path=None):
    base = Path(__file__).resolve().parent
    if manifest['stage'] == 'authorized_cost_pilot':
        pilots = [json.loads((base / name).read_text()) for name in ('pilot.json', 'pair-pilot.json')]
        for pilot in pilots:
            if all(manifest.get(field) == value for field, value in pilot.items()):
                return pilot
        raise ValueError('cost pilot differs from every registered maze pilot')
    registered = json.loads((base / 'grid-plan.json').read_text())
    if registration_path is not None and json.loads(Path(registration_path).read_text()) != registered:
        raise ValueError('maze registration differs from the registered pair grid')
    return registered


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
