import json
from pathlib import Path
import sys

from protocol import PILOT_PARTS, RECOVERY_PARTS, pilot, recovery


def measurement_identity():
    path = Path(__file__).with_name('measurement.json')
    identity = json.loads(path.read_text())
    if len(identity.get('measurement_commit', '')) != 40 or len(identity.get('binary_sha256', '')) != 64:
        raise ValueError('VCP measurement source and binary have not been frozen')
    return identity


def authorization_manifest(manifest, registration_path=None):
    if manifest['stage'] != 'authorized_cost_pilot':
        raise ValueError('VCP stage 2 is not registered or approved; stop after the stage-1 plan')
    part = manifest.get('pilot_part')
    if part in RECOVERY_PARTS:
        return recovery(part) | measurement_identity()
    if part not in PILOT_PARTS:
        raise ValueError('unregistered VCP pilot part')
    return pilot(part) | measurement_identity()


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
