import json
from pathlib import Path
import sys

from protocol import pilot


def measurement_identity():
    identity = json.loads(Path(__file__).with_name('measurement.json').read_text())
    if len(identity.get('measurement_commit', '')) != 40 or len(identity.get('binary_sha256', '')) != 64:
        raise ValueError('measurement identity is not frozen')
    return identity


def authorization_manifest(manifest, registration_path=None):
    if manifest['stage'] != 'authorized_cost_pilot':
        raise ValueError('stage 2 requires a separate approved execution registration')
    return pilot() | measurement_identity()


if __name__ == '__main__':
    if sys.argv[1:] != ['--derive-manifest']:
        raise SystemExit('expected --derive-manifest')
    print(json.dumps(authorization_manifest(**json.load(sys.stdin))))
