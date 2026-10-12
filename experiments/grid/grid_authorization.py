import json
from pathlib import Path
import subprocess
import sys

from grid_protocol import load_plan

TRANSPORT_FIELDS = {
    'authorization_plan', 'authorization_sha256', 'budget_scope', 'budget_scope_sha256',
    'stop_path', 'operations_commit', 'operations_manifest', 'operations_manifest_sha256',
    'identity_references', 'initial_dependencies', 'job_name',
}
IDENTITY_FIELDS = {'measurement_commit', 'binary_sha256', 'preset'}


def derive_manifest(manifest, registration_path=None):
    experiment = manifest.get('experiment')
    if experiment not in ('her-variants', 'maze-baselines'):
        raise ValueError('experiment has no registered manifest derivation')
    script = Path(__file__).resolve().parents[1] / experiment / 'execution.py'
    payload = dict(manifest=manifest, registration_path=str(registration_path) if registration_path else None)
    result = subprocess.run([sys.executable, '-B', str(script), '--derive-manifest'],
                            input=json.dumps(payload), text=True, capture_output=True)
    if result.returncode:
        raise ValueError('registered manifest derivation failed: ' + result.stderr.strip())
    return json.loads(result.stdout)


def require_derivation(manifest, expected):
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f'execution manifest differs from mechanical registration: {field}')
    allowed = set(expected) | TRANSPORT_FIELDS | IDENTITY_FIELDS
    if set(manifest) - allowed:
        raise ValueError('execution manifest contains unregistered fields')


def require_authorization(root, plan, digest, approval_path=None):
    if plan['stage'] == 'authorized_cost_pilot':
        require_derivation(plan, derive_manifest(plan))
        return
    if plan['stage'] not in ('grid_pending_approval', 'approved_grid', 'tuning_pending_approval'):
        raise ValueError('superseded or unsupported experiment stage')
    approval = json.loads(Path(approval_path or Path(root) / 'approval.json').read_text())
    registered = Path(plan['authorization_plan']) if 'authorization_plan' in plan else None
    authorized_digest = plan['authorization_sha256'] if registered else digest
    if registered:
        load_plan(registered, authorized_digest)
    if approval.get('plan_sha256') != authorized_digest or approval.get('approved') is not True:
        raise ValueError('grid requires explicit approval of these plan bytes')
    require_derivation(plan, derive_manifest(plan, registered))
