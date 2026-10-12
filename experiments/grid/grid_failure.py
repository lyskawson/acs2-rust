import os
import json
from pathlib import Path
import sys

from grid_protocol import stop_phase
from grid_scope import stop_path


if __name__ == '__main__':
    root, assignment, index = sys.argv[1:]
    plan = json.loads((Path(root) / 'plan.json').read_text())
    target = stop_path(root, plan)
    if not target.exists():
        stop_phase(target, dict(error='node launcher or srun failed',
                   assignment=assignment, index=int(index), job=os.environ.get('SLURM_JOB_ID')))
