import os
from pathlib import Path
import sys

from grid_protocol import stop_phase


if __name__ == '__main__':
    root, assignment, index = sys.argv[1:]
    if not (Path(root) / 'STOP.json').exists():
        stop_phase(Path(root) / 'STOP.json', dict(error='node launcher or srun failed',
                   assignment=assignment, index=int(index), job=os.environ.get('SLURM_JOB_ID')))
