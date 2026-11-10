import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALCS_COMMIT = 'ed0e76282a121fa2bad7dac82b744b2cc763eb72'
FIXTURE = ROOT / 'fixtures/alcs_maze_geometries.json'
KNOWN = {'MiyazakiA': [([3, 6], 9, 0), ([6, 3], 0, 9)],
         'Woods101_5': [([9, 3], 9, 0)], 'Woods102': [([9, 3], 9, 0)]}


def build(checkout):
    git = ['/usr/bin/git', '-C', str(Path(checkout))]
    files = subprocess.check_output(git + ['ls-tree', '--name-only', ALCS_COMMIT + ':environment/acs2_mazes'], text=True).splitlines()
    entries = []
    for filename in files:
        raw = subprocess.check_output(git + ['show', ALCS_COMMIT + ':environment/acs2_mazes/' + filename])
        text = re.sub(r'//[^\n]*', '', raw.decode()).split('.maze =', 1)[1]
        matrix = [[{'OBSTACLE': 1, 'CORRIDOR': 0, 'PRIZE': 9}[token]
                   for token in re.findall(r'OBSTACLE|CORRIDOR|PRIZE', row)]
                  for row in re.findall(r'\{([^{}]+)\}', text)]
        matrix = [row for row in matrix if row]
        if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError('invalid geometry: ' + filename)
        name = Path(filename).stem
        task = {'Maze4': 'maze4', 'MazeMA': 'mazema'}.get(name, name.lower() + '_alcs')
        entries.append(dict(file=filename, sha256=hashlib.sha256(raw).hexdigest(), task=task, matrix=matrix,
            benchmark_geometry=(name + '-ounold') if name in KNOWN else None,
            known_benchmark_differences=[dict(cell=cell, alcs=alcs, benchmark=bench) for cell, alcs, bench in KNOWN.get(name, [])]))
    if len(entries) != 24:
        raise ValueError('expected all 24 ALCS source files')
    return dict(commit=ALCS_COMMIT, files=entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkout')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    value = build(args.checkout)
    if args.write:
        FIXTURE.write_text(json.dumps(value, indent=2) + '\n')
    elif value != json.loads(FIXTURE.read_text()):
        raise SystemExit('ALCS source hashes or geometry differ')
    print('Verified 24 ALCS source hashes and matrices at ' + ALCS_COMMIT)
