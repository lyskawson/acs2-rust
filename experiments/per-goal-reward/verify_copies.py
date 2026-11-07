import argparse
import hashlib
import json
from pathlib import Path


def inventory(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('archive inventory requires regular files, without symlinks')
        if path.is_file():
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def verify(root, expected):
    actual = inventory(root)
    if actual != expected:
        raise ValueError('archive copies differ in paths or SHA-256 bytes')
    return dict(verified=True, files=len(actual), comparison='Independent SHA-256 inventories of every study archive file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--against')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, json.loads(Path(args.against).read_text()))
                     if args.against else inventory(args.root), sort_keys=True))
