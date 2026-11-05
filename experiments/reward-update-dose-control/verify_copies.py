import argparse
import hashlib
import json
from pathlib import Path


def inventory(root):
    root = Path(root)
    files = {}
    for name in ['pilot-registration.json', 'pilot-manifest.json', 'pilot-scope.json', 'reference-index.json', 'model-index.json']:
        path = root / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ['pilot', 'references', 'models', 'operations']:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file():
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in root.glob('*.jsonl'):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def verify(root, expected):
    actual = inventory(root)
    if actual != expected:
        raise ValueError('archive copies differ in paths or SHA-256 bytes')
    return dict(verified=True, files=len(actual), comparison='SHA-256 of every frozen input, operation and raw pilot artifact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--against')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, json.loads(Path(args.against).read_text())) if args.against else inventory(args.root), sort_keys=True))
