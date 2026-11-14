"""Verify the released source and fixed splits; optionally run a CPU smoke workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from gecko.data.predefined import read_json, safe_path, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    integrity = read_json(root / 'reproduce/checksums.json')
    for relative, expected in integrity.items():
        path = safe_path(root, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Missing or changed release file: {relative}')
    if [p for p in integrity if p.lower().endswith(('.md', '.markdown'))] != ['README.md']:
        raise ValueError('The release must contain exactly one Markdown file: README.md')
    split_status = verify(root / 'reproduce/splits')
    print(json.dumps({'release_files_verified': len(integrity),
                      'splits': split_status}, indent=2), flush=True)
    if args.smoke:
        subprocess.run([sys.executable, str(root / 'reproduce/smoke.py')], cwd=root, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
