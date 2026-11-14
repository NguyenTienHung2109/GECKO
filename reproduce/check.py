"""Check the released files, fixed splits and trained weights; optionally run a CPU smoke workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from gecko.data.predefined import TensorStore, read_json, safe_path, verify


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
    artifacts = root / 'reproduce/artifacts'
    store = TensorStore(artifacts)
    checked = set()

    def inspect(value):
        if isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, dict):
            if set(value) == {'tensor'}:
                digest = value['tensor']['sha256']
                if digest not in checked:
                    store.decode(value); checked.add(digest)
            elif set(value) == {'inline', 'tensor_record'}:
                store.decode(value)
            else:
                for child in value.values():
                    inspect(child)

    catalog = read_json(artifacts / 'catalog.json')
    for entry in catalog['artifacts']:
        inspect(read_json(safe_path(artifacts, entry['file']), entry['sha256']))
    print(json.dumps({'release_files_verified': len(integrity), 'splits': split_status,
                      'trained_reference_runs': len(catalog['artifacts']),
                      'trained_tensor_objects_verified': len(checked)}, indent=2), flush=True)
    if args.smoke:
        subprocess.run([sys.executable, str(root / 'reproduce/smoke.py')], cwd=root, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
