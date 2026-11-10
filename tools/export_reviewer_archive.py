"""Export the reviewer snapshot without Git metadata, logs, caches or checkpoints."""
import argparse
import hashlib
from pathlib import Path
import zipfile

DIRECTORIES = ('src', 'configs', 'tests', 'tools', 'examples', 'docs', 'splits')
ROOT_FILES = ('README.md', 'LICENSE.md', 'NOTICE', 'pyproject.toml', 'environment.yml',
              'pytest.ini', '.gitignore', 'MANIFEST.in', 'SHA256SUMS')


def files(root):
    selected = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        for path in (root / name).rglob('*'):
            if path.is_file() and not any(p.startswith('.') or p in ('__pycache__',)
                                          or p.endswith('.egg-info') for p in path.relative_to(root).parts):
                if path.suffix not in ('.pyc', '.pyo'):
                    if path.is_symlink():
                        raise ValueError(f'Release must not contain symlinks: {path}')
                    selected.append(path)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root) and any(args.output.resolve().is_relative_to(root / d) for d in DIRECTORIES):
        parser.error('Place the archive outside the source directories')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = files(root)
    if args.output.resolve() in selected:
        parser.error('Archive would include itself')
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected:
            member = zipfile.ZipInfo('gecko/' + path.relative_to(root).as_posix(), date_time=(2026, 11, 10, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(member, path.read_bytes())
    print(f'{hashlib.sha256(args.output.read_bytes()).hexdigest()}  {args.output}')
    print(f'{len(selected)} files; no Git history or archive comment')


if __name__ == '__main__':
    main()
