"""Inventory tensor payloads in trusted GECKO stream artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

from gecko.data.partitioning.base import tensor_sha256


def tensor_entries(value, path=()):
    if torch.is_tensor(value):
        yield path, tensor_sha256(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            key_path = ('key', type(key).__name__, repr(key))
            yield from tensor_entries(child, (*path, key_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_entries(child, (*path, ('index', index)))


def content_hash(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    entries = sorted(tensor_entries(payload))
    if not entries:
        raise ValueError(f'No tensors found in {path}')
    canonical = json.dumps(entries, ensure_ascii=True, separators=(',', ':'))
    return len(entries), hashlib.sha256(canonical.encode('ascii')).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description='Hash tensor paths, dtypes, shapes and values in trusted .pt files only.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--file-list', type=Path, help='newline-delimited transfer manifest')
    inputs.add_argument('--file', type=Path, nargs='+', help='explicit candidate files')
    args = parser.parse_args()
    torch.set_num_threads(1)
    files = args.file
    if args.file_list:
        files = [Path(line) for line in args.file_list.read_text().splitlines()
                 if line.endswith('.pt')]
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(('file', 'tensor_count', 'tensor_tree_sha256_v1'))
    cache = {}
    for path in files:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in cache:
            cache[identity] = content_hash(path)
        writer.writerow((path, *cache[identity]))


if __name__ == '__main__':
    main()
