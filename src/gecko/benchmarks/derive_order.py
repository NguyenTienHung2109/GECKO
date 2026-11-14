"""Derive an order-only stream variant while reusing scientific components."""

from __future__ import annotations

import argparse
from pathlib import Path
from gecko.data.streams import derive_order_variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True)
    parser.add_argument("--order-profile", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    for path in derive_order_variants(
        args.stream,
        args.order_profile,
        root=args.output_root,
    ):
        print(path)
    return 0




if __name__ == "__main__":
    raise SystemExit(main())

