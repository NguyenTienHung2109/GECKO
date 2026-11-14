"""Construct a selected benchmark scenario using its original data recipe."""
import sys
from gecko.benchmarks.paper import main


if __name__ == '__main__':
    raise SystemExit(main(['--stage', 'construct', *sys.argv[1:]]))
