"""Run the self-contained CPU construction, audit, training and report example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/gecko_v1/scenarios/synthetic_smoke.yaml"
    output = args.output.resolve()
    prefix = [sys.executable, "-m", "gecko"]
    construction = subprocess.run(
        [*prefix, "construct", "--config", str(config), "--output-root", str(output)],
        cwd=root, check=True, text=True, capture_output=True,
    )
    stream = Path(construction.stdout.strip().splitlines()[-1])
    print(f"Stream: {stream}", flush=True)
    subprocess.run([*prefix, "audit", "--stream", str(stream)], cwd=root, check=True)
    subprocess.run(
        [*prefix, "run", "--config", str(config), "--stream", str(stream),
         "--output-root", str(output), "--strategy", "fedavg", "--cl-algorithm", "Bare",
         "--model", "uefa_gcn", "--device", "cpu", "--wandb-mode", "disabled",
         "--run-tag", "synthetic-smoke"],
        cwd=root, check=True,
    )
    subprocess.run(
        [*prefix, "report", "--stream", str(stream), "--include-ineligible"],
        cwd=root, check=True,
    )
    print(json.dumps({"status": "passed", "stream": str(stream), "device": "cpu"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
