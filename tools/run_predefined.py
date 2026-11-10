"""Run one explicit method on a released fixed split, without resampling it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from gecko.benchmarks.paper import METHODS
from gecko.data.predefined import read_catalog, load_predefined
from gecko.data.streams.store import save_stream


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True, dest='cell_id')
    parser.add_argument('--method', required=True, choices=tuple(METHODS))
    parser.add_argument('--catalog', type=Path, default=root / 'splits/predefined')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('generated_streams/predefined'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    cells = [c for c in read_catalog(args.catalog)['streams'] if c['id'] == args.cell_id]
    if len(cells) != 1:
        parser.error('Unknown predefined split ID; use python -m gecko.data.predefined list')
    cell = cells[0]
    scenario = 'S7' if cell['scenario'].startswith('LP-') else cell['scenario']
    method = METHODS[args.method]
    if scenario not in method.configs:
        parser.error('No original method recipe for this scenario; use the explicit gecko run CLI for extensions')
    if args.method == 'FedFST' and cell['order'] != 'synchronized':
        parser.error('The original native FedFST recipe is synchronized only')
    recipe = root / 'configs/gecko_v1/algorithms' / method.configs[scenario]
    plan = {'split': cell, 'method': args.method, 'strategy': method.strategy,
            'model': method.model, 'method_config': str(recipe.relative_to(root))}
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan:
        return 0
    bundle = load_predefined(args.catalog, args.cell_id, args.data_root)
    stream = save_stream(bundle, args.output / args.cell_id)
    # The normal run command enforces stream/method support and eligibility;
    # importing an old stream never bypasses those checks.
    command = [sys.executable, '-m', 'gecko', 'run', '--config', str(stream / 'config.yaml'),
               '--stream', str(stream), '--method-config', str(recipe), '--strategy', method.strategy,
               '--cl-algorithm', method.continual_method, '--model', method.model,
               '--model-seed', str(cell['seed']), '--device', args.device,
               '--wandb-mode', 'disabled', '--run-tag', 'reviewer-fixed-split']
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
