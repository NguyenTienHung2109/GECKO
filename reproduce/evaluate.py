"""Evaluate released trained weights at every stage, without training or resampling."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from gecko.data.predefined import TensorStore, load_predefined, read_catalog, read_json, safe_path
from gecko.engine.coordinator import FederatedCoordinator
from gecko.engine.parameter_manifest import build_parameter_manifest
from gecko.engine.runtime import _restore_dynamic_model_state
from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.evaluation.metrics import summarize_continual_matrix


def evaluate(artifact_root: Path, split_root: Path, artifact_id: str,
             data_root: Path, device: str, tolerance: float) -> dict:
    catalog = read_json(artifact_root / 'catalog.json')
    matches = [r for r in catalog['artifacts'] if r['id'] == artifact_id]
    if len(matches) != 1:
        raise ValueError('Unknown artifact ID; use --list')
    entry = matches[0]
    payload = read_json(safe_path(artifact_root, entry['file']), entry['sha256'])
    if payload['format'] != 'gecko-trained-evaluation-v1':
        raise ValueError('Unsupported trained evaluation format')
    split = next(s for s in read_catalog(split_root)['streams'] if s['id'] == payload['split_id'])
    if split['sha256'] != payload['split_sha256']:
        raise ValueError('Evaluation weights are bound to a different fixed split')
    if payload['method'] != 'MOTION' or payload['evaluation_state'] != 'shared_parameters_with_client_local_buffers':
        raise ValueError('This release evaluates the explicitly exported native MOTION state only')
    bundle = load_predefined(split_root, payload['split_id'], data_root)
    recipe = payload['method_config']
    coordinator = FederatedCoordinator(
        bundle, strategy_name=recipe['strategy']['name'],
        algorithm_name=recipe['continual_method']['name'], model_name=payload['model'],
        model_seed=payload['seed'], device=device, method_config=recipe,
    )
    evaluator = FederatedEvaluator(bundle)
    store = TensorStore(artifact_root)
    clients, tasks = bundle.config.partition.num_clients, bundle.scenario.num_tasks
    matrix = torch.full((clients, tasks, tasks), float('nan'))
    counts = torch.zeros(clients, tasks)
    rows = []
    if [s['stage'] for s in payload['stages']] != list(range(tasks)):
        raise ValueError('Incomplete stage coverage')
    for stage_record in payload['stages']:
        stage = stage_record['stage']
        states = store.decode(stage_record['clients'])
        if set(states) != set(coordinator.clients):
            raise ValueError('Incomplete client coverage')
        for cid, state in states.items():
            model = coordinator.clients[cid].model
            model.load_state_dict(state['model_state'], strict=True)
            _restore_dynamic_model_state(model, build_parameter_manifest(model, coordinator.parameter_policy),
                                         state['dynamic_model_state'])
            seen = set(state['seen_tasks'])
            expected_seen = set(bundle.orders.client_orders[cid][:stage + 1])
            if seen != expected_seen:
                raise ValueError('Checkpoint exposure does not match the fixed task order')
            for task in sorted(seen):
                metrics = evaluator.evaluate(model, cid, task, seen, final_stage=stage == tasks - 1)
                value = metrics['accuracy']
                if not torch.isfinite(torch.tensor(value)):
                    raise ValueError('Undefined accuracy in a released reference evaluation')
                matrix[cid, stage, task] = value
                counts[cid, task] = bundle.evaluation_shards[cid][task].test_query_ids.numel()
                rows.append({'artifact': artifact_id, 'stage': stage, 'client': cid,
                             'task': task, 'accuracy': value})
        print(f"Evaluated stage {stage + 1}/{tasks}", flush=True)
    reference = store.decode(payload['expected_matrix'])
    if not torch.equal(torch.isnan(matrix), torch.isnan(reference)):
        raise ValueError('Evaluation population differs from the archived result')
    error = float((matrix - reference).abs().nan_to_num().max())
    summary = summarize_continual_matrix(matrix, bundle.orders, counts, base_metric='accuracy')
    acquisition = float(torch.stack([matrix[c, bundle.orders.inverse_client_orders[c][t], t]
                                    for c in range(clients) for t in range(tasks)]).mean())
    output = {'id': artifact_id, 'training_performed': False, 'evaluated_cells': len(rows),
              'max_absolute_cell_error': error, 'tolerance': tolerance,
              'matches_archived_result': error <= tolerance, 'summary': summary,
              'acquisition': acquisition, 'rows': rows}
    return output


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--id', default='S2-MOTION-Base-seed0')
    parser.add_argument('--artifacts', type=Path, default=root / 'artifacts')
    parser.add_argument('--splits', type=Path, default=root / 'splits')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--output', type=Path, default=Path('outputs/pretrained-evaluation'))
    args = parser.parse_args()
    if args.list:
        print(json.dumps(read_json(args.artifacts / 'catalog.json'), indent=2))
        return 0
    if args.threads < 1 or not 0 <= args.tolerance < 1:
        parser.error('threads must be positive and tolerance must be in [0, 1)')
    torch.set_num_threads(args.threads)
    result = evaluate(args.artifacts, args.splits, args.id, args.data_root, args.device, args.tolerance)
    rows = result.pop('rows')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'per_cell.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['matches_archived_result'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
