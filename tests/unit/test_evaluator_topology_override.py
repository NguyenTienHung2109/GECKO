from __future__ import annotations

import pytest
import torch

from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.models.backbones import GECKOGraphModel

from tests.helpers import make_stream


def _model(stream):
    return GECKOGraphModel(
        stream.scenario.num_features,
        stream.scenario.num_classes,
        hidden_size=8,
        num_layers=2,
        problem_type=stream.scenario.problem_type,
    )


def test_evaluator_uses_owned_strict_local_topology_override_without_stream_mutation():
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    evaluator = FederatedEvaluator(stream)
    model = _model(stream)
    client_id = 0
    task_id = stream.orders.global_task(client_id, 0)
    base = stream.partition.client_graphs[client_id].edge_index
    base_before = base.clone()
    override = torch.cat((base, torch.tensor([[0], [0]], dtype=torch.long)), dim=1)
    expected = override.clone()
    seen = []
    original = model.forward_queries

    def recording(features, edges, queries, problem_type):
        seen.append(edges.detach().cpu().clone())
        return original(features, edges, queries, problem_type)

    model.forward_queries = recording
    evaluator.evaluate(
        model,
        client_id,
        task_id,
        {task_id},
        edge_index_override=override,
    )
    override.zero_()

    assert seen
    assert torch.equal(seen[0], expected)
    assert torch.equal(base, base_before)


def test_evaluator_topology_override_fails_closed_on_remote_or_reference_edges():
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    model = _model(stream)
    client_id = 0
    task_id = stream.orders.global_task(client_id, 0)
    num_nodes = stream.partition.client_graphs[client_id].node_features.shape[0]
    evaluator = FederatedEvaluator(stream)

    with pytest.raises(ValueError, match="strict-local"):
        evaluator.evaluate(
            model,
            client_id,
            task_id,
            {task_id},
            edge_index_override=torch.tensor(
                [[0, num_nodes], [1, 0]], dtype=torch.long
            ),
        )
    with pytest.raises(ValueError, match="int64 shape"):
        evaluator.evaluate(
            model,
            client_id,
            task_id,
            {task_id},
            edge_index_override=torch.zeros((2, 1), dtype=torch.float32),
        )

    reference = FederatedEvaluator(stream, reference_view=object())
    with pytest.raises(ValueError, match="centralized references"):
        reference.evaluate(
            model,
            client_id,
            task_id,
            {task_id},
            edge_index_override=torch.empty((2, 0), dtype=torch.long),
        )


def test_predict_central_accepts_the_same_validated_override():
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    evaluator = FederatedEvaluator(stream)
    model = _model(stream)
    client_id = 0
    task_id = stream.orders.global_task(client_id, 0)
    base = stream.partition.client_graphs[client_id].edge_index
    override = torch.cat((base, torch.tensor([[0], [0]], dtype=torch.long)), dim=1)
    logits, labels, query_ids = evaluator.predict_central(
        model,
        client_id,
        task_id,
        {task_id},
        edge_index_override=override,
    )
    assert logits.shape[0] == labels.shape[0] == query_ids.shape[0]
