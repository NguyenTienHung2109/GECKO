from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from gecko.engine.protocol import BroadcastPayload
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import MethodArtifact
from gecko.engine.protocol import ParameterEntry
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import ResourceLedger
from gecko.engine.protocol import RoundContext
from gecko.algorithms.context import ClientMethodContext
from gecko.algorithms.topology import TopologyOverlay


def _context() -> ClientMethodContext:
    features = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    queries = torch.tensor([0, 2], dtype=torch.long)
    labels = torch.tensor([1, 0], dtype=torch.long)

    def forward(model, node_features, edge_index, values):
        assert isinstance(model, torch.nn.Module)
        assert int(edge_index.max()) < node_features.shape[0]
        return node_features[values].sum(dim=1, keepdim=True)

    def encode(model, node_features, edge_index, layer_index):
        assert isinstance(model, torch.nn.Module)
        assert layer_index is None
        assert int(edge_index.max()) < node_features.shape[0]
        return node_features * 2

    return ClientMethodContext(
        client_id=1,
        global_task_id=3,
        stage_index=0,
        round_index=2,
        problem_type="NC",
        incremental_setting="class",
        train_queries=queries,
        train_labels=labels,
        valid_class_mask=torch.tensor([True, True, False]),
        node_features=features,
        base_edge_index=edges,
        context_edge_index=None,
        forward_queries=forward,
        encode_nodes=encode,
    )


def test_client_method_context_owns_current_local_tensors_and_exposes_capabilities():
    context = _context()
    model = torch.nn.Identity()
    assert context.global_task_id == 3
    assert callable(context.forward_queries)
    assert callable(context.encode_nodes)
    assert torch.equal(context.forward_queries(model), torch.tensor([[3.0], [21.0]]))
    assert torch.equal(
        context.encode_nodes(model),
        torch.arange(12, dtype=torch.float32).reshape(4, 3) * 2,
    )

    leaked = context.node_features
    leaked.zero_()
    assert context.node_features.sum() > 0
    queries = context.train_queries
    queries.fill_(3)
    assert torch.equal(context.train_queries, torch.tensor([0, 2]))
    with pytest.raises(FrozenInstanceError):
        context.client_id = 9


def test_client_method_context_has_no_central_or_global_node_fields_and_rejects_remote_indices():
    context = _context()
    model = torch.nn.Identity()
    forbidden = {
        "stream",
        "ownership",
        "node_owner",
        "local_to_global",
        "global_to_local",
        "validation_queries",
        "validation_labels",
        "test_queries",
        "test_labels",
        "future_tasks",
    }
    assert forbidden.isdisjoint(context.__dataclass_fields__)
    with pytest.raises(ValueError, match="non-local"):
        context.forward_queries(model, torch.tensor([4]))
    with pytest.raises(ValueError, match="non-local"):
        context.encode_nodes(
            model, edge_index=torch.tensor([[0, 4], [1, 2]], dtype=torch.long)
        )


def test_frozen_tensor_map_and_payload_defensively_clone_wire_state():
    source = torch.tensor([1.0, 2.0])
    state = FrozenTensorMap({"weight": source})
    source.zero_()
    assert torch.equal(state["weight"], torch.tensor([1.0, 2.0]))
    exposed = state["weight"]
    exposed.fill_(7)
    assert torch.equal(state["weight"], torch.tensor([1.0, 2.0]))
    with pytest.raises(AttributeError):
        state.extra = 1

    round_context = RoundContext(
        stage_index=0,
        round_index=1,
        participant_ids=(2, 0),
        client_global_task_ids={0: 4, 2: 3},
    )
    assert round_context.client_global_task_ids == ((2, 3), (0, 4))
    assert round_context.task_for(0) == 4
    payload = BroadcastPayload(
        client_id=2,
        round_context=round_context,
        reason="training",
        model_state={"weight": torch.ones(2)},
        metadata={"reason": "round", "ids": [2, 0]},
    )
    assert payload.model_state.payload_bytes == 8
    assert payload.metadata_dict() == {"ids": [2, 0], "reason": "round"}
    assert not hasattr(payload, "train_labels")


def test_round_context_rejects_task_maps_that_expose_nonparticipant_entries():
    with pytest.raises(ValueError, match="exactly"):
        RoundContext(
            stage_index=0,
            round_index=0,
            participant_ids=(0,),
            client_global_task_ids={0: 1, 9: 1},
        )


def test_topology_overlay_is_owned_deterministic_and_never_mutates_base():
    base = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    original = base.clone()
    added = torch.tensor([[0], [3]], dtype=torch.long)
    deleted = torch.tensor([[1], [2]], dtype=torch.long)
    overlay = TopologyOverlay(
        client_id=0,
        global_task_id=2,
        method_name="DSLR",
        num_nodes=4,
        base_edge_index=base,
        added_edge_index=added,
        deleted_edge_index=deleted,
    )
    added.fill_(2)
    deleted.fill_(0)
    materialized = overlay.apply(base)
    assert torch.equal(base, original)
    assert torch.equal(
        materialized,
        torch.tensor([[0, 2, 0], [1, 3, 3]], dtype=torch.long),
    )
    escaped = overlay.added_edge_index
    escaped.zero_()
    assert torch.equal(overlay.added_edge_index, torch.tensor([[0], [3]]))
    assert len(overlay.overlay_id) == 64
    assert overlay.payload_bytes == 32


def test_topology_overlay_fails_closed_on_remote_absent_or_wrong_base_arcs():
    base = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="strict-local"):
        TopologyOverlay(
            client_id=0,
            global_task_id=0,
            method_name="DSLR",
            num_nodes=3,
            base_edge_index=base,
            added_edge_index=torch.tensor([[0], [3]], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="absent"):
        TopologyOverlay(
            client_id=0,
            global_task_id=0,
            method_name="DSLR",
            num_nodes=3,
            base_edge_index=base,
            deleted_edge_index=torch.tensor([[2], [0]], dtype=torch.long),
        )
    overlay = TopologyOverlay(
        client_id=0,
        global_task_id=0,
        method_name="DSLR",
        num_nodes=3,
        base_edge_index=base,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        overlay.apply(torch.tensor([[1, 0], [2, 1]], dtype=torch.long))


def test_undirected_overlay_expands_one_logical_arc_to_both_orientations():
    base = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    overlay = TopologyOverlay(
        client_id=0,
        global_task_id=0,
        method_name="DSLR",
        num_nodes=3,
        base_edge_index=base,
        added_edge_index=torch.tensor([[1], [2]], dtype=torch.long),
        deleted_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        undirected=True,
    )
    assert set(map(tuple, overlay.added_edge_index.t().tolist())) == {(1, 2), (2, 1)}
    assert set(map(tuple, overlay.deleted_edge_index.t().tolist())) == {(0, 1), (1, 0)}
    assert set(map(tuple, overlay.apply(base).t().tolist())) == {(1, 2), (2, 1)}


def test_resource_ledger_preserves_legacy_payload_and_separates_new_resources():
    ledger = ResourceLedger(
        initialization_model_downlink_bytes=10,
        training_model_uplink_bytes=20,
        training_model_downlink_bytes=30,
        training_auxiliary_uplink_bytes=7,
        training_auxiliary_downlink_bytes=9,
        evaluation_sync_bytes=11,
        replay_bytes=100,
    )
    assert ledger.communication_payload_bytes == 50
    assert ledger.training_wire_bytes == 66
    assert ledger.evaluation_wire_bytes == 11
    assert ledger.total_wire_bytes == 87
    assert ledger.replay_bytes == 100
    with pytest.raises(ValueError, match="non-negative"):
        ResourceLedger(training_model_uplink_bytes=-1)


def test_parameter_manifest_is_disjoint_and_method_artifact_is_digest_locked():
    manifest = ParameterManifest(
        (
            ParameterEntry(
                "encoder.weight",
                "shared_trainable",
                (4, 3),
                "torch.float32",
                True,
                True,
            ),
            ParameterEntry("bn.running_mean", "local_buffer", (4,), "torch.float32"),
            ParameterEntry(
                "classifier.observed", "dynamic_metadata", (3,), "torch.bool"
            ),
        )
    )
    assert manifest.shared_trainable == ("encoder.weight",)
    assert manifest.local_buffers == ("bn.running_mean",)
    assert manifest.dynamic_metadata == ("classifier.observed",)
    with pytest.raises(ValueError, match="disjoint"):
        ParameterManifest(
            (
                ParameterEntry("same", "local_buffer"),
                ParameterEntry("same", "dynamic_metadata"),
            )
        )

    artifact = MethodArtifact.build(
        artifact_type="fed-pub-proxy",
        version="1",
        payload={"features": torch.arange(4, dtype=torch.float32)},
        metadata={"seed": 7},
        serialized_bytes=64,
    )
    assert len(artifact.sha256) == 64
    assert artifact.payload.payload_bytes == 16
    with pytest.raises(ValueError, match="digest"):
        MethodArtifact(
            artifact_type=artifact.artifact_type,
            version=artifact.version,
            sha256="0" * 64,
            payload=artifact.payload,
            metadata={"seed": 7},
            serialized_bytes=64,
        )


def test_evaluation_selection_fails_closed_on_ambiguous_state_ownership():
    EvaluationSelection(client_id=0, source="post_local")
    EvaluationSelection(
        client_id=0,
        source="personalized",
        model_state={"weight": torch.ones(1)},
        count_evaluation_sync=True,
    )
    with pytest.raises(ValueError, match="requires"):
        EvaluationSelection(client_id=0, source="shared")
    with pytest.raises(ValueError, match="must not carry"):
        EvaluationSelection(
            client_id=0,
            source="post_local",
            model_state={"weight": torch.ones(1)},
        )
