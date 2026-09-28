from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from torch import nn

from gecko.engine.parameter_manifest import build_parameter_manifest
from gecko.engine.parameter_policy import SharedParameterPolicy
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import StatefulStrategyProtocol
from gecko.algorithms.federated.fedgta import FedGTAStrategy
from gecko.algorithms.federated.fedgta import fedgta_moments
from gecko.algorithms.federated.fedgta import fedgta_origin_moments
from gecko.algorithms.federated.fedgta import fedgta_personalized_aggregate
from gecko.algorithms.federated.fedgta import fedgta_personalized_weights
from gecko.algorithms.federated.fedgta import fedgta_smoothing_confidence
from gecko.algorithms.federated.fedgta import propagate_strict_local_labels


def test_strict_local_propagation_uses_only_current_train_labels():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    labels, states, degrees = propagate_strict_local_labels(
        logits=logits,
        edge_index=edges,
        current_train_nodes=torch.tensor([0]),
        current_train_labels=torch.tensor([1]),
    )

    assert len(states) == 5
    assert torch.equal(labels[0], torch.tensor([0.0, 1.0]))
    assert torch.equal(degrees, torch.tensor([1.0, 2.0, 1.0]))

    # No held-out labels are an input: perturbing non-train logits changes only
    # model-probability-derived values, never an implicit label clamp.
    changed, _, _ = propagate_strict_local_labels(
        logits=logits + torch.tensor([[0.0, 0.0], [4.0, -4.0], [0.0, 0.0]]),
        edge_index=edges,
        current_train_nodes=torch.tensor([0]),
        current_train_labels=torch.tensor([1]),
    )
    assert torch.equal(changed[0], torch.tensor([0.0, 1.0]))
    assert not torch.equal(changed[1], labels[1])


def test_origin_moments_and_reversed_entropy_confidence_match_closed_forms():
    state = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    origin = torch.tensor([0.375, 0.625, 0.15625, 0.40625])
    assert torch.allclose(
        fedgta_origin_moments((state,), moment_order=2), origin
    )
    centered = torch.tensor([0.0, 0.0, 0.015625, 0.015625])
    assert torch.allclose(
        fedgta_moments((state,), moment_order=2, moment_type="hybrid"),
        torch.cat((origin, centered)),
    )
    confidence = fedgta_smoothing_confidence(
        propagated_labels=state, degrees=torch.tensor([1.0, 2.0])
    )
    expected = sum(
        degree
        * (
            state.shape[1] * torch.exp(torch.tensor(-1.0))
            + (row * row.log()).sum()
        )
        for row, degree in zip(state, (1.0, 2.0))
    )
    assert torch.allclose(confidence, expected)
    assert confidence > 0


def test_personalized_aggregation_is_thresholded_confidence_weighted_and_order_invariant():
    moments = {
        2: torch.tensor([1.0, 0.0]),
        7: torch.tensor([0.99, 0.01]),
        9: torch.tensor([0.0, 1.0]),
    }
    confidence = {2: torch.tensor(2.0), 7: torch.tensor(1.0), 9: torch.tensor(3.0)}
    weights = fedgta_personalized_weights(
        moments_by_client=moments,
        confidence_by_client=confidence,
        similarity_threshold=0.9,
    )
    assert set(weights[2]) == {2, 7}
    assert set(weights[7]) == {2, 7}
    assert set(weights[9]) == {9}
    assert torch.allclose(sum(weights[2].values()), torch.tensor(1.0))

    states = {client: {"w": torch.tensor([float(client)])} for client in moments}
    output = fedgta_personalized_aggregate(
        model_states=states, personalized_weights=weights
    )
    assert torch.allclose(output[2]["w"], torch.tensor([11.0 / 3.0]))
    assert torch.equal(output[9]["w"], torch.tensor([9.0]))

def test_paper_similarity_threshold_is_inclusive():
    weights = fedgta_personalized_weights(
        moments_by_client={
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([1.0, 0.0]),
        },
        confidence_by_client={0: 1.0, 1: 3.0},
        similarity_threshold=1.0,
    )

    assert set(weights[0]) == {0, 1}
    assert torch.allclose(weights[0][0], torch.tensor(0.25))
    assert torch.allclose(weights[0][1], torch.tensor(0.75))


def test_nonpositive_smoothing_confidence_is_rejected():
    with pytest.raises(ValueError, match="positive finite"):
        fedgta_personalized_weights(
            moments_by_client={0: torch.tensor([1.0, 0.0])},
            confidence_by_client={0: torch.tensor(-1.0)},
            similarity_threshold=0.5,
        )


def test_personalized_state_isolation_starts_from_owned_initial_shared_state():
    model = nn.Linear(3, 2, bias=False)
    policy = SharedParameterPolicy()
    shared = policy.extract(model)
    strategy = FedGTAStrategy()
    strategy.initialize(shared, build_parameter_manifest(model, policy), [4, 8])

    first = strategy.personalized_state(4)
    first["weight"].add_(10.0)
    assert not torch.equal(first["weight"], strategy.personalized_state(4)["weight"])
    assert torch.equal(strategy.personalized_state(4)["weight"], strategy.shared_state["weight"])
    assert torch.equal(strategy.personalized_state(8)["weight"], strategy.shared_state["weight"])

def test_personalized_broadcast_evaluation_and_checkpoint_round_trip():
    model = nn.Linear(3, 2, bias=False)
    policy = SharedParameterPolicy()
    shared = policy.extract(model)
    manifest = build_parameter_manifest(model, policy)
    strategy = FedGTAStrategy()
    strategy.initialize(shared, manifest, [4, 8])
    context = RoundContext(0, 0, (4, 8), {4: 0, 8: 0})

    training = strategy.prepare_payload(context, 4, "training")
    evaluation = strategy.select_evaluation(8, 0)
    assert training.resources.training_model_downlink_bytes == training.model_state.payload_bytes
    assert evaluation.source == "personalized"
    assert evaluation.count_evaluation_sync is True
    assert torch.equal(evaluation.model_state["weight"], strategy.personalized_state(8)["weight"])

    restored = FedGTAStrategy()
    restored.initialize(shared, manifest, [4, 8])
    restored.load_state_dict(strategy.state_dict())
    assert torch.equal(restored.shared_state["weight"], strategy.shared_state["weight"])
    assert torch.equal(
        restored.personalized_state(4)["weight"], strategy.personalized_state(4)["weight"]
    )

def test_typed_aggregation_personalizes_active_clients_and_preserves_inactive_state():
    model = nn.Linear(1, 1, bias=False)
    policy = SharedParameterPolicy()
    shared = policy.extract(model)
    manifest = build_parameter_manifest(model, policy)
    strategy = FedGTAStrategy(similarity_threshold=1.0)
    strategy.initialize(shared, manifest, [4, 8, 9])
    inactive_before = strategy.personalized_state(9)
    context = RoundContext(0, 0, (4, 8), {4: 0, 8: 0})
    uploads = (
        ClientUpload(
            client_id=4,
            global_task_id=0,
            weight=1,
            training_loss=0.0,
            model_state={"weight": torch.tensor([[4.0]])},
            auxiliary_state={
                "moments": torch.tensor([1.0, 0.0]),
                "confidence": torch.tensor([1.0]),
            },
        ),
        ClientUpload(
            client_id=8,
            global_task_id=0,
            weight=999,
            training_loss=0.0,
            model_state={"weight": torch.tensor([[8.0]])},
            auxiliary_state={
                "moments": torch.tensor([1.0, 0.0]),
                "confidence": torch.tensor([3.0]),
            },
        ),
    )

    result = strategy.aggregate(context, uploads)
    assert torch.equal(strategy.personalized_state(4)["weight"], torch.tensor([[7.0]]))
    assert torch.equal(strategy.personalized_state(8)["weight"], torch.tensor([[7.0]]))
    assert torch.equal(strategy.personalized_state(9)["weight"], inactive_before["weight"])
    assert torch.equal(result.shared_state["weight"], torch.tensor([[7.0]]))
    assert dict(result.diagnostics)["aggregation"] == (
        "thresholded_confidence_weighted_personalized"
    )
    assert isinstance(strategy, StatefulStrategyProtocol)

    restored = FedGTAStrategy(similarity_threshold=1.0)
    restored.initialize(shared, manifest, [4, 8, 9])
    restored.load_state_dict(strategy.state_dict())
    assert torch.equal(restored.personalized_state(4)["weight"], torch.tensor([[7.0]]))
    assert torch.equal(
        restored.personalized_state(9)["weight"], inactive_before["weight"]
    )
    assert restored.diagnostics()["completed_rounds"] == 1
    assert restored.diagnostics()["clients_with_last_known_statistics"] == [4, 8]

def test_fedgta_k1_aggregate_preserves_the_single_participant_state():
    model = nn.Linear(1, 1, bias=False)
    policy = SharedParameterPolicy()
    shared = policy.extract(model)
    strategy = FedGTAStrategy()
    strategy.initialize(shared, build_parameter_manifest(model, policy), [4])
    context = RoundContext(0, 0, (4,), {4: 0})
    upload = ClientUpload(
        client_id=4,
        global_task_id=0,
        weight=1,
        training_loss=0.0,
        model_state={"weight": torch.tensor([[5.0]])},
        auxiliary_state={
            "moments": torch.tensor([1.0, 0.0]),
            "confidence": torch.tensor([2.0]),
        },
    )

    result = strategy.aggregate(context, (upload,))

    assert torch.equal(strategy.personalized_state(4)["weight"], torch.tensor([[5.0]]))
    assert torch.equal(result.shared_state["weight"], torch.tensor([[5.0]]))
    assert dict(result.diagnostics)["participants"] == (4,)
    assert strategy.diagnostics()["completed_rounds"] == 1


def test_finalize_upload_collects_only_strict_local_current_label_statistics():
    model = nn.Linear(1, 2, bias=False)
    policy = SharedParameterPolicy()
    shared = policy.extract(model)
    strategy = FedGTAStrategy(moment_order=2)
    strategy.initialize(shared, build_parameter_manifest(model, policy), [4])

    class StrictLocalContext:
        problem_type = "NC"
        global_task_id = 3
        node_features = torch.zeros(3, 1)
        train_queries = torch.tensor([0])
        train_labels = torch.tensor([1])
        valid_class_mask = None
        effective_edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )

        @staticmethod
        def forward_queries(_model, queries):
            logits = torch.tensor([[2.0, 0.0], [0.1, 1.0], [1.0, 0.2]])
            return logits[queries]

    client = SimpleNamespace(client_id=4, model=model)
    local_result = SimpleNamespace(
        client_id=4,
        global_task_id=3,
        shared_state=shared,
        weight=1,
        training_loss=0.25,
    )
    upload = strategy.finalize_upload(client, local_result, StrictLocalContext())

    assert set(upload.auxiliary_state) == {"moments", "confidence"}
    assert upload.auxiliary_state["moments"].numel() == 20
    assert torch.isfinite(upload.auxiliary_state["confidence"]).all()
    assert upload.resources.training_model_uplink_bytes == upload.model_state.payload_bytes
    assert upload.resources.training_auxiliary_uplink_bytes == upload.auxiliary_state.payload_bytes

def test_held_out_label_perturbation_cannot_change_fedgta_upload_statistics():
    model = nn.Linear(1, 2, bias=False)
    policy = SharedParameterPolicy()
    shared = policy.extract(model)
    manifest = build_parameter_manifest(model, policy)

    class StrictLocalContext:
        problem_type = "NC"
        global_task_id = 1
        node_features = torch.zeros(3, 1)
        train_queries = torch.tensor([0])
        train_labels = torch.tensor([1])
        valid_class_mask = None
        effective_edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        validation_labels = torch.tensor([0, 1, 0])
        test_labels = torch.tensor([1, 1, 1])

        @staticmethod
        def forward_queries(_model, queries):
            logits = torch.tensor([[2.0, 0.0], [0.1, 1.0], [1.0, 0.2]])
            return logits[queries]

    client = SimpleNamespace(client_id=4, model=model)
    result = SimpleNamespace(
        client_id=4,
        global_task_id=1,
        shared_state=shared,
        weight=1,
        training_loss=0.0,
    )
    first = FedGTAStrategy(moment_order=2)
    first.initialize(shared, manifest, [4])
    first_upload = first.finalize_upload(client, result, StrictLocalContext())

    StrictLocalContext.validation_labels = torch.tensor([1, 0, 1])
    StrictLocalContext.test_labels = torch.tensor([0, 0, 0])
    second = FedGTAStrategy(moment_order=2)
    second.initialize(shared, manifest, [4])
    second_upload = second.finalize_upload(client, result, StrictLocalContext())

    assert torch.equal(
        first_upload.auxiliary_state["moments"], second_upload.auxiliary_state["moments"]
    )
    assert torch.equal(
        first_upload.auxiliary_state["confidence"], second_upload.auxiliary_state["confidence"]
    )
