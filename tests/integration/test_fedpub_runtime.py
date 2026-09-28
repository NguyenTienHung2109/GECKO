from __future__ import annotations

from typing import Any

import pytest
import torch

from gecko.engine.coordinator import FederatedCoordinator
from gecko.algorithms.method_config import UnsupportedMethodConfigError
from gecko.algorithms.method_config import validate_method_config
from gecko.algorithms.federated.fedpub import FedPUBStrategy
from gecko.algorithms.federated.fedpub import build_fedpub_proxy_artifact
from gecko.algorithms.federated.fedpub import fedpub_effective_state
from gecko.algorithms.federated.fedpub import fedpub_personalized_aggregate
from gecko.algorithms.federated.fedpub import fedpub_similarity_weights

from tests.helpers import make_stream


def _config(*, lambda2: float = 1e-3, proxy_seed: int = 0) -> dict[str, Any]:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "fed_pub_uefa_v1",
        "strategy": {
            "name": "fed_pub",
            "parameters": {
                "tau": 3.0,
                "lambda1": 1e-3,
                "lambda2": lambda2,
                "proxy_seed": proxy_seed,
            },
        },
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def test_fedpub_proxy_artifact_is_deterministic_unlabeled_sbm():
    first = build_fedpub_proxy_artifact(feature_dim=7, seed=11)
    second = build_fedpub_proxy_artifact(feature_dim=7, seed=11)
    different = build_fedpub_proxy_artifact(feature_dim=7, seed=12)

    assert first.sha256 == second.sha256
    assert first.sha256 != different.sha256
    payload = first.payload.materialize()
    assert payload["features"].shape == (500, 7)
    assert payload["edge_index"].shape[0] == 2
    assert payload["block_ids"].shape == (500,)
    assert first.metadata_dict()["labels"] is False
    assert first.metadata_dict()["p_out"] == 0.01


def test_fedpub_private_mask_is_in_the_differentiable_local_forward():
    stream = make_stream("NC", "class", 2, order_profile="synchronized", rounds=1)
    coordinator = FederatedCoordinator(
        stream,
        "fed_pub",
        "Bare",
        model_name="uefa_gcn",
        model_seed=89,
        method_config=_config(),
    )
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    strategy = runtime.strategy
    client = coordinator.clients[0]
    task_id = stream.orders.global_task(0, 0)
    context = client.build_method_context(
        stream.shards[0][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    shared_keys = client.parameter_policy.shareable_keys(client.model)
    masks = strategy.trainable_parameters(client.model, context, shared_keys)
    baseline = strategy.forward_queries(client.model, context, context.train_queries)
    with torch.no_grad():
        for mask in masks:
            mask.zero_()
    masked = strategy.forward_queries(client.model, context, context.train_queries)

    assert not torch.allclose(baseline, masked)


def test_fedpub_begin_gcn_exposes_masked_functional_encoder():
    stream = make_stream("NC", "class", 2, order_profile="synchronized", rounds=1)
    result = FederatedCoordinator(
        stream,
        "fed_pub",
        "Bare",
        model_name="begin_gcn",
        model_seed=91,
        method_config=_config(),
    ).run()

    diagnostics = result["strategy_diagnostics"]
    assert diagnostics["strategy_version"] == "uefa-fed-pub-strategy-v3"
    assert diagnostics["functional_embedding"] == (
        "mean_final_encoder_on_unlabeled_proxy"
    )


@pytest.mark.gpu
def test_fedpub_similarity_and_effective_state_oracles():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = {"w": torch.tensor([1.0, 2.0], device=device)}
    masks = {"w": torch.tensor([0.25, 1.0])}
    effective = fedpub_effective_state(state, masks)["w"]
    assert effective.device == device
    assert effective.dtype == state["w"].dtype
    assert torch.equal(effective.cpu(), torch.tensor([0.25, 2.0]))

    embeddings = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 1.0]),
    }
    weights = fedpub_similarity_weights(embeddings, tau=3.0)
    assert weights[0][0] > weights[0][1]
    personalized = fedpub_personalized_aggregate(
        effective_states={
            0: {"w": torch.tensor([1.0])},
            1: {"w": torch.tensor([3.0])},
        },
        weights=weights,
    )
    assert personalized[0]["w"] < personalized[1]["w"]


def test_fedpub_proximity_loss_changes_shared_parameter_gradients():
    strategy = FedPUBStrategy(lambda2=0.1)
    shared = {"weight": torch.ones(2, 2)}
    from gecko.engine.protocol import ParameterEntry
    from gecko.engine.protocol import ParameterManifest
    strategy.initialize(
        shared,
        ParameterManifest((
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(2, 2),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )),
        [0],
    )
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    context = type("Context", (), {"client_id": 0})()
    masks = strategy.trainable_parameters(model, context, ("weight",))
    optimizer = torch.optim.Adam(tuple(model.parameters()) + masks, lr=0.1)
    optimizer.zero_grad()
    loss = strategy.augment_loss(model, context, model.weight.sum() * 0.0, ("weight",))
    loss.backward()
    strategy.after_backward(model, context, ("weight",), learning_rate=0.1)
    optimizer.step()
    strategy.after_optimizer_step(model, context, ("weight",))

    assert torch.all(model.weight.grad > 0)
    updated_mask = strategy.state_dict()["masks"][0]["weight"]
    assert torch.any(updated_mask != 1.0)
    diagnostics = strategy.diagnostics()
    assert diagnostics["fidelity"] == "paper_faithful_benchmark_adaptation"
    assert diagnostics["mask_optimizer"] == "private_reset_per_round_adam"
    assert diagnostics["last_loss_terms"]["client-0"]["proximity_l2"] > 0
    assert diagnostics["last_mask_updates"]["client-0"]["mean_abs_delta"] > 0


def test_fedpub_two_inner_steps_make_sparse_mask_pruning_reachable_on_cpu():
    """The disclosed full-batch adaptation must activate its sparse mask gate."""

    strategy = FedPUBStrategy(lambda1=1e-3, lambda2=1e-3)
    shared = {"weight": torch.ones(2, 2)}
    from gecko.engine.protocol import ParameterEntry
    from gecko.engine.protocol import ParameterManifest

    strategy.initialize(
        shared,
        ParameterManifest((
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(2, 2),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )),
        [0],
    )
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    context = type("Context", (), {"client_id": 0})()
    for _ in range(80):
        masks = strategy.trainable_parameters(model, context, ("weight",))
        optimizer = torch.optim.Adam(
            tuple(model.parameters()) + masks, lr=0.01
        )
        for _ in range(strategy.local_optimizer_steps_per_epoch):
            optimizer.zero_grad()
            loss = strategy.augment_loss(
                model, context, model.weight.sum() * 0.0, ("weight",)
            )
            loss.backward()
            strategy.after_backward(model, context, ("weight",), learning_rate=0.01)
            optimizer.step()
            strategy.after_optimizer_step(model, context, ("weight",))

    diagnostics = strategy.diagnostics()
    assert diagnostics["last_mask_updates"]["client-0"]["active_fraction"] < 1.0
    evaluation = strategy.select_evaluation(0, 0)
    assert torch.count_nonzero(evaluation.model_state["weight"]) == 0


@pytest.mark.parametrize("incremental", ["task", "class"])
def test_fedpub_stateful_runtime_is_personalized_and_accounts_proxy_wire(
    incremental: str,
):
    stream = make_stream("NC", incremental, 2, order_profile="synchronized", rounds=1)
    coordinator = FederatedCoordinator(
        stream,
        "fed_pub",
        "Bare",
        model_name="uefa_gcn",
        model_seed=97,
        method_config=_config(),
    )
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    assert isinstance(runtime.strategy, FedPUBStrategy)

    result = coordinator.run()

    assert (
        result["method_config_resolution"]["support_status"]
        == "implemented_unverified"
    )
    assert result["method_config_benchmark_eligible"] is False
    assert result["evaluation_model"] == "strategy"
    assert all(
        set(record["evaluation_model_sources"].values()) == {"personalized"}
        for record in result["stages"]
    )
    diagnostics = result["strategy_diagnostics"]
    assert diagnostics["strategy_version"] == "uefa-fed-pub-strategy-v3"
    assert diagnostics["proxy_serialized_bytes"] > 0
    assert diagnostics["mask_optimizer"] == "private_reset_per_round_adam"
    assert diagnostics["local_optimizer_steps_per_epoch"] == 2
    assert diagnostics["functional_embedding"] == "mean_final_encoder_on_unlabeled_proxy"
    assert any(
        float(values["proximity_l2"]) > 0.0
        for values in diagnostics["last_loss_terms"].values()
    )
    assert result["training_budget"]["strategy_optimizer_steps_per_local_epoch"] == 2
    assert result["training_budget"]["effective_optimizer_steps_per_round"] == 2
    assert diagnostics["clients_with_last_mask_updates"]
    assert result["personalized_parameter_count"] > 0
    ledger = result["resource_ledger"]
    assert ledger["training_auxiliary_uplink_bytes"] > 0
    assert ledger["artifact_distribution_bytes"] > 0
    assert ledger["method_artifact_bytes"] > 0
    assert ledger["personalized_model_bytes"] > 0
    assert ledger["evaluation_sync_bytes"] > 0


def test_fedpub_nc_domain_and_unopened_compositions_fail_closed():
    with pytest.raises(UnsupportedMethodConfigError) as domain:
        validate_method_config(
            _config(), problem_type="NC", incremental_setting="domain"
        )
    assert domain.value.resolution.support_status == "unsupported_scientific"

    config = _config()
    config["continual_method"] = {
        "name": "DSLR",
        "parameters": {
            "beta": 0.1,
            "radius": 0.2,
            "structure_lambda": 0.5,
            "top_n": 5,
            "candidate_k": 50,
            "tau": 0.8,
            "structure_epochs": 99,
            "structure_learning_rate": 0.01,
            "structure_hidden_dim": 64,
            "structure_heads": 4,
            "replay_fraction": 0.05,
            "replay_ceiling_bytes": 16 * 1024 * 1024,
            "undirected": True,
            "selection_mode": "coverage_diversity",
            "structure_mode": "full",
        },
    }
    with pytest.raises(UnsupportedMethodConfigError) as composition:
        validate_method_config(
            config, problem_type="NC", incremental_setting="task"
        )
    assert composition.value.resolution.support_status == "unsupported_composition"


def test_fedpub_checkpoint_resume_is_equivalent_to_uninterrupted_run(tmp_path, monkeypatch):
    stream = make_stream(
        "NC", "class", 2, seed=51, order_profile="synchronized", rounds=2
    )
    baseline = FederatedCoordinator(
        stream,
        "fed_pub",
        "Bare",
        model_name="uefa_gcn",
        model_seed=101,
        method_config=_config(proxy_seed=5),
    )
    baseline_result = baseline.run()

    interrupted = FederatedCoordinator(
        stream,
        "fed_pub",
        "Bare",
        model_name="uefa_gcn",
        model_seed=101,
        method_config=_config(proxy_seed=5),
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    runtime = interrupted._stateful_runtime
    assert runtime is not None
    original_write = runtime._write_checkpoint

    class SimulatedInterruption(RuntimeError):
        pass

    def write_then_interrupt(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "round-00000001":
            raise SimulatedInterruption

    monkeypatch.setattr(runtime, "_write_checkpoint", write_then_interrupt)
    with pytest.raises(SimulatedInterruption):
        interrupted.run()

    resumed = FederatedCoordinator(
        stream,
        "fed_pub",
        "Bare",
        model_name="uefa_gcn",
        model_seed=101,
        method_config=_config(proxy_seed=5),
        resume_from=tmp_path / "round-00000001.manifest.json",
    )
    resumed_result = resumed.run()

    assert torch.allclose(
        baseline_result["A[k,s,t]"],
        resumed_result["A[k,s,t]"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert (
        baseline_result["strategy_diagnostics"]
        == resumed_result["strategy_diagnostics"]
    )
    assert resumed_result["resume_validation"]["identity_matched"] is True
