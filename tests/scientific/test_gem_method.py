"""GEM behavior without historical promotion scripts or private result files."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from gecko.algorithms.continual.gem import GEMAlgorithm, project_gem_gradient
from gecko.algorithms.method_config import resolve_method_config
from gecko.engine.coordinator import FederatedCoordinator
from tests.helpers import make_stream


def method_config() -> dict:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "gem_uefa_v1",
        "strategy": {"name": "fedavg", "parameters": {}},
        "continual_method": {
            "name": "GEM",
            "parameters": {
                "memory_size": 8,
                "margin": 0.5,
                "projection_epsilon": 0.001,
                "violation_tolerance": 1.0e-10,
            },
        },
    }


def coordinator(stream) -> FederatedCoordinator:
    return FederatedCoordinator(
        stream, "fedavg", "GEM", model_name="uefa_gcn", model_seed=811,
        method_config=method_config(),
    )


def test_projection_matches_independent_orthogonal_quadratic_oracle() -> None:
    gradient = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    memories = torch.eye(2, dtype=torch.float64)
    projected = project_gem_gradient(gradient, memories, margin=0.5, epsilon=0.001)
    expected = gradient + torch.tensor([1.0 / 1.001, 2.0 / 1.001], dtype=torch.float64)
    assert torch.allclose(projected, expected, atol=1.0e-10, rtol=0.0)
    assert bool((memories @ projected >= -0.0021).all())


def test_projection_is_exact_noop_without_conflicting_memories() -> None:
    gradient = torch.tensor([1.0, 2.0])
    assert torch.equal(project_gem_gradient(gradient, torch.eye(2)), gradient)


@pytest.mark.parametrize("problem,setting,tasks", [
    pytest.param("NC", "task", 2, id="S1-family"),
    pytest.param("NC", "class", 2, id="S2-family"),
    pytest.param("LC", "task", 2, id="S4-family"),
    pytest.param("LC", "class", 2, id="S5-family"),
    pytest.param("LC", "domain", 4, id="S6-family"),
])
def test_fedavg_executes_projection_with_bounded_private_memory(problem, setting, tasks) -> None:
    stream = make_stream(problem, setting, tasks, 0, "mild", "synchronized")
    result = coordinator(stream).run()
    assert result["resource_ledger"]["training_model_uplink_bytes"] > 0
    assert result["resource_ledger"]["training_model_downlink_bytes"] > 0
    for diagnostics in result["client_method_diagnostics"].values():
        assert diagnostics["method"] == "GEM"
        assert diagnostics["num_consolidated_tasks"] == tasks
        assert diagnostics["remembered_query_count"] <= 8
        assert diagnostics["constraint_count"] == min(tasks - 1, 8)
        assert diagnostics["minimum_dot_after"] >= -0.1


def test_private_memory_is_deterministic_and_rejects_corrupted_checkpoint() -> None:
    stream = make_stream("LC", "task", 2, 827, "mild", "synchronized")
    first, second = coordinator(stream), coordinator(stream)
    assert first.run()["summary"] == second.run()["summary"]
    for client_id, client in first.clients.items():
        method = client.algorithm
        assert method.private_state_checksum() == second.clients[client_id].algorithm.private_state_checksum()
        assert method.replay_payload_bytes() > 0
        restored = GEMAlgorithm(
            memory_size=8, margin=0.5, projection_epsilon=0.001,
            violation_tolerance=1.0e-10, problem_type="LC", incremental_setting="task",
            client_id=client_id, seed=stream.config.seed,
        )
        state = deepcopy(method.save_method_state())
        restored.load_method_state(state)
        assert restored.private_state_checksum() == method.private_state_checksum()
        corrupted = deepcopy(state)
        corrupted["memories"][0]["labels"] = corrupted["memories"][0]["labels"][:-1]
        with pytest.raises(ValueError, match="queries or labels"):
            restored.load_method_state(corrupted)


def test_gem_does_not_claim_link_prediction_support() -> None:
    result = resolve_method_config(method_config(), problem_type="LP", incremental_setting="domain")
    assert not result.runnable
    assert result.support_status == "unsupported_scientific"
