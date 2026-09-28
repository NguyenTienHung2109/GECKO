from __future__ import annotations

import random

import numpy as np
import torch

from gecko.engine import FederatedCoordinator
from gecko.engine.parameter_manifest import build_parameter_manifest
from gecko.algorithms.federated.legacy import LegacyStrategyAdapter
from gecko.algorithms.continual.bare import BareAlgorithm
from gecko.models.capabilities import attach_v2_model_capabilities

from tests.helpers import make_stream


def _adapter(coordinator, strategy_name: str) -> LegacyStrategyAdapter:
    adapter = LegacyStrategyAdapter(strategy_name)
    adapter.initialize(
        coordinator.global_state,
        build_parameter_manifest(
            coordinator.global_model, coordinator.parameter_policy
        ),
        coordinator.clients,
    )
    return adapter


def test_stateful_client_legacy_adapter_matches_unchanged_local_update_exactly():
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    legacy = FederatedCoordinator(
        stream,
        "local_only",
        "Bare",
        model_name="uefa_gcn",
        model_seed=19,
    )
    stateful = FederatedCoordinator(
        stream,
        "local_only",
        "Bare",
        model_name="uefa_gcn",
        model_seed=19,
    )
    legacy_client = legacy.clients[0]
    stateful_client = stateful.clients[0]
    attach_v2_model_capabilities(stateful_client.model)
    task_id = stream.orders.global_task(0, 0)
    shard = stream.shards[0][task_id]

    expected = legacy_client.update(
        shard,
        global_task_id=task_id,
        server_state=legacy.global_state,
        strategy="local_only",
    )
    context = stateful_client.build_method_context(
        shard,
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    stateful_client.begin_task_stateful(context)
    actual = stateful_client.update_stateful(
        context,
        server_state=stateful.global_state,
        strategy=_adapter(stateful, "local_only"),
    )

    assert actual.client_id == expected.client_id
    assert actual.global_task_id == expected.global_task_id
    assert actual.weight == expected.weight
    assert actual.training_loss == expected.training_loss
    assert actual.model_state.payload_bytes == expected.communication_bytes
    for name, tensor in expected.shared_state.items():
        assert torch.equal(actual.model_state[name], tensor)


class RecordingAlgorithm(BareAlgorithm):
    def __init__(self, events, **kwargs):
        super().__init__(**kwargs)
        self.events = events

    def before_round(self, context):
        self.events.append("before_round")

    def after_backward(self, model, context):
        self.events.append("after_backward")
        super().after_backward(model, context)

    def after_round(self, model, context):
        self.events.append("after_round")
        super().after_round(model, context)


class RecordingStrategy(LegacyStrategyAdapter):
    def __init__(self, events):
        super().__init__("fedavg")
        self.events = events

    def transform_gradients(self, model, method_context, shared_keys):
        self.events.append("transform_gradients")


def test_stateful_client_applies_strategy_correction_before_algorithm_mask():
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_seed=23,
    )
    client = coordinator.clients[0]
    attach_v2_model_capabilities(client.model)
    events = []
    client.algorithm = RecordingAlgorithm(
        events,
        problem_type=client.scenario.problem_type,
        incremental_setting=client.scenario.incremental_type,
        client_id=client.client_id,
    )
    client.state.continual_state = client.algorithm.state
    strategy = RecordingStrategy(events)
    strategy.initialize(
        coordinator.global_state,
        build_parameter_manifest(
            coordinator.global_model, coordinator.parameter_policy
        ),
        coordinator.clients,
    )
    task_id = stream.orders.global_task(0, 0)
    context = client.build_method_context(
        stream.shards[0][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )

    client.update_stateful(
        context,
        server_state=coordinator.global_state,
        strategy=strategy,
    )

    assert events == [
        "before_round",
        "transform_gradients",
        "after_backward",
        "after_round",
    ]


def test_option_i_control_gradient_restores_model_method_gradients_modes_and_rng():
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_seed=29,
    )
    client = coordinator.clients[0]
    task_id = stream.orders.global_task(0, 0)
    context = client.build_method_context(
        stream.shards[0][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    client.algorithm.state["marker"] = torch.tensor([7.0])
    client.model.train()
    client.model.layers[0].eval()
    for parameter in client.model.parameters():
        parameter.grad = torch.randn_like(parameter)

    model_before = {
        name: value.detach().clone()
        for name, value in client.model.state_dict().items()
    }
    gradients_before = {
        name: parameter.grad.detach().clone()
        for name, parameter in client.model.named_parameters()
    }
    modes_before = {
        name: module.training for name, module in client.model.named_modules()
    }
    method_before = client.algorithm.save_method_state()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    numpy_before = (
        numpy_before[0],
        numpy_before[1].copy(),
        numpy_before[2],
        numpy_before[3],
        numpy_before[4],
    )
    torch_before = torch.get_rng_state().clone()
    shared_at_broadcast = {
        name: torch.zeros_like(value)
        for name, value in coordinator.global_state.items()
    }

    controls = client.control_gradient_at_shared(
        context,
        shared_at_broadcast,
        client.parameter_policy.shareable_keys(client.model),
    )

    assert tuple(controls) == client.parameter_policy.shareable_keys(client.model)
    assert all(torch.isfinite(value).all() for value in controls.values())
    for name, value in client.model.state_dict().items():
        assert torch.equal(value, model_before[name])
    for name, parameter in client.model.named_parameters():
        assert torch.equal(parameter.grad, gradients_before[name])
    assert client.algorithm.save_method_state().keys() == method_before.keys()
    assert torch.equal(client.algorithm.state["marker"], method_before["marker"])
    assert {
        name: module.training for name, module in client.model.named_modules()
    } == modes_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_post_strategy_nc_output_mask_zeros_inactive_corrected_rows():
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_seed=31,
    )
    client = coordinator.clients[0]
    attach_v2_model_capabilities(client.model)

    class InjectingStrategy(LegacyStrategyAdapter):
        def __init__(self):
            super().__init__("fedavg")

        def transform_gradients(self, model, method_context, shared_keys):
            for parameter in model.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.add_(3.0)

    strategy = InjectingStrategy()
    strategy.initialize(
        coordinator.global_state,
        build_parameter_manifest(
            coordinator.global_model, coordinator.parameter_policy
        ),
        coordinator.clients,
    )
    task_id = stream.orders.global_task(0, 0)
    context = client.build_method_context(
        stream.shards[0][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    mask = context.valid_class_mask
    assert mask is not None and bool(mask.any()) and bool((~mask).any())
    weight_before = client.model.node_head.weight.detach().clone()
    bias_before = client.model.node_head.bias.detach().clone()

    client.update_stateful(
        context,
        server_state=coordinator.global_state,
        strategy=strategy,
    )

    inactive = ~mask.cpu()
    active = mask.cpu()
    assert torch.equal(client.model.node_head.weight[inactive], weight_before[inactive])
    assert torch.equal(client.model.node_head.bias[inactive], bias_before[inactive])
    assert not torch.equal(client.model.node_head.weight[active], weight_before[active])
    assert not torch.equal(client.model.node_head.bias[active], bias_before[active])


def test_strategy_may_use_full_output_loss_without_changing_context_mask() -> None:
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_seed=31,
    )
    client = coordinator.clients[0]
    task_id = stream.orders.global_task(0, 0)
    context = client.build_method_context(
        stream.shards[0][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    mask = context.valid_class_mask
    assert mask is not None and bool(mask.any()) and bool((~mask).any())

    class FullOutputStrategy(LegacyStrategyAdapter):
        def __init__(self):
            super().__init__("fedavg")

        def training_class_mask(self, method_context):
            assert method_context.valid_class_mask is not None
            return None

    strategy = FullOutputStrategy()
    strategy.initialize(
        coordinator.global_state,
        build_parameter_manifest(
            coordinator.global_model, coordinator.parameter_policy
        ),
        coordinator.clients,
    )
    weight_before = client.model.node_head.weight.detach().clone()
    bias_before = client.model.node_head.bias.detach().clone()

    client.update_stateful(
        context,
        server_state=coordinator.global_state,
        strategy=strategy,
    )

    inactive = ~mask.cpu()
    assert torch.equal(context.valid_class_mask, mask)
    assert not torch.equal(
        client.model.node_head.weight[inactive], weight_before[inactive]
    )
    assert not torch.equal(
        client.model.node_head.bias[inactive], bias_before[inactive]
    )
