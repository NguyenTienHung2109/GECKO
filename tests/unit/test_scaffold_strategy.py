from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gecko.engine.accounting import ResourceLedger
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import ParameterEntry
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import StatefulStrategyProtocol
from gecko.algorithms.federated.scaffold import ScaffoldStrategy
from gecko.algorithms.federated.scaffold import scaffold_scalar_sgd_diagnostic_round
from gecko.types import LocalUpdateResult


class _TwoParameterModel(nn.Module):
    def __init__(self, shared: float = 1.0, local: float = 9.0) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor([shared]))
        self.local = nn.Parameter(torch.tensor([local]))


class _SharedOnlyPolicy:
    @staticmethod
    def extract(model: _TwoParameterModel):
        return {"shared": model.shared.detach().cpu().clone()}


class _FakeClient:
    def __init__(self, client_id: int, option_i_control: float) -> None:
        self.client_id = client_id
        self.model = _TwoParameterModel()
        self.parameter_policy = _SharedOnlyPolicy()
        self.state = SimpleNamespace(strategy_state={})
        self.received = []
        self.algorithm = SimpleNamespace(
            on_broadcast=lambda context, payload: self.received.append(payload.reason)
        )
        self.option_i_control = option_i_control
        self.control_calls = 0

    def load_shared_state(self, state):
        with torch.no_grad():
            self.model.shared.copy_(state["shared"])

    def control_gradient_at_shared(self, context, shared_state, shared_keys):
        assert tuple(shared_keys) == ("shared",)
        assert shared_state["shared"].shape == self.model.shared.shape
        self.control_calls += 1
        # The capability is required to leave the post-local model untouched.
        return {"shared": torch.tensor([self.option_i_control])}


def _manifest() -> ParameterManifest:
    return ParameterManifest(
        (
            ParameterEntry(
                name="shared",
                kind="shared_trainable",
                shape=(1,),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
            ParameterEntry(
                name="local",
                kind="local_trainable",
                shape=(1,),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=False,
            ),
        )
    )


def _method_context(client_id: int, *, stage: int = 0, round_index: int = 0):
    return SimpleNamespace(
        client_id=client_id,
        global_task_id=3,
        stage_index=stage,
        round_index=round_index,
    )


def _local_result(client: _FakeClient, *, value: float, weight: int):
    with torch.no_grad():
        client.model.shared.fill_(value)
    return LocalUpdateResult(
        client_id=client.client_id,
        global_task_id=3,
        shared_state={"shared": torch.tensor([value])},
        weight=weight,
        training_loss=value,
        communication_bytes=4,
        shareable_keys=("shared",),
    )


def _initialize(
    *, correction_enabled: bool = True, control_updates_enabled: bool = True
) -> ScaffoldStrategy:
    strategy = ScaffoldStrategy(
        correction_enabled=correction_enabled,
        control_updates_enabled=control_updates_enabled,
    )
    strategy.initialize({"shared": torch.tensor([1.0])}, _manifest(), (0, 1))
    assert isinstance(strategy, StatefulStrategyProtocol)
    return strategy


def _run_two_client_round(
    strategy: ScaffoldStrategy,
    clients: dict[int, _FakeClient],
    *,
    participant_order=(0, 1),
    upload_order=(0, 1),
):
    context = RoundContext(0, 0, participant_order, {0: 3, 1: 3})
    strategy.prepare_round(context, {0: 1, 1: 3})
    payloads = {}
    uploads = {}
    for client_id in participant_order:
        payload = strategy.prepare_payload(context, client_id, "training")
        payloads[client_id] = payload
        strategy.client_receive(
            clients[client_id], payload, _method_context(client_id)
        )
        uploads[client_id] = strategy.finalize_upload(
            clients[client_id],
            _local_result(
                clients[client_id], value={0: 2.0, 1: 4.0}[client_id], weight={0: 1, 1: 3}[client_id]
            ),
            _method_context(client_id),
        )
    result = strategy.aggregate(
        context, tuple(uploads[client_id] for client_id in upload_order)
    )
    return payloads, uploads, result


def test_scaffold_option_i_weighted_adaptation_and_exact_split_bytes():
    strategy = _initialize()
    clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    payloads, uploads, result = _run_two_client_round(strategy, clients)

    # Active current-query weights are 1:3 for model aggregation.
    assert result.shared_state["shared"].item() == pytest.approx(3.5)
    # The same all-client weights are used for the last-known control identity.
    assert strategy.server_control["shared"].item() == pytest.approx(5.0)
    assert strategy.client_control(0)["shared"].item() == 2.0
    assert strategy.client_control(1)["shared"].item() == 6.0
    assert clients[0].control_calls == clients[1].control_calls == 1

    ledger = ResourceLedger()
    for payload in payloads.values():
        assert payload.resources.training_model_downlink_bytes == 4
        assert payload.resources.training_auxiliary_downlink_bytes == 4
        ledger.merge(payload.resources)
    ledger.merge(result.resources)
    assert result.resources.training_model_uplink_bytes == 8
    assert result.resources.training_auxiliary_uplink_bytes == 8
    assert ledger.communication_payload_bytes == 16
    assert ledger.training_wire_bytes == 32
    assert ledger.training_wire_bytes == 2 * ledger.communication_payload_bytes
    assert all(upload.model_state_semantics == "delta" for upload in uploads.values())


def test_scaffold_correction_is_shared_only_and_masks_can_run_after_it():
    strategy = _initialize()
    clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    _run_two_client_round(strategy, clients)

    context = RoundContext(0, 1, (0,), {0: 3})
    strategy.prepare_round(context, {0: 1, 1: 3})
    payload = strategy.prepare_payload(context, 0, "training")
    method_context = _method_context(0, round_index=1)
    strategy.client_receive(clients[0], payload, method_context)
    clients[0].model.shared.grad = torch.tensor([7.0])
    clients[0].model.local.grad = torch.tensor([11.0])
    strategy.transform_gradients(
        clients[0].model, method_context, ("shared",)
    )

    # g - c_i + c = 7 - 2 + 5 = 10; the local parameter is untouched.
    assert clients[0].model.shared.grad.item() == pytest.approx(10.0)
    assert clients[0].model.local.grad.item() == 11.0
    # A later Task/Class hook can mask the complete corrected gradient.
    clients[0].model.shared.grad.zero_()
    assert clients[0].model.shared.grad.item() == 0.0


def test_scaffold_partial_participation_preserves_inactive_control():
    strategy = _initialize()
    clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    _run_two_client_round(strategy, clients)
    inactive_before = strategy.client_control(1)

    clients[0].option_i_control = 4.0
    context = RoundContext(0, 1, (0,), {0: 3})
    strategy.prepare_round(context, {0: 1, 1: 3})
    payload = strategy.prepare_payload(context, 0, "training")
    method_context = _method_context(0, round_index=1)
    strategy.client_receive(clients[0], payload, method_context)
    upload = strategy.finalize_upload(
        clients[0],
        _local_result(clients[0], value=4.5, weight=1),
        method_context,
    )
    result = strategy.aggregate(context, (upload,))

    assert result.shared_state["shared"].item() == pytest.approx(4.5)
    assert strategy.client_control(1)["shared"].item() == inactive_before[
        "shared"
    ].item()
    assert strategy.server_control["shared"].item() == pytest.approx(5.5)
    diagnostics = strategy.diagnostics()
    assert diagnostics["client_control_update_counts"] == {"0": 2, "1": 1}


def test_scaffold_participant_and_upload_order_are_numerically_invariant():
    first = _initialize()
    second = _initialize()
    first_clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    second_clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    _, _, first_result = _run_two_client_round(
        first,
        first_clients,
        participant_order=(0, 1),
        upload_order=(1, 0),
    )
    _, _, second_result = _run_two_client_round(
        second,
        second_clients,
        participant_order=(1, 0),
        upload_order=(0, 1),
    )
    assert torch.equal(
        first_result.shared_state["shared"], second_result.shared_state["shared"]
    )
    assert torch.equal(
        first.server_control["shared"], second.server_control["shared"]
    )


def test_scaffold_rejects_invalid_upload_atomically():
    strategy = _initialize()
    clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    context = RoundContext(0, 0, (0, 1), {0: 3, 1: 3})
    strategy.prepare_round(context, {0: 1, 1: 3})
    uploads = {}
    for client_id in (0, 1):
        method_context = _method_context(client_id)
        payload = strategy.prepare_payload(context, client_id, "training")
        strategy.client_receive(clients[client_id], payload, method_context)
        uploads[client_id] = strategy.finalize_upload(
            clients[client_id],
            _local_result(
                clients[client_id], value={0: 2.0, 1: 4.0}[client_id], weight={0: 1, 1: 3}[client_id]
            ),
            method_context,
        )
    bad = ClientUpload(
        client_id=0,
        global_task_id=3,
        weight=1,
        training_loss=2.0,
        model_state={"shared": torch.tensor([99.0])},
        model_state_semantics="delta",
        auxiliary_state=uploads[0].auxiliary_state,
        resources=uploads[0].resources,
    )
    before_model = strategy.shared_state
    before_server_control = strategy.server_control
    before_private = {
        client_id: clients[client_id].state.strategy_state[
            "scaffold_uefa_adam_v1"
        ]["control"]["shared"].clone()
        for client_id in clients
    }
    with pytest.raises(ValueError, match="staged"):
        strategy.aggregate(context, (bad, uploads[1]))
    assert torch.equal(strategy.shared_state["shared"], before_model["shared"])
    assert torch.equal(
        strategy.server_control["shared"], before_server_control["shared"]
    )
    for client_id in clients:
        assert torch.equal(
            clients[client_id].state.strategy_state["scaffold_uefa_adam_v1"][
                "control"
            ]["shared"],
            before_private[client_id],
        )
    # The valid staged uploads can still commit after the rejected attempt.
    strategy.aggregate(context, (uploads[1], uploads[0]))


def test_scaffold_n1_population_control_identity_and_zero_next_correction():
    strategy = ScaffoldStrategy()
    strategy.initialize({"shared": torch.tensor([1.0])}, _manifest(), (0,))
    client = _FakeClient(0, 3.0)
    context = RoundContext(0, 0, (0,), {0: 3})
    strategy.prepare_round(context, {0: 2})
    payload = strategy.prepare_payload(context, 0, "training")
    method_context = _method_context(0)
    strategy.client_receive(client, payload, method_context)
    upload = strategy.finalize_upload(
        client, _local_result(client, value=2.0, weight=2), method_context
    )
    strategy.aggregate(context, (upload,))
    assert strategy.server_control["shared"].item() == 3.0
    assert strategy.client_control(0)["shared"].item() == 3.0

    second = RoundContext(0, 1, (0,), {0: 3})
    strategy.prepare_round(second, {0: 2})
    payload = strategy.prepare_payload(second, 0, "training")
    second_context = _method_context(0, round_index=1)
    strategy.client_receive(client, payload, second_context)
    client.model.shared.grad = torch.tensor([5.0])
    strategy.transform_gradients(client.model, second_context, ("shared",))
    assert client.model.shared.grad.item() == 5.0


def test_scaffold_exact_degeneration_matches_weighted_fedavg_and_wire_shape():
    strategy = _initialize(
        correction_enabled=False, control_updates_enabled=False
    )
    assert strategy.is_exact_fedavg_degeneration
    assert strategy.legacy_delegate_name == "fedavg"
    clients = {0: _FakeClient(0, 100.0), 1: _FakeClient(1, -100.0)}
    payloads, uploads, result = _run_two_client_round(strategy, clients)

    assert torch.equal(result.shared_state["shared"], torch.tensor([3.5]))
    assert all(not payload.auxiliary_state for payload in payloads.values())
    assert all(not upload.auxiliary_state for upload in uploads.values())
    assert all(
        upload.model_state_semantics == "full_shared_state"
        for upload in uploads.values()
    )
    assert result.resources.training_auxiliary_uplink_bytes == 0
    assert clients[0].control_calls == clients[1].control_calls == 0
    assert strategy.server_control["shared"].item() == 0.0


def test_scaffold_state_roundtrip_is_owned_strict_and_resumable():
    strategy = _initialize()
    clients = {0: _FakeClient(0, 2.0), 1: _FakeClient(1, 6.0)}
    _run_two_client_round(strategy, clients)
    saved = strategy.state_dict()
    saved["shared_state"]["shared"].fill_(99.0)
    assert strategy.shared_state["shared"].item() == pytest.approx(3.5)

    checkpoint = strategy.state_dict()
    restored = _initialize()
    restored.load_state_dict(checkpoint)
    checkpoint["server_control"]["shared"].zero_()
    assert restored.shared_state["shared"].item() == pytest.approx(3.5)
    assert restored.server_control["shared"].item() == pytest.approx(5.0)
    assert restored.client_control(1)["shared"].item() == 6.0
    assert restored.diagnostics()["completed_rounds"] == 1

    with pytest.raises(ValueError, match="fields"):
        restored.load_state_dict({"name": "scaffold"})
    mismatched = restored.state_dict()
    mismatched["variant"] = "not-scaffold"
    with pytest.raises(ValueError, match="configuration identity"):
        restored.load_state_dict(mismatched)
    broken_identity = restored.state_dict()
    broken_identity["server_control"]["shared"].add_(1.0)
    with pytest.raises(ValueError, match="weighted last-known"):
        restored.load_state_dict(broken_identity)


def test_scaffold_evaluation_uses_shared_post_broadcast_model_only():
    strategy = _initialize()
    selection = strategy.select_evaluation(0, 4)
    assert selection.source == "post_broadcast"
    assert selection.count_evaluation_sync
    assert selection.model_state["shared"].item() == 1.0

    context = RoundContext(4, 2, (0,), {0: 3})
    payload = strategy.prepare_payload(context, 0, "evaluation")
    assert payload.resources.evaluation_sync_bytes == 4
    assert not payload.auxiliary_state


def test_scalar_sgd_oracle_option_i_matches_algorithm_one_closed_form():
    result = scaffold_scalar_sgd_diagnostic_round(
        server_model=10.0,
        server_control=2.0,
        client_controls={0: 1.0, 1: 3.0},
        local_gradient_steps={0: (4.0, 2.0), 1: (6.0, 8.0)},
        local_learning_rate=0.5,
        control_option="option_i",
        option_i_server_gradients={0: 5.0, 1: 7.0},
    )
    assert result.diagnostic_only
    assert dict(result.local_models) == {0: 6.0, 1: 4.0}
    assert result.server_model == pytest.approx(5.0)
    assert result.server_control == pytest.approx(6.0)
    assert dict(result.client_controls) == {0: 5.0, 1: 7.0}


def test_scalar_sgd_oracle_option_ii_and_partial_participation_closed_forms():
    full = scaffold_scalar_sgd_diagnostic_round(
        server_model=10.0,
        server_control=2.0,
        client_controls={0: 1.0, 1: 3.0},
        local_gradient_steps={0: (4.0, 2.0), 1: (6.0, 8.0)},
        local_learning_rate=0.5,
        control_option="option_ii",
    )
    assert full.server_model == pytest.approx(5.0)
    assert full.server_control == pytest.approx(5.0)
    assert dict(full.client_controls) == {0: 3.0, 1: 7.0}

    partial = scaffold_scalar_sgd_diagnostic_round(
        server_model=10.0,
        server_control=1.0,
        client_controls={0: 0.0, 1: 2.0},
        local_gradient_steps={0: (3.0,)},
        local_learning_rate=0.5,
        control_option="option_ii",
    )
    assert partial.server_model == pytest.approx(8.0)
    assert partial.server_control == pytest.approx(2.5)
    assert partial.client_control(0) == pytest.approx(3.0)
    assert partial.client_control(1) == pytest.approx(2.0)


def test_scalar_sgd_oracle_one_local_step_k1_closed_form():
    result = scaffold_scalar_sgd_diagnostic_round(
        server_model=10.0,
        server_control=2.0,
        client_controls={0: 1.0, 1: 3.0},
        local_gradient_steps={0: (4.0,)},
        local_learning_rate=0.5,
        control_option="option_i",
        option_i_server_gradients={0: 5.0},
    )

    # K=1: y = 10 - 0.5 * (4 - 1 + 2) = 7.5.
    assert dict(result.local_models) == {0: pytest.approx(7.5)}
    assert result.server_model == pytest.approx(7.5)
    # c <- 2 + (1 / N) * (5 - 1), with N=2.
    assert result.server_control == pytest.approx(4.0)
    assert result.client_control(0) == pytest.approx(5.0)
    assert result.client_control(1) == pytest.approx(3.0)
