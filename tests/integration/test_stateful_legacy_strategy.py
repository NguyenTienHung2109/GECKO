from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gecko.engine.protocol import ParameterEntry
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import StatefulStrategyProtocol
from gecko.algorithms.federated.legacy import LegacyStrategyAdapter
from gecko.types import LocalUpdateResult


class FakeClient:
    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.model = nn.Linear(1, 1, bias=False)
        self.algorithm = SimpleNamespace(
            on_broadcast=lambda context, payload: setattr(
                self, "broadcast_reason", payload.reason
            )
        )

    def load_shared_state(self, state):
        with torch.no_grad():
            self.model.weight.copy_(state["weight"])


def _manifest() -> ParameterManifest:
    return ParameterManifest(
        (
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(1, 1),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )
    )


def _result(client_id: int, value: float, weight: int) -> LocalUpdateResult:
    return LocalUpdateResult(
        client_id,
        3,
        {"weight": torch.tensor([[value]])},
        weight,
        value,
        4,
        ("weight",),
    )


def test_legacy_strategy_adapter_satisfies_protocol_and_matches_weighted_fedavg():
    adapter = LegacyStrategyAdapter("fedavg")
    assert isinstance(adapter, StatefulStrategyProtocol)
    adapter.initialize({"weight": torch.tensor([[1.0]])}, _manifest(), (0, 1))
    context = RoundContext(
        stage_index=0,
        round_index=0,
        participant_ids=(1, 0),
        client_global_task_ids={0: 3, 1: 3},
    )

    clients = {client_id: FakeClient(client_id) for client_id in (0, 1)}
    payload = adapter.prepare_payload(context, 1, "training")
    assert payload is not None
    assert payload.resources.training_model_downlink_bytes == 4
    adapter.client_receive(clients[1], payload, SimpleNamespace())
    assert clients[1].model.weight.item() == 1.0
    assert clients[1].broadcast_reason == "training"

    uploads = (
        adapter.finalize_upload(clients[1], _result(1, 4.0, 3), None),
        adapter.finalize_upload(clients[0], _result(0, 2.0, 1), None),
    )
    aggregation = adapter.aggregate(context, uploads)
    assert aggregation.shared_state["weight"].item() == pytest.approx(3.5)
    assert aggregation.resources.training_model_uplink_bytes == 8
    assert adapter.shared_state["weight"].item() == pytest.approx(3.5)

    selection = adapter.select_evaluation(0, 0)
    assert selection.source == "post_broadcast"
    assert selection.count_evaluation_sync
    assert selection.model_state["weight"].item() == pytest.approx(3.5)


def test_local_only_adapter_never_broadcasts_or_counts_training_payload():
    adapter = LegacyStrategyAdapter("local_only")
    adapter.initialize({"weight": torch.tensor([[1.0]])}, _manifest(), (0,))
    context = RoundContext(0, 0, (0,), {0: 3})
    assert adapter.prepare_payload(context, 0, "training") is None
    client = FakeClient(0)
    upload = adapter.finalize_upload(client, _result(0, 7.0, 1), None)
    result = adapter.aggregate(context, (upload,))
    assert result.shared_state["weight"].item() == 1.0
    assert result.resources.communication_payload_bytes == 0
    assert adapter.select_evaluation(0, 0).source == "post_local"


def test_legacy_adapter_state_round_trip_is_owned_and_fail_closed():
    adapter = LegacyStrategyAdapter("fedprox")
    adapter.initialize({"weight": torch.tensor([[2.0]])}, _manifest(), (0,))
    state = adapter.state_dict()
    state["shared_state"]["weight"].fill_(9)
    assert adapter.shared_state["weight"].item() == 2.0

    restored = adapter.state_dict()
    restored["shared_state"]["weight"].fill_(5)
    adapter.load_state_dict(restored)
    restored["shared_state"]["weight"].zero_()
    assert adapter.shared_state["weight"].item() == 5.0
    with pytest.raises(ValueError, match="fields"):
        adapter.load_state_dict({"name": "fedprox"})
