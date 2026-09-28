from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from gecko.engine.accounting import ResourceLedger
from gecko.engine.checkpoint import CheckpointIdentity
from gecko.engine.checkpoint import CheckpointIdentityError
from gecko.engine.checkpoint import CheckpointIntegrityError
from gecko.engine.checkpoint import canonical_json_digest
from gecko.engine.checkpoint import load_checkpoint
from gecko.engine.checkpoint import save_checkpoint
from gecko.engine import checkpoint as checkpoint_module


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        stream_id="uefa-v1-test-stream",
        stream_hash="1" * 64,
        scientific_fingerprint="2" * 64,
        source_sha="3" * 40,
        stream_content_digest="7" * 64,
        source_tree_digest="6" * 64,
        run_config_digest="4" * 64,
        method_config_digest="5" * 64,
        strategy="scaffold",
        method="Bare",
        model="begin_gcn",
        model_seed=0,
    )


def test_resource_ledger_preserves_legacy_model_wire_total_and_split_totals():
    ledger = ResourceLedger(
        training_model_uplink_bytes=11,
        training_model_downlink_bytes=13,
        training_auxiliary_uplink_bytes=17,
        training_auxiliary_downlink_bytes=19,
        evaluation_sync_bytes=23,
        artifact_distribution_bytes=29,
        replay_bytes=31,
        method_artifact_bytes=37,
        client_persistent_bytes=41,
        server_persistent_bytes=43,
    )

    assert ledger.communication_payload_bytes == 24
    assert ledger.training_wire_bytes == 60
    assert ledger.evaluation_wire_bytes == 23
    assert ledger.total_wire_bytes == 112
    assert ledger.persistent_state_bytes == 152
    rendered = ledger.to_dict()
    assert rendered["communication_payload_bytes"] == 24
    assert ResourceLedger.from_dict(rendered) == ledger


def test_resource_ledger_rejects_invalid_counts_and_derived_total_tampering():
    with pytest.raises(ValueError, match="cannot be negative"):
        ResourceLedger(training_model_uplink_bytes=-1)
    with pytest.raises(TypeError, match="integer"):
        ResourceLedger(training_model_uplink_bytes=True)
    ledger = ResourceLedger(training_model_uplink_bytes=3)
    with pytest.raises(KeyError, match="Unknown resource"):
        ledger.add(unknown_bytes=1)
    rendered = ledger.to_dict()
    rendered["communication_payload_bytes"] = 99
    with pytest.raises(ValueError, match="derived total mismatch"):
        ResourceLedger.from_dict(rendered)


def test_checkpoint_round_trip_is_weights_only_canonical_and_exactly_sized(
    tmp_path, monkeypatch
):
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = {
        "server": {"shared": shared, "control": torch.ones(2)},
        "clients": {
            0: {
                "model": shared,
                "seen_tasks": (0, 2),
                "seen_class_mask": torch.tensor([True, False, True]),
            }
        },
        "rng": {"torch_cpu": torch.get_rng_state()},
        "metadata": [None, True, 7, 1.25, "safe"],
    }
    written = save_checkpoint(
        tmp_path,
        "stage-000-round-001",
        identity=_identity(),
        cursor={"next_stage": 0, "next_round": 2, "global_step": 2},
        state=state,
    )
    manifest_payload = written.manifest_path.read_bytes()
    assert manifest_payload.endswith(b"\n")
    assert b": " not in manifest_payload
    assert written.manifest_bytes == written.manifest_path.stat().st_size
    assert written.weights_bytes == written.weights_path.stat().st_size
    assert written.total_bytes == written.manifest_bytes + written.weights_bytes
    assert not list(tmp_path.glob(".*.tmp"))

    original = checkpoint_module.torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", recording_load)
    loaded = load_checkpoint(written.manifest_path, expected_identity=_identity())
    assert calls == [True]
    assert loaded.cursor == {"next_stage": 0, "next_round": 2, "global_step": 2}
    assert torch.equal(loaded.state["server"]["shared"], shared)
    assert loaded.state["server"]["shared"] is loaded.state["clients"][0]["model"]
    assert loaded.state["clients"][0]["seen_tasks"] == (0, 2)
    assert loaded.resume_validation.identity_matched is True
    assert loaded.resume_validation.benchmark_eligible is True
    assert loaded.total_bytes == written.total_bytes


def test_checkpoint_identity_mismatch_fails_closed_and_override_is_recorded(tmp_path):
    written = save_checkpoint(
        tmp_path,
        "round-000",
        identity=_identity(),
        cursor={"next_stage": 0, "next_round": 1},
        state={"weight": torch.ones(1)},
    )
    expected = replace(_identity(), method_config_digest="a" * 64)
    with pytest.raises(CheckpointIdentityError, match="method_config_digest"):
        load_checkpoint(written.manifest_path, expected_identity=expected)
    with pytest.raises(ValueError, match="non-empty reason"):
        load_checkpoint(
            written.manifest_path,
            expected_identity=expected,
            diagnostic_override_reason="   ",
        )

    loaded = load_checkpoint(
        written.manifest_path,
        expected_identity=expected,
        diagnostic_override_reason="diagnose configuration migration only",
    )
    record = loaded.resume_validation.to_dict()
    assert record["identity_matched"] is False
    assert record["diagnostic_override_used"] is True
    assert record["benchmark_eligible"] is False
    assert (
        record["diagnostic_override_reason"] == "diagnose configuration migration only"
    )
    assert record["mismatch_fields"] == {
        "method_config_digest": {
            "expected": "a" * 64,
            "checkpoint": "5" * 64,
        }
    }


@pytest.mark.parametrize(
    "unsafe", [{"bad": object()}, {"bad": {1, 2}}, {"bad": float("nan")}]
)
def test_checkpoint_writer_rejects_non_safe_nested_state(tmp_path, unsafe):
    with pytest.raises(CheckpointIntegrityError):
        save_checkpoint(
            tmp_path,
            "unsafe",
            identity=_identity(),
            cursor={"next_stage": 0},
            state=unsafe,
        )
    assert not (tmp_path / "unsafe.manifest.json").exists()


def test_checkpoint_detects_tensor_tampering_before_loading(tmp_path):
    written = save_checkpoint(
        tmp_path,
        "tampered",
        identity=_identity(),
        cursor={"next_stage": 1},
        state={"weight": torch.arange(4)},
    )
    payload = bytearray(written.weights_path.read_bytes())
    payload[-1] ^= 1
    written.weights_path.write_bytes(payload)
    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_checkpoint(written.manifest_path, expected_identity=_identity())


def test_checkpoint_rejects_manifest_and_tensor_symlinks(tmp_path):
    first = save_checkpoint(
        tmp_path / "manifest",
        "safe",
        identity=_identity(),
        cursor={"next_stage": 0},
        state={"weight": torch.ones(1)},
    )
    real_manifest = first.manifest_path.with_name("real-manifest.json")
    first.manifest_path.rename(real_manifest)
    first.manifest_path.symlink_to(real_manifest)
    with pytest.raises(CheckpointIntegrityError, match="symlink"):
        load_checkpoint(first.manifest_path, expected_identity=_identity())

    second = save_checkpoint(
        tmp_path / "weights",
        "safe",
        identity=_identity(),
        cursor={"next_stage": 0},
        state={"weight": torch.ones(1)},
    )
    real_weights = second.weights_path.with_name("real-weights.pt")
    second.weights_path.rename(real_weights)
    second.weights_path.symlink_to(real_weights)
    with pytest.raises(CheckpointIntegrityError, match="symlink"):
        load_checkpoint(second.manifest_path, expected_identity=_identity())


def test_manifest_is_the_last_atomic_commit_point(tmp_path, monkeypatch):
    def interrupted_manifest_publish(path, payload):
        raise RuntimeError("simulated manifest interruption")

    monkeypatch.setattr(
        checkpoint_module, "_atomic_publish_bytes", interrupted_manifest_publish
    )
    with pytest.raises(RuntimeError, match="simulated manifest interruption"):
        save_checkpoint(
            tmp_path,
            "interrupted",
            identity=_identity(),
            cursor={"next_stage": 0},
            state={"weight": torch.ones(1)},
        )
    assert not (tmp_path / "interrupted.manifest.json").exists()
    assert list(tmp_path.glob("weights-*.pt"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_canonical_config_digest_is_key_order_invariant_and_rejects_nonfinite():
    assert canonical_json_digest({"b": 2, "a": [1, True]}) == canonical_json_digest(
        {"a": [1, True], "b": 2}
    )
    with pytest.raises(CheckpointIntegrityError, match="canonical JSON-safe"):
        canonical_json_digest({"bad": float("inf")})


def test_checkpoint_manifest_schema_is_strict(tmp_path):
    written = save_checkpoint(
        tmp_path,
        "schema",
        identity=_identity(),
        cursor={"next_stage": 0},
        state={"weight": torch.ones(1)},
    )
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    written.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="Unsupported checkpoint schema"):
        load_checkpoint(written.manifest_path, expected_identity=_identity())
