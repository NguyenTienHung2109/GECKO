from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gecko.engine import FederatedCoordinator
from gecko.models.backbones import GECKOGraphModel
from gecko.data.partitioning.assignment import _support_tensor
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder
from gecko.data.streams import audit_stream
from gecko.data.streams import derive_participation_variant
from gecko.data.streams import gc_objects
from gecko.data.streams import load_stream
from gecko.data.streams import migrate_stream_to_v2
from gecko.data.streams import save_stream
from gecko.validation import ArtifactIntegrityError
from gecko.validation import PartitionInfeasibleError

from tests.helpers import CASES
from tests.helpers import assert_tensor_map_equal
from tests.helpers import make_config
from tests.helpers import make_stream


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_01_ownership_is_complete(problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    owner = stream.partition.node_owner
    assert owner.shape == (stream.scenario.node_features.shape[0],)
    assert bool(((owner >= 0) & (owner < stream.config.partition.num_clients)).all())


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_02_ownership_is_disjoint(problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    owned = torch.cat(
        [graph.local_to_global for graph in stream.partition.client_graphs.values()]
    )
    assert torch.unique(owned).numel() == owned.numel()


def test_03_ownership_is_persistent_across_tasks_and_execution():
    stream = make_stream("NC", "task", 2)
    before = stream.partition.node_owner.clone()
    FederatedCoordinator(stream, "fedavg", "Bare").run()
    assert torch.equal(before, stream.partition.node_owner)


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_04_each_owned_node_occurs_in_exactly_one_local_graph(problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    occurrences = torch.zeros(stream.scenario.node_features.shape[0], dtype=torch.long)
    for graph in stream.partition.client_graphs.values():
        occurrences[graph.local_to_global] += 1
    assert torch.equal(occurrences, torch.ones_like(occurrences))


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_05_strict_local_graphs_have_no_cross_client_edges(problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    for client_id, graph in stream.partition.client_graphs.items():
        global_edges = graph.local_to_global[graph.edge_index]
        assert bool((stream.partition.node_owner[global_edges] == client_id).all())


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_06_global_local_mappings_round_trip(problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    for graph in stream.partition.client_graphs.values():
        for local_id, global_id in enumerate(graph.local_to_global.tolist()):
            assert graph.global_to_local[global_id] == local_id


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_07_same_seed_regenerates_equivalent_logical_artifacts(problem, incremental, num_tasks):
    config = make_config(problem, incremental, num_tasks)
    first_spec = build_synthetic_spec(problem, incremental, num_tasks=num_tasks, num_clients=2)
    second_spec = build_synthetic_spec(problem, incremental, num_tasks=num_tasks, num_clients=2)
    first = StreamBuilder(config).build(first_spec)
    second = StreamBuilder(config).build(second_spec)
    assert torch.equal(first.scenario.edge_index, second.scenario.edge_index)
    assert torch.equal(first.scenario.labels, second.scenario.labels)
    assert torch.equal(first.partition.node_owner, second.partition.node_owner)
    assert first.orders == second.orders
    assert first.participation == second.participation


def test_08_different_method_does_not_alter_stream_artifacts():
    stream = make_stream("NC", "task", 2)
    before = (
        stream.scenario.labels.clone(),
        stream.partition.node_owner.clone(),
        stream.participation.trace.copy(),
    )
    FederatedCoordinator(stream, "fedavg", "Bare").run()
    FederatedCoordinator(stream, "fedprox", "EWC").run()
    assert torch.equal(before[0], stream.scenario.labels)
    assert torch.equal(before[1], stream.partition.node_owner)
    assert before[2] == stream.participation.trace


def test_09_different_model_does_not_alter_stream_artifacts():
    stream = make_stream("LC", "class", 2)
    before = stream.scenario.query_endpoints.clone()
    factory = lambda: GECKOGraphModel(8, 4, hidden_size=5, num_layers=1, problem_type="LC")
    FederatedCoordinator(stream, "fedavg", "Bare", model_factory=factory).run()
    assert torch.equal(before, stream.scenario.query_endpoints)


def test_10_order_profile_changes_only_order_and_stream_identity():
    synchronized = make_stream("LP", "domain", 2, order_profile="synchronized")
    hard = make_stream("LP", "domain", 2, order_profile="hard")
    assert torch.equal(synchronized.scenario.labels, hard.scenario.labels)
    assert torch.equal(synchronized.scenario.query_endpoints, hard.scenario.query_endpoints)
    assert torch.equal(synchronized.partition.node_owner, hard.partition.node_owner)
    assert synchronized.orders.client_orders != hard.orders.client_orders
    assert synchronized.stream_id != hard.stream_id


def test_11_spatial_profile_does_not_change_global_task_definitions():
    easy = make_stream("LC", "domain", 4, spatial_profile="easy")
    hard = make_stream("LC", "domain", 4, spatial_profile="hard")
    assert torch.equal(easy.scenario.query_task_ids, hard.scenario.query_task_ids)
    assert_tensor_map_equal(
        easy.scenario.query_ids_by_task_split,
        hard.scenario.query_ids_by_task_split,
    )


def test_20_dense_support_failure_has_exact_client_task_split_report():
    config = make_config("NC", "task", 2, minimum_support=10_000)
    scenario = build_synthetic_spec("NC", "task", num_tasks=2, num_clients=2)
    with pytest.raises(PartitionInfeasibleError) as raised:
        StreamBuilder(config).build(scenario)
    message = str(raised.value)
    assert "client=" in message
    assert "task=" in message
    assert "split=" in message
    assert "required=10000" in message


def test_21_lp_support_counts_positives_not_abundant_negatives():
    spec = build_synthetic_spec("LP", "domain", num_tasks=2, num_clients=2)
    positive_id = int(torch.nonzero(spec.labels == 1, as_tuple=True)[0][0])
    spec.labels.zero_()
    spec.labels[positive_id] = 1
    endpoints = spec.query_endpoints[positive_id]
    owner = torch.zeros(spec.node_features.shape[0], dtype=torch.long)
    owner[endpoints[1]] = 1

    support = _support_tensor(spec, owner, num_clients=2)

    task = int(spec.query_task_ids[positive_id])
    split_names = ("train", "val", "test")
    split = next(
        index
        for index, name in enumerate(split_names)
        if positive_id in spec.query_ids_by_task_split[task][name].tolist()
    )
    assert support[:, task, split].sum() == 0


def test_22_lp_candidate_coverage_counts_all_internal_queries():
    stream = make_stream("LP", "domain", 2)
    endpoints = stream.scenario.query_endpoints
    owner = stream.partition.node_owner
    expected = float(
        (owner[endpoints[:, 0]] == owner[endpoints[:, 1]]).float().mean()
    )

    diagnostics = stream.partition.diagnostics
    assert diagnostics["support_semantics"] == "positive_queries"
    assert diagnostics["lp_internal_candidate_coverage"] == pytest.approx(expected)


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_23_internal_query_coverage_is_a_probability(problem, incremental, num_tasks):
    coverage = make_stream(problem, incremental, num_tasks).partition.diagnostics[
        "internal_query_coverage"
    ]
    assert 0.0 <= coverage <= 1.0


@pytest.mark.parametrize("problem,incremental,num_tasks", CASES)
def test_30_artifact_save_load_preserves_semantics(tmp_path, problem, incremental, num_tasks):
    stream = make_stream(problem, incremental, num_tasks)
    path = save_stream(stream, tmp_path, repository_root=tmp_path)
    loaded = load_stream(path)
    assert loaded.stream_id == stream.stream_id
    assert torch.equal(loaded.scenario.labels, stream.scenario.labels)
    assert torch.equal(loaded.partition.node_owner, stream.partition.node_owner)
    assert loaded.orders.client_orders == stream.orders.client_orders
    for client, tasks in stream.shards.items():
        for task, shard in tasks.items():
            restored = loaded.shards[client][task]
            assert torch.equal(restored.train_queries, shard.train_queries)
            assert torch.equal(restored.train_labels, shard.train_labels)


def test_31_checksums_detect_mutated_and_missing_artifacts(tmp_path):
    path = save_stream(make_stream("NC", "task", 2), tmp_path, repository_root=tmp_path)
    orders = path / "orders.json"
    orders.write_text(orders.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        audit_stream(path)
    path = save_stream(make_stream("NC", "task", 2), tmp_path / "second", repository_root=tmp_path)
    (path / "tasks.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="actual.*None"):
        audit_stream(path)


def _schema_v2_stream(order_profile: str):
    config = replace(
        make_config("NC", "task", 2, order_profile=order_profile),
        benchmark_schema_version=2,
    )
    spec = build_synthetic_spec("NC", "task", num_tasks=2, num_clients=2)
    return StreamBuilder(config).build(spec)


def test_32_schema_v2_content_addressed_stream_loads_and_audits(tmp_path):
    stream = _schema_v2_stream("synchronized")
    path = save_stream(stream, tmp_path, repository_root=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_layout"] == "content_addressed_hardlink_v2"
    assert {"scenario", "partition", "queries", "evaluation", "order"} <= set(
        manifest["component_digests"]
    )
    for name, record in manifest["component_objects"].items():
        object_path = (tmp_path / "objects" / record["object_key"]).resolve()
        assert object_path.exists()
        assert os.path.samefile(path / name, object_path)
        assert object_path.stat().st_mode & 0o222 == 0
    loaded = load_stream(path)
    assert loaded.stream_id == stream.stream_id
    assert audit_stream(path)["artifact_layout"] == "content_addressed_hardlink_v2"


def test_33_order_only_change_reuses_non_order_components(tmp_path):
    synchronized = save_stream(
        _schema_v2_stream("synchronized"), tmp_path, repository_root=tmp_path
    )
    hard = save_stream(_schema_v2_stream("hard"), tmp_path, repository_root=tmp_path)
    first = json.loads((synchronized / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((hard / "manifest.json").read_text(encoding="utf-8"))
    for component in ("scenario", "partition", "queries", "evaluation"):
        assert first["component_digests"][component] == second["component_digests"][component]
    assert first["component_digests"]["order"] != second["component_digests"]["order"]


def test_33b_fast_order_derivation_reuses_immutable_graph_objects(tmp_path):
    from gecko.data.streams import derive_order_variant

    synchronized = save_stream(
        _schema_v2_stream("synchronized"), tmp_path, repository_root=tmp_path
    )
    hard = derive_order_variant(
        synchronized,
        "hard",
        root=tmp_path,
        repository_root=tmp_path,
    )
    assert audit_stream(hard)["valid"] is True
    first = json.loads((synchronized / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((hard / "manifest.json").read_text(encoding="utf-8"))
    for component in (
        "scenario",
        "partition",
        "queries",
        "evaluation",
        "participation",
    ):
        assert first["component_digests"][component] == second["component_digests"][component]
    assert first["component_digests"]["order"] != second["component_digests"]["order"]
    assert first["component_digests"]["provenance"] != second["component_digests"]["provenance"]
    assert first["scientific_fingerprint"] != second["scientific_fingerprint"]
    assert os.path.samefile(synchronized / "scenario.pt", hard / "scenario.pt")
    assert json.loads((hard / "orders.json").read_text(encoding="utf-8"))[
        "diagnostics"
    ]["normalized_pairwise_kendall_distance"] > 0


def test_33c_order_derivation_script_bootstraps_repository_imports():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "derive_gecko_order_variant.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_33d_batch_order_derivation_audits_source_once(monkeypatch, tmp_path):
    import gecko.data.streams.store as artifacts

    synchronized = save_stream(
        _schema_v2_stream("synchronized"), tmp_path, repository_root=tmp_path
    )
    calls = 0
    from gecko.data.streams import integrity
    original = integrity._audit_stream_unlocked

    def counting_audit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(integrity, "_audit_stream_unlocked", counting_audit)
    paths = artifacts.derive_order_variants(
        synchronized,
        ("mild", "hard"),
        root=tmp_path,
        repository_root=tmp_path,
    )
    assert calls == 1
    assert {path.name for path in paths} == {
        "task-order-legacy-mild", "task-order-unsynchronized"
    }
    assert all(audit_stream(path)["valid"] for path in paths)


def test_33e_participation_derivation_preserves_all_data_artifacts(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source = save_stream(
        StreamBuilder(
            replace(
                make_config(
                    "NC", "task", 2, order_profile="synchronized", rounds=2
                ),
                benchmark_schema_version=2,
            )
        ).build(build_synthetic_spec("NC", "task", num_tasks=2, num_clients=2)),
        source_root,
        repository_root=tmp_path,
    )
    source_manifest_before = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    source_checksums_before = json.loads(
        (source / "checksums.json").read_text(encoding="utf-8")
    )

    target = derive_participation_variant(
        source, 5, root=target_root, repository_root=tmp_path
    )
    assert derive_participation_variant(
        source, 5, root=target_root, repository_root=tmp_path
    ) == target
    assert audit_stream(target)["valid"] is True
    source_manifest_after = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    source_checksums_after = json.loads(
        (source / "checksums.json").read_text(encoding="utf-8")
    )
    target_manifest = json.loads(
        (target / "manifest.json").read_text(encoding="utf-8")
    )
    assert source_manifest_before == source_manifest_after
    assert source_checksums_before == source_checksums_after
    assert target_manifest["config_hash"] != source_manifest_before["config_hash"]
    assert target_manifest["stream_id"] != source_manifest_before["stream_id"]
    assert target_manifest["derived_from"]["package_digest"] == source_manifest_before[
        "package_digest"
    ]
    for component in ("scenario", "partition", "queries", "evaluation"):
        assert target_manifest["component_digests"][component] == source_manifest_before[
            "component_digests"
        ][component]
    for name, digest in source_manifest_before["artifact_checksums"].items():
        if name not in {"config.yaml", "orders.json", "participation.json"}:
            assert target_manifest["artifact_checksums"][name] == digest

    old_orders = json.loads((source / "orders.json").read_text(encoding="utf-8"))
    new_orders = json.loads((target / "orders.json").read_text(encoding="utf-8"))
    for key in (
        "canonical_order",
        "client_orders",
        "inverse_client_orders",
        "cohort_assignments",
        "block_size",
        "seed",
    ):
        assert new_orders[key] == old_orders[key]
    old_participation = json.loads(
        (source / "participation.json").read_text(encoding="utf-8")
    )
    new_participation = json.loads(
        (target / "participation.json").read_text(encoding="utf-8")
    )
    assert old_participation["rounds_per_stage"] == 2
    assert new_participation["rounds_per_stage"] == 5
    for stage, rounds in old_participation["trace"].items():
        for round_id, clients in rounds.items():
            assert new_participation["trace"][stage][round_id] == clients
    assert load_stream(target).config.training.local_epochs_per_round == 1


def test_34_schema_v2_object_tampering_is_detected(tmp_path):
    path = save_stream(_schema_v2_stream("synchronized"), tmp_path, repository_root=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["component_objects"]["orders.json"]
    object_path = (tmp_path / "objects" / record["object_key"]).resolve()
    assert object_path.stat().st_mode & 0o222 == 0
    object_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    object_path.write_text(object_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        audit_stream(path)


def test_35_schema_v2_concurrent_generation_is_race_safe(tmp_path):
    synchronized = _schema_v2_stream("synchronized")
    hard = _schema_v2_stream("hard")
    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(
            executor.map(
                lambda stream: save_stream(stream, tmp_path, repository_root=tmp_path),
                (synchronized, hard),
            )
        )
    for path in paths:
        assert audit_stream(path)["valid"] is True
    assert not list((tmp_path / "objects").rglob(".*.tmp"))


def test_36_schema_v2_survives_relocation_without_hardlinks(tmp_path):
    source = tmp_path / "source"
    path = save_stream(
        _schema_v2_stream("synchronized"), source, repository_root=tmp_path
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated, copy_function=shutil.copy2)
    relocated_stream = relocated / path.relative_to(source)
    assert audit_stream(relocated_stream)["valid"] is True
    assert load_stream(relocated_stream).stream_id == load_stream(path).stream_id


def test_37_gc_deletes_only_unreferenced_objects(tmp_path):
    path = save_stream(
        _schema_v2_stream("synchronized"), tmp_path, repository_root=tmp_path
    )
    orphan = tmp_path / "objects" / "scenario" / ("0" * 64) / "orphan.pt"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    orphan.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    dry_run = gc_objects(tmp_path)
    assert dry_run["unreferenced_objects"] == 1
    assert dry_run["deleted_objects"] == 0
    executed = gc_objects(tmp_path, dry_run=False)
    assert executed["deleted_objects"] == 1
    assert not orphan.exists()
    assert audit_stream(path)["valid"] is True


def test_38_schema_v2_cleans_stale_crash_temporary(tmp_path):
    stream = _schema_v2_stream("synchronized")
    path = save_stream(stream, tmp_path, repository_root=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    object_path = (
        tmp_path
        / "objects"
        / manifest["component_objects"]["scenario.pt"]["object_key"]
    ).resolve()
    temporary = object_path.parent / ".scenario.pt.crash.tmp"
    temporary.write_bytes(b"partial")
    os.utime(temporary, (0, 0))
    save_stream(stream, tmp_path, repository_root=tmp_path)
    assert not temporary.exists()
    assert audit_stream(path)["valid"] is True


def test_39_schema_v1_migration_preserves_stream_tensors(tmp_path):
    original = make_stream("NC", "task", 2)
    v1_path = save_stream(original, tmp_path / "v1", repository_root=tmp_path)
    migrated_path = migrate_stream_to_v2(
        v1_path, tmp_path / "v2", repository_root=tmp_path
    )
    migrated = load_stream(migrated_path)
    assert migrated.config.benchmark_schema_version == 2
    assert torch.equal(migrated.scenario.edge_index, original.scenario.edge_index)
    assert torch.equal(migrated.partition.node_owner, original.partition.node_owner)
    assert migrated.orders.client_orders == original.orders.client_orders
    assert audit_stream(migrated_path)["artifact_layout"] == "content_addressed_hardlink_v2"


def test_40_schema_v2_components_are_root_location_independent(tmp_path):
    stream = _schema_v2_stream("synchronized")
    first_path = save_stream(stream, tmp_path / "first", repository_root=tmp_path)
    second_path = save_stream(stream, tmp_path / "second", repository_root=tmp_path)
    first = json.loads((first_path / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((second_path / "manifest.json").read_text(encoding="utf-8"))
    assert first["component_digests"] == second["component_digests"]
    assert first["artifact_checksums"] == second["artifact_checksums"]
    assert first["scientific_fingerprint"] == second["scientific_fingerprint"]
