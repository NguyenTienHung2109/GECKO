from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import stat
import subprocess
import tarfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.cli.main import main
from gecko.data.streams import StreamBuilder
from gecko.data.streams import audit_stream
from gecko.data.streams import gc_objects
from gecko.data.streams import load_stream
from gecko.data.streams import load_trusted_legacy_stream
from gecko.data.streams import save_stream
from gecko.data.streams import store as artifact_module
from gecko.data.streams.manifest import repository_provenance
from gecko.validation import ArtifactIntegrityError

from tests.helpers import make_config


def _security_stream(profile: str = "synchronized"):
    config = replace(
        make_config("NC", "task", 2, order_profile=profile),
        benchmark_schema_version=2,
    )
    scenario = build_synthetic_spec("NC", "task", num_tasks=2, num_clients=2)
    return StreamBuilder(config).build(scenario)


def _process_save(root: str, profile: str, queue) -> None:
    try:
        path = save_stream(_security_stream(profile), root, repository_root=root)
        queue.put(("ok", str(path)))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def _process_load(path: str, queue) -> None:
    try:
        queue.put(("ok", load_stream(path).stream_id))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def _process_gc(root: str, queue) -> None:
    try:
        queue.put(("ok", gc_objects(root, dry_run=False)))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def _process_die_while_holding_store_lock(root: str, ready) -> None:
    storage_root = Path(root).resolve()
    artifact_module._initialize_store(storage_root)
    with artifact_module._store_lock(storage_root, exclusive=True):
        temporary = storage_root / "objects" / "scenario" / ".killed.tmp"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"partial")
        ready.set()
        time.sleep(60)


def _rewrite_manifest(path: Path, mutate) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    checksums_path = path / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["manifest.json"] = artifact_module._checksum(manifest_path)
    checksums_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8"
    )


def test_public_loader_uses_weights_only_for_every_torch_payload(tmp_path, monkeypatch):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    original = artifact_module.torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact_module.torch, "load", recording_load)
    assert load_stream(path).stream_id
    assert calls and all(value is True for value in calls)


@pytest.mark.skipif(shutil.which("git") is None, reason="optional Git metadata diagnostic")
def test_repository_provenance_distinguishes_clean_and_dirty_worktrees(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "uefa@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "UEFA Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    clean = repository_provenance(repository, collect_git=True)
    assert clean["worktree_clean"] is True
    assert clean["dirty_status_sha256"] is None
    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = repository_provenance(repository, collect_git=True)
    assert dirty["commit_sha"] == clean["commit_sha"]
    assert dirty["worktree_clean"] is False
    assert len(dirty["dirty_status_sha256"]) == 64


def test_malicious_pickle_cannot_execute_through_safe_torch_loader(tmp_path):
    marker = tmp_path / "executed"

    class Malicious:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    payload = tmp_path / "malicious.pt"
    torch.save({"payload": Malicious()}, payload)
    with pytest.raises(Exception):
        artifact_module._torch_load(payload, public_safe=True)
    assert not marker.exists()


def test_pickle_capable_loader_is_explicitly_trusted_only(tmp_path, monkeypatch):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    original = artifact_module.torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(artifact_module.torch, "load", recording_load)
    assert load_trusted_legacy_stream(path).stream_id
    assert calls and all(value is False for value in calls)
    with pytest.raises(ValueError, match="cannot disable verification"):
        load_stream(path, verify=False)


@pytest.mark.parametrize("unsafe", ["../escape.pt", "/tmp/escape.pt", "a/b.pt"])
def test_public_audit_rejects_unsafe_checksum_paths(tmp_path, unsafe):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    checksums_path = path / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums[unsafe] = "0" * 64
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="Unsafe|Malformed"):
        audit_stream(path)


def test_public_audit_rejects_payload_symlink_escape(tmp_path):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    payload = path / "tasks.json"
    payload.unlink()
    payload.symlink_to(external)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        audit_stream(path)


def test_public_loader_rejects_symlink_stream_and_store_control_files(tmp_path):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    stream_link = tmp_path / "stream-link"
    stream_link.symlink_to(path, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlink.*stream root"):
        load_stream(stream_link)

    lock_path = tmp_path / artifact_module.STORE_LOCK
    lock_path.unlink()
    lock_path.symlink_to(tmp_path / "external-lock")
    with pytest.raises(ArtifactIntegrityError, match="symlink.*lock"):
        load_stream(path)


def test_gc_removes_stale_temporary_files_only_after_age_gate(tmp_path):
    save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    temporary = tmp_path / "objects" / "scenario" / ".stale.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(b"partial")
    old = time.time() - 7200
    os.utime(temporary, (old, old))
    report = gc_objects(tmp_path, dry_run=True)
    assert report["stale_temporaries_deleted"] == 1
    assert temporary.exists()
    report = gc_objects(tmp_path, dry_run=False)
    assert report["stale_temporaries_deleted"] == 1
    assert not temporary.exists()


def test_manifest_rejects_unsafe_object_key_and_declared_size(tmp_path):
    first = save_stream(_security_stream(), tmp_path / "key", repository_root=tmp_path)
    _rewrite_manifest(
        first,
        lambda manifest: manifest["component_objects"]["scenario.pt"].update(
            {"object_key": "../scenario.pt"}
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="Unsafe object key"):
        audit_stream(first)

    second = save_stream(_security_stream(), tmp_path / "size", repository_root=tmp_path)
    _rewrite_manifest(
        second,
        lambda manifest: manifest["artifact_sizes"].update(
            {"scenario.pt": artifact_module.MAX_ARTIFACT_BYTES + 1}
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="Unsafe declared artifact size"):
        audit_stream(second)


def test_manifest_rejects_unknown_schema_and_component_type(tmp_path):
    first = save_stream(_security_stream(), tmp_path / "schema", repository_root=tmp_path)
    _rewrite_manifest(
        first,
        lambda manifest: manifest.update({"benchmark_schema_version": 99}),
    )
    with pytest.raises(ArtifactIntegrityError, match="Unsupported benchmark schema"):
        audit_stream(first)

    second = save_stream(_security_stream(), tmp_path / "component", repository_root=tmp_path)
    _rewrite_manifest(
        second,
        lambda manifest: manifest["component_objects"]["scenario.pt"].update(
            {"component": "query"}
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="Invalid component type"):
        audit_stream(second)


def test_finalized_shared_objects_cannot_be_opened_writable(tmp_path):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["component_objects"]["scenario.pt"]
    object_path = tmp_path / "objects" / record["object_key"]
    assert object_path.stat().st_mode & 0o222 == 0
    assert (path / "scenario.pt").stat().st_mode & 0o222 == 0
    with pytest.raises(PermissionError):
        object_path.open("r+b")


def test_signature_policy_supports_unsigned_and_rejects_strict_missing(tmp_path):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    assert audit_stream(path, signature_policy="allow_unsigned")["valid"] is True
    with pytest.raises(ArtifactIntegrityError, match="requires a signed manifest"):
        audit_stream(path, signature_policy="require_signed")


def test_signed_manifest_requires_valid_explicit_verifier(tmp_path):
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)

    def sign(manifest):
        manifest["signature"] = {
            "mode": "signed",
            "scheme": "sigstore",
            "identity": "release@example.invalid",
            "signature": "test-detached-signature",
            "signed_package_digest": manifest["package_digest"],
        }

    _rewrite_manifest(path, sign)
    with pytest.raises(ArtifactIntegrityError, match="explicit GPG/Sigstore verifier"):
        audit_stream(path)
    with pytest.raises(ArtifactIntegrityError, match="verification failed"):
        audit_stream(path, signature_verifier=lambda *args: False)
    assert audit_stream(path, signature_verifier=lambda *args: True)["valid"] is True


def test_multi_process_identical_and_distinct_publication_is_safe(tmp_path):
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(target=_process_save, args=(str(tmp_path), profile, queue))
        for profile in ("synchronized", "synchronized", "hard")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]
    assert all(status == "ok" for status, _ in results), results
    paths = {Path(value) for status, value in results}
    assert len(paths) == 2
    assert all(audit_stream(path)["valid"] for path in paths)
    assert not list((tmp_path / "objects").rglob(".*.tmp"))


def test_gc_is_safe_during_manifest_creation_and_trainer_read(tmp_path):
    baseline = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    save_process = context.Process(
        target=_process_save, args=(str(tmp_path), "hard", queue)
    )
    gc_process = context.Process(target=_process_gc, args=(str(tmp_path), queue))
    read_process = context.Process(target=_process_load, args=(str(baseline), queue))
    for process in (save_process, gc_process, read_process):
        process.start()
    for process in (save_process, gc_process, read_process):
        process.join(60)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in range(3)]
    assert all(status == "ok" for status, _ in results), results
    assert audit_stream(baseline)["valid"] is True


def test_process_death_releases_store_lock_and_partial_temp_is_ignored(tmp_path):
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(
        target=_process_die_while_holding_store_lock,
        args=(str(tmp_path), ready),
    )
    process.start()
    assert ready.wait(timeout=10)
    process.terminate()
    process.join(10)
    assert process.exitcode is not None
    path = save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    assert audit_stream(path)["valid"] is True
    assert all(".killed.tmp" not in value for value in gc_objects(tmp_path)["objects"])


def test_interrupted_gc_is_recoverable(tmp_path, monkeypatch):
    save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    for index in range(2):
        orphan = tmp_path / "objects" / "scenario" / (str(index) * 64) / "orphan.pt"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"orphan")
        orphan.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    original = Path.unlink
    calls = {"count": 0}

    def interrupted(path, *args, **kwargs):
        if path.name == "orphan.pt":
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("simulated interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        gc_objects(tmp_path, dry_run=False)
    monkeypatch.setattr(Path, "unlink", original)
    assert gc_objects(tmp_path, dry_run=False)["deleted_objects"] == 1


def test_tar_rsync_hardlink_copy_and_read_only_portability(tmp_path):
    source = tmp_path / "source"
    stream = save_stream(_security_stream(), source, repository_root=tmp_path)
    relative = stream.relative_to(source)

    linked = tmp_path / "linked"
    shutil.copytree(source, linked, copy_function=os.link)
    assert load_stream(linked / relative).stream_id

    archive = tmp_path / "store.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source, arcname="store")
    extracted = tmp_path / "extracted"
    with tarfile.open(archive) as handle:
        handle.extractall(extracted)
    assert load_stream(extracted / "store" / relative).stream_id

    rsynced = tmp_path / "rsynced"
    subprocess.run(
        ["rsync", "-a", f"{source}/", f"{rsynced}/"],
        check=True,
    )
    assert load_stream(rsynced / relative).stream_id

    for path in sorted(source.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(0o555)
    source.chmod(0o555)
    try:
        assert load_stream(stream).stream_id
    finally:
        source.chmod(0o755)
        for path in source.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)


def test_gc_cli_has_non_mutating_default_and_explicit_execute(tmp_path, capsys):
    save_stream(_security_stream(), tmp_path, repository_root=tmp_path)
    orphan = tmp_path / "objects" / "scenario" / ("f" * 64) / "orphan.pt"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    orphan.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    assert main(["gc-objects", "--root", str(tmp_path)]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["dry_run"] is True
    assert orphan.exists()

    assert main(["gc-objects", "--root", str(tmp_path), "--execute"]) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["dry_run"] is False
    assert not orphan.exists()
