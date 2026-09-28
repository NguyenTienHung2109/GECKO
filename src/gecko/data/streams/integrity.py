from __future__ import annotations

from gecko.data.streams.manifest import COMPONENT_FILES
from gecko.data.streams.manifest import MANIFEST_FORMAT
from gecko.data.streams.manifest import MAX_ARTIFACT_BYTES
from gecko.data.streams.manifest import MAX_JSON_BYTES
from gecko.data.streams.manifest import MAX_MANIFEST_BYTES
from gecko.data.streams.manifest import MAX_STREAM_BYTES
from gecko.data.streams.manifest import SIGNATURE_POLICIES

from gecko.data.streams.manifest import MAX_JSON_BYTES

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import Dict
from gecko.validation import ArtifactIntegrityError
from gecko.data.streams.manifest import MAX_JSON_BYTES

_CHECKSUM_CACHE: dict[tuple[int, int, int, int], str] = {}


def _read_json(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> Any:
    if path.is_symlink():
        raise ArtifactIntegrityError(f"Refusing symlink JSON artifact: {path.name}")
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise ArtifactIntegrityError(f"Required JSON artifact is missing: {path.name}") from error
    if size > maximum_bytes:
        raise ArtifactIntegrityError(
            f"JSON artifact exceeds the size limit: {path.name} ({size} bytes)."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"Malformed JSON artifact: {path.name}") from error


def _validate_filename(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ArtifactIntegrityError("Artifact filename must be a non-empty string.")
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or name in {".", ".."}:
        raise ArtifactIntegrityError(f"Unsafe artifact path: {name!r}")
    if "\\" in name or "/" in name or "\x00" in name:
        raise ArtifactIntegrityError(f"Malformed artifact filename: {name!r}")
    return name


def _validate_object_key(key: Any, *, component: str, name: str) -> str:
    if not isinstance(key, str):
        raise ArtifactIntegrityError(f"Object key for {name} must be a string.")
    candidate = PurePosixPath(key)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ArtifactIntegrityError(f"Unsafe object key for {name}: {key!r}")
    if len(candidate.parts) != 3:
        raise ArtifactIntegrityError(f"Malformed object key for {name}: {key!r}")
    key_component, digest, key_name = candidate.parts
    if key_component != component or key_name != name:
        raise ArtifactIntegrityError(f"Object key identity mismatch for {name}.")
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ArtifactIntegrityError(f"Malformed object digest for {name}.")
    return key


def _safe_payload_path(stream_path: Path, name: Any) -> Path:
    filename = _validate_filename(name)
    path = stream_path / filename
    if path.is_symlink():
        raise ArtifactIntegrityError(f"Refusing symlink stream payload: {filename}")
    return path


def _resolve_public_stream_path(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise ArtifactIntegrityError("Refusing a symlink as the public stream root.")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ArtifactIntegrityError(f"Stream path does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise ArtifactIntegrityError(f"Stream path is not a directory: {candidate}")
    return resolved


def _checksum(path: Path) -> str:
    before = path.stat()
    cache_key = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    cached = _CHECKSUM_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    value = digest.hexdigest()
    after = path.stat()
    after_key = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if after_key != cache_key:
        raise ArtifactIntegrityError(f"Artifact changed while hashing: {path}")
    _CHECKSUM_CACHE[cache_key] = value
    return value


def _validate_manifest_schema(manifest: Any, *, public_safe: bool) -> Dict[str, Any]:
    from gecko.data.streams.manifest import COMPONENT_FILES
    from gecko.data.streams.manifest import MANIFEST_FORMAT
    from gecko.data.streams.manifest import MAX_ARTIFACT_BYTES
    from gecko.data.streams.manifest import MAX_STREAM_BYTES
    from gecko.data.streams.manifest import _component_name
    if not isinstance(manifest, dict):
        raise ArtifactIntegrityError("Stream manifest must be a JSON object.")
    required = {
        "manifest_format",
        "benchmark_name",
        "benchmark_schema_version",
        "artifact_checksums",
        "artifact_sizes",
        "scientific_fingerprint",
        "package_digest",
        "signature",
    }
    missing = required - manifest.keys()
    if missing:
        if public_safe:
            raise ArtifactIntegrityError(
                f"Stream manifest is missing safe-schema fields: {sorted(missing)}"
            )
        return manifest
    if manifest["manifest_format"] != MANIFEST_FORMAT:
        raise ArtifactIntegrityError("Unknown UEFA stream manifest format.")
    if manifest["benchmark_name"] != "UEFA":
        raise ArtifactIntegrityError("Manifest benchmark identity is not UEFA.")
    version = manifest["benchmark_schema_version"]
    if type(version) is not int or version not in {1, 2}:
        raise ArtifactIntegrityError(f"Unsupported benchmark schema version: {version!r}")
    for field in ("artifact_checksums", "artifact_sizes"):
        if not isinstance(manifest[field], dict):
            raise ArtifactIntegrityError(f"Manifest {field} must be an object.")
    checksums = manifest["artifact_checksums"]
    sizes = manifest["artifact_sizes"]
    if set(checksums) != set(sizes):
        raise ArtifactIntegrityError("Manifest checksum and size tables do not align.")
    total_size = 0
    for name, digest in checksums.items():
        _validate_filename(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ArtifactIntegrityError(f"Malformed checksum for {name}.")
        size = sizes[name]
        if type(size) is not int or size < 0 or size > MAX_ARTIFACT_BYTES:
            raise ArtifactIntegrityError(f"Unsafe declared artifact size for {name}: {size!r}")
        total_size += size
    if total_size > MAX_STREAM_BYTES:
        raise ArtifactIntegrityError("Declared stream payload exceeds the total size limit.")
    if version == 2:
        store = manifest.get("object_store")
        if store != {
            "scope": "storage_root",
            "root_name": "objects",
            "key_format": "component/sha256/filename",
        }:
            raise ArtifactIntegrityError("Malformed schema-v2 object-store declaration.")
        records = manifest.get("component_objects")
        if not isinstance(records, dict) or set(records) != set(checksums):
            raise ArtifactIntegrityError("Schema-v2 component records do not align.")
        for name, record in records.items():
            if not isinstance(record, dict):
                raise ArtifactIntegrityError(f"Malformed component record for {name}.")
            component = record.get("component")
            if component not in COMPONENT_FILES or _component_name(name) != component:
                raise ArtifactIntegrityError(f"Invalid component type for {name}.")
            if record.get("sha256") != checksums[name]:
                raise ArtifactIntegrityError(f"Component checksum mismatch for {name}.")
            if record.get("size_bytes") != sizes[name]:
                raise ArtifactIntegrityError(f"Component size mismatch for {name}.")
            _validate_object_key(
                record.get("object_key"), component=component, name=name
            )
    _validate_signature_schema(manifest["signature"])
    return manifest


def _validate_signature_schema(signature: Any) -> Dict[str, Any]:
    if not isinstance(signature, dict):
        raise ArtifactIntegrityError("Manifest signature metadata must be an object.")
    expected = {
        "mode",
        "scheme",
        "identity",
        "signature",
        "signed_package_digest",
    }
    if set(signature) != expected:
        raise ArtifactIntegrityError("Manifest signature metadata has an invalid schema.")
    mode = signature["mode"]
    if mode == "unsigned_development":
        if any(signature[key] is not None for key in expected - {"mode"}):
            raise ArtifactIntegrityError("Unsigned signature metadata contains signed fields.")
    elif mode == "signed":
        if signature["scheme"] not in {"gpg", "sigstore"}:
            raise ArtifactIntegrityError("Unsupported manifest signature scheme.")
        for key in ("identity", "signature", "signed_package_digest"):
            if not isinstance(signature[key], str) or not signature[key]:
                raise ArtifactIntegrityError(f"Signed manifest is missing {key}.")
    else:
        raise ArtifactIntegrityError(f"Unknown manifest signature mode: {mode!r}")
    return signature


def _verify_manifest_signature(
    manifest: Dict[str, Any],
    *,
    policy: str,
    verifier: SignatureVerifier | None,
) -> None:
    from gecko.data.streams.manifest import SIGNATURE_POLICIES
    from gecko.data.streams.store import SignatureVerifier
    if policy not in SIGNATURE_POLICIES:
        raise ValueError(f"Unknown signature policy: {policy!r}")
    signature = _validate_signature_schema(manifest["signature"])
    if policy == "ignore":
        return
    if signature["mode"] == "unsigned_development":
        if policy == "require_signed":
            raise ArtifactIntegrityError("Strict signature policy requires a signed manifest.")
        return
    if signature["signed_package_digest"] != manifest["package_digest"]:
        raise ArtifactIntegrityError("Signature targets a different package digest.")
    if verifier is None:
        raise ArtifactIntegrityError(
            "A signed manifest requires an explicit GPG/Sigstore verifier."
        )
    if not verifier(
        signature["scheme"],
        signature["identity"],
        signature["signed_package_digest"],
        signature["signature"],
    ):
        raise ArtifactIntegrityError("Manifest signature verification failed.")


def audit_stream(
    path: str | Path,
    *,
    public_safe: bool = True,
    signature_policy: str = "allow_unsigned",
    signature_verifier: SignatureVerifier | None = None,
) -> Dict[str, Any]:
    from gecko.data.streams.store import SignatureVerifier
    from gecko.data.streams.store import _discover_store_root
    from gecko.data.streams.store import _store_lock
    stream_path = (
        _resolve_public_stream_path(path) if public_safe else Path(path).resolve()
    )
    if public_safe:
        store_root = _discover_store_root(stream_path)
        with _store_lock(store_root, exclusive=False):
            return _audit_stream_unlocked(
                stream_path,
                public_safe=True,
                signature_policy=signature_policy,
                signature_verifier=signature_verifier,
                store_root=store_root,
            )
    return _audit_stream_unlocked(
        stream_path,
        public_safe=False,
        signature_policy=signature_policy,
        signature_verifier=signature_verifier,
        store_root=None,
    )


def _audit_stream_unlocked(
    stream_path: Path,
    *,
    public_safe: bool,
    signature_policy: str,
    signature_verifier: SignatureVerifier | None,
    store_root: Path | None,
) -> Dict[str, Any]:
    from gecko.data.streams.manifest import MAX_ARTIFACT_BYTES
    from gecko.data.streams.manifest import MAX_MANIFEST_BYTES
    from gecko.data.streams.store import SignatureVerifier
    from gecko.data.streams.manifest import _canonical_digest
    from gecko.data.streams.manifest import _package_digest
    from gecko.data.streams.manifest import _scientific_fingerprint
    checksums_path = stream_path / "checksums.json"
    expected = _read_json(checksums_path)
    if not isinstance(expected, dict):
        raise ArtifactIntegrityError("checksums.json must be a JSON object.")
    failures = {}
    for name, digest in expected.items():
        artifact = _safe_payload_path(stream_path, name)
        if not artifact.exists():
            failures[name] = {"expected": digest, "actual": None}
            continue
        size = artifact.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise ArtifactIntegrityError(
                f"Artifact exceeds the public size limit: {name} ({size} bytes)."
            )
        actual = _checksum(artifact)
        if actual != digest:
            failures[name] = {"expected": digest, "actual": actual}
    if failures:
        raise ArtifactIntegrityError(f"Stream checksum failure: {failures}")
    manifest_path = stream_path / "manifest.json"
    manifest = _validate_manifest_schema(
        _read_json(manifest_path, maximum_bytes=MAX_MANIFEST_BYTES),
        public_safe=public_safe,
    )
    actual_payload_checksums = {
        name: _checksum(_safe_payload_path(stream_path, name))
        for name in manifest["artifact_checksums"]
    }
    if actual_payload_checksums != manifest["artifact_checksums"]:
        raise ArtifactIntegrityError("Manifest artifact_checksums do not match payload files.")
    if manifest.get("artifact_layout") == "content_addressed_hardlink_v2":
        if store_root is None and public_safe:
            raise ArtifactIntegrityError("Schema-v2 public audit requires a store root.")
        object_root = None if store_root is None else (store_root / "objects").resolve()
        grouped: Dict[str, Dict[str, str]] = {}
        for name, record in manifest.get("component_objects", {}).items():
            payload_path = _safe_payload_path(stream_path, name)
            if payload_path.stat().st_mode & 0o222:
                raise ArtifactIntegrityError(f"Stream payload is writable: {name}")
            if object_root is not None:
                object_path = object_root / record["object_key"]
                if object_path.is_symlink():
                    raise ArtifactIntegrityError(f"Component object is a symlink: {name}")
                resolved_object = object_path.resolve()
                if not resolved_object.is_relative_to(object_root):
                    raise ArtifactIntegrityError(f"Component object escapes store: {name}")
                if (
                    not resolved_object.exists()
                    or resolved_object.stat().st_size != record["size_bytes"]
                    or _checksum(resolved_object) != record["sha256"]
                ):
                    raise ArtifactIntegrityError(
                        f"Component object checksum/size failure: {name}"
                    )
                if resolved_object.stat().st_mode & 0o222:
                    raise ArtifactIntegrityError(
                        f"Component object is writable after finalization: {name}"
                    )
            if manifest["artifact_checksums"].get(name) != record["sha256"]:
                raise ArtifactIntegrityError(
                    f"Component and stream payload disagree: {name}"
                )
            grouped.setdefault(record["component"], {})[name] = record["sha256"]
        actual_component_digests = {
            component: _canonical_digest(files)
            for component, files in sorted(grouped.items())
        }
        if actual_component_digests != manifest.get("component_digests"):
            raise ArtifactIntegrityError("Component digest mismatch.")
    scientific = _scientific_fingerprint(manifest)
    if scientific != manifest["scientific_fingerprint"]:
        raise ArtifactIntegrityError("Scientific fingerprint mismatch.")
    package = _package_digest(manifest)
    if package != manifest["package_digest"]:
        raise ArtifactIntegrityError("Package digest mismatch.")
    if "signature" in manifest:
        _verify_manifest_signature(
            manifest,
            policy=signature_policy,
            verifier=signature_verifier,
        )
    elif signature_policy == "require_signed":
        raise ArtifactIntegrityError("Strict signature policy requires signature metadata.")
    return {
        "valid": True,
        "files": len(expected),
        "checksums": expected,
        "scientific_fingerprint": scientific,
        "package_digest": package,
        "artifact_layout": manifest.get("artifact_layout", "self_contained_v1"),
        "component_digests": manifest.get("component_digests", {}),
        "audit_policy_version": manifest.get("audit_policy_version"),
        "audit_policy_hash": manifest.get("audit_policy_hash"),
        "invariant_status": manifest.get("invariant_status", "unassessed"),
        "quality_status": manifest.get("quality_status", "unassessed"),
        "release_status": manifest.get("release_status", "unassessed"),
        "release_eligibility": manifest.get("release_eligibility", False),
        "benchmark_tier": manifest.get("benchmark_tier", "unassessed"),
        "benchmark_eligible": manifest.get("benchmark_eligible", False),
        "leaderboard_ready": manifest.get("leaderboard_ready", False),
    }


