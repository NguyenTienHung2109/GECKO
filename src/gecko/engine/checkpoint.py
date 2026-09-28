"""Safe round/stage checkpoint serialization for stateful UEFA runs."""

from __future__ import annotations


import hashlib
import json
import math
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Iterator
from typing import Mapping

import torch


CHECKPOINT_FORMAT = "uefa-round-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_LOCK = ".uefa-checkpoint.lock"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_WEIGHTS_BYTES = 32 * 1024 * 1024 * 1024
MAX_LOGICAL_TENSOR_BYTES = 64 * 1024 * 1024 * 1024
MAX_SAFE_TREE_DEPTH = 128
MAX_SAFE_TREE_NODES = 2_000_000
MAX_TENSORS = 1_000_000

_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {
    "format",
    "schema_version",
    "checkpoint_id",
    "created_at_utc",
    "identity",
    "cursor_tree",
    "state_tree",
    "tensor_bundle",
    "tensor_index",
    "manifest_sha256",
}


class CheckpointError(RuntimeError):
    """Base class for safe checkpoint failures."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint artifact or safe-state schema is invalid."""


class CheckpointIdentityError(CheckpointError):
    """Raised when resume identity differs without a diagnostic override."""


@dataclass(frozen=True)
class CheckpointIdentity:
    """Scientific and execution identity that must match on normal resume."""

    stream_id: str
    stream_hash: str
    scientific_fingerprint: str
    source_sha: str | None
    stream_content_digest: str
    source_tree_digest: str | None
    run_config_digest: str
    method_config_digest: str
    strategy: str
    method: str
    model: str
    model_seed: int

    def __post_init__(self) -> None:
        for name in ("stream_id", "strategy", "method", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Checkpoint identity {name} must be non-empty.")
        for name in (
            "stream_hash",
            "scientific_fingerprint",
            "stream_content_digest",
            "run_config_digest",
            "method_config_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
                raise ValueError(
                    f"Checkpoint identity {name} must be lowercase SHA-256."
                )
        for name in ("source_sha", "source_tree_digest"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"Optional checkpoint metadata {name} must be text or null.")
        if isinstance(self.model_seed, bool) or not isinstance(self.model_seed, int):
            raise TypeError("Checkpoint identity model_seed must be an integer.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointIdentity":
        if not isinstance(value, Mapping):
            raise CheckpointIntegrityError("Checkpoint identity must be an object.")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise CheckpointIntegrityError(
                "Checkpoint identity fields do not match the required schema."
            )
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as error:
            raise CheckpointIntegrityError(str(error)) from error


@dataclass(frozen=True)
class ResumeValidation:
    """Recorded result of the resume identity gate."""

    identity_matched: bool
    diagnostic_override_used: bool
    benchmark_eligible: bool
    mismatch_fields: Mapping[str, Mapping[str, Any]]
    diagnostic_override_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_matched": self.identity_matched,
            "diagnostic_override_used": self.diagnostic_override_used,
            "benchmark_eligible": self.benchmark_eligible,
            "mismatch_fields": {
                name: dict(values)
                for name, values in sorted(self.mismatch_fields.items())
            },
            "diagnostic_override_reason": self.diagnostic_override_reason,
        }


@dataclass(frozen=True)
class CheckpointWriteResult:
    manifest_path: Path
    weights_path: Path
    manifest_bytes: int
    weights_bytes: int
    total_bytes: int
    manifest_sha256: str
    weights_sha256: str


@dataclass(frozen=True)
class CheckpointLoadResult:
    state: Any
    cursor: Any
    identity: CheckpointIdentity
    resume_validation: ResumeValidation
    manifest_path: Path
    weights_path: Path
    manifest_bytes: int
    weights_bytes: int
    total_bytes: int
    manifest_sha256: str
    weights_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointIntegrityError(
            "Checkpoint metadata is not canonical JSON-safe."
        ) from error
    return (rendered + "\n").encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    """Hash finite JSON primitives with the checkpoint canonical encoding."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlink_components(path: Path) -> None:
    candidate = path.absolute()
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise CheckpointIntegrityError(
                f"Refusing symlink checkpoint path component: {current}"
            )


def _validate_checkpoint_id(value: str) -> str:
    if not isinstance(value, str) or _CHECKPOINT_ID.fullmatch(value) is None:
        raise ValueError(
            "checkpoint_id must be 1-128 safe alphanumeric/dot/dash/underscore characters."
        )
    return value


def _validate_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise CheckpointIntegrityError("Checkpoint filename must be a safe basename.")
    if "/" in value or "\\" in value or "\x00" in value:
        raise CheckpointIntegrityError(f"Unsafe checkpoint filename: {value!r}")
    return value


@contextmanager
def _checkpoint_lock(directory: Path) -> Iterator[None]:
    lock_path = directory / CHECKPOINT_LOCK
    if lock_path.is_symlink():
        raise CheckpointIntegrityError("Refusing a symlink checkpoint lock.")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_publish_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class _SafeTreeEncoder:
    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.tensor_index: dict[str, dict[str, Any]] = {}
        self._tensor_ids: dict[int, str] = {}
        self._active_containers: set[int] = set()
        self._nodes = 0
        self._logical_tensor_bytes = 0

    def encode(self, value: Any, *, depth: int = 0) -> dict[str, Any]:
        self._nodes += 1
        if self._nodes > MAX_SAFE_TREE_NODES:
            raise CheckpointIntegrityError("Checkpoint safe-state tree is too large.")
        if depth > MAX_SAFE_TREE_DEPTH:
            raise CheckpointIntegrityError("Checkpoint safe-state tree is too deep.")
        if value is None:
            return {"kind": "none"}
        if type(value) is bool:
            return {"kind": "bool", "value": value}
        if type(value) is int:
            return {"kind": "int", "value": value}
        if type(value) is float:
            if not math.isfinite(value):
                raise CheckpointIntegrityError(
                    "Checkpoint safe-state floats must be finite."
                )
            return {"kind": "float", "value": value}
        if type(value) is str:
            return {"kind": "string", "value": value}
        if torch.is_tensor(value):
            return self._encode_tensor(value)
        if type(value) is dict:
            return self._encode_mapping(value, depth=depth)
        if type(value) in {list, tuple}:
            return self._encode_sequence(value, depth=depth)
        raise CheckpointIntegrityError(
            "Checkpoint state supports only tensors, finite primitives, "
            "dicts with str/int keys, lists, and tuples; "
            f"received {type(value).__name__}."
        )

    def _encode_tensor(self, value: torch.Tensor) -> dict[str, Any]:
        existing = self._tensor_ids.get(id(value))
        if existing is not None:
            return {"kind": "tensor", "key": existing}
        if len(self.tensors) >= MAX_TENSORS:
            raise CheckpointIntegrityError("Checkpoint contains too many tensors.")
        if value.layout != torch.strided or value.is_quantized:
            raise CheckpointIntegrityError(
                "Checkpoint tensors must be dense, strided, and non-quantized."
            )
        key = f"tensor/{len(self.tensors):08d}"
        tensor = value.detach().cpu().contiguous().clone()
        logical_bytes = tensor.numel() * tensor.element_size()
        self._logical_tensor_bytes += logical_bytes
        if self._logical_tensor_bytes > MAX_LOGICAL_TENSOR_BYTES:
            raise CheckpointIntegrityError(
                "Checkpoint tensors exceed the logical byte limit."
            )
        self._tensor_ids[id(value)] = key
        self.tensors[key] = tensor
        self.tensor_index[key] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "logical_bytes": logical_bytes,
        }
        return {"kind": "tensor", "key": key}

    @staticmethod
    def _encode_key(value: Any) -> dict[str, Any]:
        if type(value) is str:
            return {"kind": "string", "value": value}
        if type(value) is int:
            return {"kind": "int", "value": value}
        raise CheckpointIntegrityError(
            "Checkpoint mapping keys must be strings or integers."
        )

    def _enter_container(self, value: Any) -> None:
        identifier = id(value)
        if identifier in self._active_containers:
            raise CheckpointIntegrityError("Checkpoint state cannot contain cycles.")
        self._active_containers.add(identifier)

    def _leave_container(self, value: Any) -> None:
        self._active_containers.remove(id(value))

    def _encode_mapping(self, value: dict[Any, Any], *, depth: int) -> dict[str, Any]:
        self._enter_container(value)
        try:
            keyed = [(self._encode_key(key), item) for key, item in value.items()]
            keyed.sort(key=lambda pair: _canonical_json_bytes(pair[0]))
            items = [
                {
                    "key": encoded_key,
                    "value": self.encode(item, depth=depth + 1),
                }
                for encoded_key, item in keyed
            ]
            return {"kind": "mapping", "items": items}
        finally:
            self._leave_container(value)

    def _encode_sequence(
        self, value: list[Any] | tuple[Any, ...], *, depth: int
    ) -> dict[str, Any]:
        self._enter_container(value)
        try:
            return {
                "kind": "tuple" if type(value) is tuple else "list",
                "items": [self.encode(item, depth=depth + 1) for item in value],
            }
        finally:
            self._leave_container(value)


class _SafeTreeDecoder:
    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self._nodes = 0

    def decode(self, node: Any, *, depth: int = 0) -> Any:
        self._nodes += 1
        if self._nodes > MAX_SAFE_TREE_NODES:
            raise CheckpointIntegrityError("Checkpoint safe-state tree is too large.")
        if depth > MAX_SAFE_TREE_DEPTH:
            raise CheckpointIntegrityError("Checkpoint safe-state tree is too deep.")
        if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
            raise CheckpointIntegrityError("Malformed checkpoint safe-state node.")
        kind = node["kind"]
        if kind == "none":
            self._require_fields(node, {"kind"})
            return None
        if kind == "bool":
            self._require_fields(node, {"kind", "value"})
            if type(node["value"]) is not bool:
                raise CheckpointIntegrityError("Malformed checkpoint bool node.")
            return node["value"]
        if kind == "int":
            self._require_fields(node, {"kind", "value"})
            if type(node["value"]) is not int:
                raise CheckpointIntegrityError("Malformed checkpoint int node.")
            return node["value"]
        if kind == "float":
            self._require_fields(node, {"kind", "value"})
            if type(node["value"]) is not float or not math.isfinite(node["value"]):
                raise CheckpointIntegrityError("Malformed checkpoint float node.")
            return node["value"]
        if kind == "string":
            self._require_fields(node, {"kind", "value"})
            if type(node["value"]) is not str:
                raise CheckpointIntegrityError("Malformed checkpoint string node.")
            return node["value"]
        if kind == "tensor":
            self._require_fields(node, {"kind", "key"})
            key = node["key"]
            if not isinstance(key, str) or key not in self.tensors:
                raise CheckpointIntegrityError(
                    "Checkpoint references an unknown tensor."
                )
            return self.tensors[key]
        if kind in {"list", "tuple"}:
            self._require_fields(node, {"kind", "items"})
            if not isinstance(node["items"], list):
                raise CheckpointIntegrityError("Malformed checkpoint sequence node.")
            values = [self.decode(item, depth=depth + 1) for item in node["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "mapping":
            self._require_fields(node, {"kind", "items"})
            if not isinstance(node["items"], list):
                raise CheckpointIntegrityError("Malformed checkpoint mapping node.")
            output: dict[str | int, Any] = {}
            for item in node["items"]:
                if not isinstance(item, dict) or set(item) != {"key", "value"}:
                    raise CheckpointIntegrityError("Malformed checkpoint mapping item.")
                key = self._decode_key(item["key"])
                if key in output:
                    raise CheckpointIntegrityError("Duplicate checkpoint mapping key.")
                output[key] = self.decode(item["value"], depth=depth + 1)
            return output
        raise CheckpointIntegrityError(f"Unknown checkpoint safe-state kind: {kind!r}")

    @staticmethod
    def _require_fields(node: Mapping[str, Any], expected: set[str]) -> None:
        if set(node) != expected:
            raise CheckpointIntegrityError(
                "Checkpoint safe-state node has extra fields."
            )

    @classmethod
    def _decode_key(cls, node: Any) -> str | int:
        if not isinstance(node, dict) or set(node) != {"kind", "value"}:
            raise CheckpointIntegrityError("Malformed checkpoint mapping key.")
        if node["kind"] == "string" and type(node["value"]) is str:
            return node["value"]
        if node["kind"] == "int" and type(node["value"]) is int:
            return node["value"]
        raise CheckpointIntegrityError("Checkpoint mapping key has an invalid type.")


def _write_tensor_temporary(
    directory: Path, tensors: Mapping[str, torch.Tensor]
) -> Path:
    temporary = directory / f".weights.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(tensors), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > MAX_WEIGHTS_BYTES:
            raise CheckpointIntegrityError(
                "Checkpoint tensor bundle exceeds the size limit."
            )
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_tensor_bundle(directory: Path, temporary: Path) -> tuple[Path, str, int]:
    digest = _sha256_file(temporary)
    size = temporary.stat().st_size
    target = directory / f"weights-{digest}.pt"
    if target.is_symlink():
        raise CheckpointIntegrityError("Refusing a symlink checkpoint tensor bundle.")
    if target.exists():
        if not target.is_file() or target.stat().st_size != size:
            raise CheckpointIntegrityError(
                "Existing content-addressed tensor bundle has an incompatible size."
            )
        if _sha256_file(target) != digest:
            raise CheckpointIntegrityError(
                "Existing content-addressed tensor bundle has an incompatible checksum."
            )
        temporary.unlink()
    else:
        os.replace(temporary, target)
        _fsync_directory(directory)
    return target, digest, size


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def save_checkpoint(
    directory: str | Path,
    checkpoint_id: str,
    *,
    identity: CheckpointIdentity,
    cursor: Any,
    state: Any,
) -> CheckpointWriteResult:
    """Atomically publish a safe tensor bundle, then its manifest commit point."""

    checkpoint_id = _validate_checkpoint_id(checkpoint_id)
    if not isinstance(identity, CheckpointIdentity):
        raise TypeError("identity must be a CheckpointIdentity.")
    root = Path(directory).absolute()
    _reject_symlink_components(root)
    root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root)
    encoder = _SafeTreeEncoder()
    cursor_tree = encoder.encode(cursor)
    state_tree = encoder.encode(state)
    manifest_path = root / f"{checkpoint_id}.manifest.json"
    with _checkpoint_lock(root):
        temporary = _write_tensor_temporary(root, encoder.tensors)
        try:
            weights_path, weights_digest, weights_size = _publish_tensor_bundle(
                root, temporary
            )
        finally:
            temporary.unlink(missing_ok=True)
        manifest: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity.to_dict(),
            "cursor_tree": cursor_tree,
            "state_tree": state_tree,
            "tensor_bundle": {
                "filename": weights_path.name,
                "sha256": weights_digest,
                "size_bytes": weights_size,
            },
            "tensor_index": encoder.tensor_index,
        }
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        manifest_payload = _canonical_json_bytes(manifest)
        if len(manifest_payload) > MAX_MANIFEST_BYTES:
            raise CheckpointIntegrityError(
                "Checkpoint manifest exceeds the size limit."
            )
        _atomic_publish_bytes(manifest_path, manifest_payload)
    return CheckpointWriteResult(
        manifest_path=manifest_path,
        weights_path=weights_path,
        manifest_bytes=len(manifest_payload),
        weights_bytes=weights_size,
        total_bytes=len(manifest_payload) + weights_size,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        weights_sha256=weights_digest,
    )


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    _reject_symlink_components(path)
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise CheckpointIntegrityError(
            f"Checkpoint manifest is missing: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode):
        raise CheckpointIntegrityError("Refusing a symlink checkpoint manifest.")
    if not stat.S_ISREG(metadata.st_mode):
        raise CheckpointIntegrityError("Checkpoint manifest is not a regular file.")
    if metadata.st_size > MAX_MANIFEST_BYTES:
        raise CheckpointIntegrityError("Checkpoint manifest exceeds the size limit.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        payload = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise CheckpointIntegrityError("Checkpoint manifest exceeds the size limit.")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(
            "Checkpoint manifest is malformed JSON."
        ) from error
    if not isinstance(value, dict):
        raise CheckpointIntegrityError("Checkpoint manifest must be a JSON object.")
    return value, payload


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        raise CheckpointIntegrityError(
            "Checkpoint manifest fields do not match the schema."
        )
    if manifest["format"] != CHECKPOINT_FORMAT:
        raise CheckpointIntegrityError("Unknown checkpoint manifest format.")
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIntegrityError("Unsupported checkpoint schema version.")
    try:
        _validate_checkpoint_id(manifest["checkpoint_id"])
    except (TypeError, ValueError) as error:
        raise CheckpointIntegrityError(str(error)) from error
    if (
        not isinstance(manifest["created_at_utc"], str)
        or not manifest["created_at_utc"]
    ):
        raise CheckpointIntegrityError("Checkpoint creation time is missing.")
    CheckpointIdentity.from_dict(manifest["identity"])
    bundle = manifest["tensor_bundle"]
    if not isinstance(bundle, dict) or set(bundle) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise CheckpointIntegrityError("Malformed checkpoint tensor bundle record.")
    filename = _validate_filename(bundle["filename"])
    digest = bundle["sha256"]
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        raise CheckpointIntegrityError("Malformed checkpoint tensor checksum.")
    if filename != f"weights-{digest}.pt":
        raise CheckpointIntegrityError("Checkpoint tensor filename/checksum mismatch.")
    size = bundle["size_bytes"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not (0 <= size <= MAX_WEIGHTS_BYTES)
    ):
        raise CheckpointIntegrityError("Unsafe checkpoint tensor bundle size.")
    index = manifest["tensor_index"]
    if not isinstance(index, dict) or len(index) > MAX_TENSORS:
        raise CheckpointIntegrityError("Malformed checkpoint tensor index.")
    logical_total = 0
    for key, record in index.items():
        if not isinstance(key, str) or re.fullmatch(r"tensor/[0-9]{8}", key) is None:
            raise CheckpointIntegrityError("Malformed checkpoint tensor key.")
        if not isinstance(record, dict) or set(record) != {
            "dtype",
            "shape",
            "numel",
            "logical_bytes",
        }:
            raise CheckpointIntegrityError("Malformed checkpoint tensor metadata.")
        if not isinstance(record["dtype"], str) or not record["dtype"].startswith(
            "torch."
        ):
            raise CheckpointIntegrityError("Malformed checkpoint tensor dtype.")
        shape = record["shape"]
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise CheckpointIntegrityError("Malformed checkpoint tensor shape.")
        numel = record["numel"]
        logical_bytes = record["logical_bytes"]
        if (
            isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel < 0
            or isinstance(logical_bytes, bool)
            or not isinstance(logical_bytes, int)
            or logical_bytes < 0
        ):
            raise CheckpointIntegrityError("Malformed checkpoint tensor byte metadata.")
        expected_numel = math.prod(shape)
        if numel != expected_numel:
            raise CheckpointIntegrityError("Checkpoint tensor numel/shape mismatch.")
        logical_total += logical_bytes
    if logical_total > MAX_LOGICAL_TENSOR_BYTES:
        raise CheckpointIntegrityError(
            "Checkpoint logical tensors exceed the size limit."
        )
    manifest_digest = manifest["manifest_sha256"]
    if (
        not isinstance(manifest_digest, str)
        or _HEX_64.fullmatch(manifest_digest) is None
    ):
        raise CheckpointIntegrityError("Malformed checkpoint manifest checksum.")
    if _manifest_digest(manifest) != manifest_digest:
        raise CheckpointIntegrityError("Checkpoint manifest checksum mismatch.")


def _load_tensors(
    path: Path,
    *,
    expected_digest: str,
    expected_size: int,
    tensor_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    _reject_symlink_components(path)
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise CheckpointIntegrityError(
            "Checkpoint tensor bundle is missing."
        ) from error
    if stat.S_ISLNK(metadata.st_mode):
        raise CheckpointIntegrityError("Refusing a symlink checkpoint tensor bundle.")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
        raise CheckpointIntegrityError("Checkpoint tensor bundle size mismatch.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        digest = hashlib.sha256()
        while block := handle.read(1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise CheckpointIntegrityError(
                "Checkpoint tensor bundle checksum mismatch."
            )
        handle.seek(0)
        try:
            raw = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise CheckpointIntegrityError(
                "Checkpoint tensor bundle failed safe weights-only loading."
            ) from error
    if not isinstance(raw, dict) or set(raw) != set(tensor_index):
        raise CheckpointIntegrityError(
            "Checkpoint tensor keys do not match the manifest."
        )
    output: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise CheckpointIntegrityError(
                "Checkpoint tensor bundle contains a non-tensor payload."
            )
        record = tensor_index[key]
        if tensor.device.type != "cpu":
            raise CheckpointIntegrityError("Checkpoint tensor did not load on CPU.")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise CheckpointIntegrityError("Checkpoint tensor layout is unsupported.")
        if (
            str(tensor.dtype) != record["dtype"]
            or list(tensor.shape) != record["shape"]
            or tensor.numel() != record["numel"]
            or tensor.numel() * tensor.element_size() != record["logical_bytes"]
        ):
            raise CheckpointIntegrityError("Checkpoint tensor metadata mismatch.")
        output[key] = tensor
    return output


def _identity_mismatches(
    expected: CheckpointIdentity, actual: CheckpointIdentity
) -> dict[str, dict[str, Any]]:
    """Compare scientific/execution identity, excluding informational source fields."""

    expected_values = asdict(expected)
    actual_values = asdict(actual)
    return {
        name: {"expected": expected_values[name], "checkpoint": actual_values[name]}
        for name in sorted(expected_values)
        if name not in {"source_sha", "source_tree_digest"}
        and expected_values[name] != actual_values[name]
    }


def load_checkpoint(
    manifest_path: str | Path,
    *,
    expected_identity: CheckpointIdentity,
    diagnostic_override_reason: str | None = None,
) -> CheckpointLoadResult:
    """Load a verified weights-only checkpoint and enforce resume identity."""

    if not isinstance(expected_identity, CheckpointIdentity):
        raise TypeError("expected_identity must be a CheckpointIdentity.")
    path = Path(manifest_path).absolute()
    manifest, manifest_payload = _read_manifest(path)
    _validate_manifest(manifest)
    actual_identity = CheckpointIdentity.from_dict(manifest["identity"])
    mismatches = _identity_mismatches(expected_identity, actual_identity)
    if mismatches:
        if diagnostic_override_reason is None:
            raise CheckpointIdentityError(
                "Checkpoint identity mismatch: " + ", ".join(sorted(mismatches))
            )
        if (
            not isinstance(diagnostic_override_reason, str)
            or not diagnostic_override_reason.strip()
        ):
            raise ValueError(
                "A diagnostic identity override requires a non-empty reason."
            )
        validation = ResumeValidation(
            identity_matched=False,
            diagnostic_override_used=True,
            benchmark_eligible=False,
            mismatch_fields=mismatches,
            diagnostic_override_reason=diagnostic_override_reason.strip(),
        )
    else:
        if diagnostic_override_reason is not None:
            raise ValueError(
                "Diagnostic override was supplied but checkpoint identity matches."
            )
        validation = ResumeValidation(
            identity_matched=True,
            diagnostic_override_used=False,
            benchmark_eligible=True,
            mismatch_fields={},
            diagnostic_override_reason=None,
        )
    bundle = manifest["tensor_bundle"]
    weights_path = path.parent / _validate_filename(bundle["filename"])
    if weights_path.parent.resolve() != path.parent.resolve():
        raise CheckpointIntegrityError(
            "Checkpoint tensor bundle escapes its directory."
        )
    tensors = _load_tensors(
        weights_path,
        expected_digest=bundle["sha256"],
        expected_size=bundle["size_bytes"],
        tensor_index=manifest["tensor_index"],
    )
    decoder = _SafeTreeDecoder(tensors)
    cursor = decoder.decode(manifest["cursor_tree"])
    state = decoder.decode(manifest["state_tree"])
    return CheckpointLoadResult(
        state=state,
        cursor=cursor,
        identity=actual_identity,
        resume_validation=validation,
        manifest_path=path,
        weights_path=weights_path,
        manifest_bytes=len(manifest_payload),
        weights_bytes=bundle["size_bytes"],
        total_bytes=len(manifest_payload) + bundle["size_bytes"],
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        weights_sha256=bundle["sha256"],
    )
