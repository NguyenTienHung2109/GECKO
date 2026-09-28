from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Mapping
import torch

DSLR_REPLAY_CEILING_BYTES = 16 * 1024 * 1024


DSLR_REPLAY_FRACTION = 0.05


DSLR_DEFAULT_STRUCTURE_LAMBDA = 0.5


DSLR_DEFAULT_TOP_N = 5


DSLR_DEFAULT_CANDIDATE_K = 50


DSLR_DEFAULT_TAU = 0.8


DSLR_DEFAULT_STRUCTURE_EPOCHS = 99


DSLR_DEFAULT_STRUCTURE_LR = 0.01


DSLR_DEFAULT_STRUCTURE_HEADS = 4


DSLR_SELECTION_MODES = ("coverage_diversity", "mean_feature")


DSLR_STRUCTURE_MODES = ("none", "link_only", "node_only", "full")


DSLR_LINK_REDUCTIONS = ("sum", "mean")


DSLR_DIAGNOSTIC_VARIANTS = {
    ("mean_feature", "none"): "mf_no_structure",
    ("coverage_diversity", "none"): "cd_selection_only",
    ("mean_feature", "full"): "mf_full_structure",
    ("coverage_diversity", "link_only"): "cd_link_only",
    ("coverage_diversity", "node_only"): "cd_node_only",
}


_SNAPSHOT_FORMAT = "uefa-dslr-replay-snapshot-v1"


_SNAPSHOT_MAGIC = b"UEFA-DSLR-SNAPSHOT-V1\0"


_CHECKPOINT_VERSION = "uefa-dslr-method-checkpoint-v2-explicit-link-reduction"


_STATE_VERSION = "uefa-dslr-private-state-v3-explicit-link-reduction"


_MAX_METADATA_BYTES = 64 * 1024


_TENSOR_ORDER = ("feature", "embedding", "candidate_local_indices")


_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


_DTYPES = {
    str(torch.float16): torch.float16,
    str(torch.bfloat16): torch.bfloat16,
    str(torch.float32): torch.float32,
    str(torch.float64): torch.float64,
    str(torch.int64): torch.int64,
}


def _canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("DSLR metadata is not canonical JSON-safe.") from error
    return rendered.encode("utf-8")


def _owned_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.is_quantized:
        raise ValueError(f"{name} must be a dense, strided tensor.")
    return value.detach().cpu().clone().contiguous()


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _validate_positive_int(value: object, *, name: str) -> int:
    checked = _validate_nonnegative_int(value, name=name)
    if checked == 0:
        raise ValueError(f"{name} must be positive.")
    return checked


def _validate_probability(
    value: object, *, name: str, open_upper: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar.")
    checked = float(value)
    upper_ok = checked < 1.0 if open_upper else checked <= 1.0
    if not math.isfinite(checked) or checked < 0.0 or not upper_ok:
        relation = "[0, 1)" if open_upper else "[0, 1]"
        raise ValueError(f"{name} must be finite and in {relation}.")
    return checked


def _validate_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar.")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return checked


def _validate_local_edges(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    name: str = "edge_index",
) -> torch.Tensor:
    edges = _owned_tensor(edge_index, name=name)
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"{name} must have int64 shape [2, num_edges].")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError(f"{name} contains an endpoint outside the strict-local graph.")
    return edges


def _validate_local_indices(
    values: torch.Tensor,
    *,
    num_nodes: int,
    name: str,
    unique: bool = False,
) -> torch.Tensor:
    indices = _owned_tensor(values, name=name)
    if indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 tensor.")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= num_nodes):
        raise ValueError(f"{name} contains an endpoint outside the strict-local graph.")
    if unique and len(set(int(value) for value in indices.tolist())) != indices.numel():
        raise ValueError(f"{name} must not contain duplicates.")
    return indices


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    """Hash nested checkpoint-safe state with explicit container type tags."""

    if torch.is_tensor(value):
        digest.update(b"tensor\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(_tensor_bytes(value))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _update_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))


def _tensor_descriptor(value: torch.Tensor) -> Dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "nbytes": int(value.numel() * value.element_size()),
    }


@dataclass(frozen=True, init=False, eq=False)
class DSLRReplaySnapshot:
    """One selected local node and its paper-Eq.-(11) candidate set.

    Candidate indices are frozen when the snapshot is selected, before a later
    task arrives.  This preserves the paper's exclusion of future-task nodes
    without retaining a global graph or any candidate label.
    """

    client_id: int
    global_task_id: int
    stage_index: int
    class_id: int
    source_local_index: int
    snapshot_id: str
    _feature: torch.Tensor
    _embedding: torch.Tensor
    _candidate_local_indices: torch.Tensor

    def __init__(
        self,
        *,
        client_id: int,
        global_task_id: int,
        stage_index: int,
        class_id: int,
        source_local_index: int,
        feature: torch.Tensor,
        embedding: torch.Tensor,
        candidate_local_indices: torch.Tensor,
    ) -> None:
        for name, value in (
            ("client_id", client_id),
            ("global_task_id", global_task_id),
            ("stage_index", stage_index),
            ("class_id", class_id),
            ("source_local_index", source_local_index),
        ):
            object.__setattr__(self, name, _validate_nonnegative_int(value, name=name))
        owned_feature = _owned_tensor(feature, name="feature")
        owned_embedding = _owned_tensor(embedding, name="embedding")
        candidates = _owned_tensor(
            candidate_local_indices, name="candidate_local_indices"
        )
        if owned_feature.dtype not in _FLOAT_DTYPES or owned_feature.ndim != 1:
            raise ValueError(
                "DSLR snapshot feature must be one-dimensional floating point."
            )
        if owned_embedding.dtype not in _FLOAT_DTYPES or owned_embedding.ndim != 1:
            raise ValueError(
                "DSLR snapshot embedding must be one-dimensional floating point."
            )
        if not owned_feature.numel() or not owned_embedding.numel():
            raise ValueError("DSLR snapshot feature and embedding cannot be empty.")
        if not bool(torch.isfinite(owned_feature).all()) or not bool(
            torch.isfinite(owned_embedding).all()
        ):
            raise ValueError("DSLR snapshot tensors must be finite.")
        if candidates.dtype != torch.long or candidates.ndim != 1:
            raise ValueError("candidate_local_indices must be one-dimensional int64.")
        if candidates.numel() and int(candidates.min()) < 0:
            raise ValueError("candidate_local_indices cannot contain negative values.")
        candidate_values = tuple(int(value) for value in candidates.tolist())
        if len(candidate_values) != len(set(candidate_values)):
            raise ValueError("candidate_local_indices must not contain duplicates.")
        if self.source_local_index in candidate_values:
            raise ValueError("A replay node cannot be its own Eq.-(11) candidate.")
        object.__setattr__(self, "_feature", owned_feature)
        object.__setattr__(self, "_embedding", owned_embedding)
        object.__setattr__(
            self,
            "_candidate_local_indices",
            candidates,
        )
        object.__setattr__(self, "snapshot_id", _snapshot_digest(self))

    @property
    def feature(self) -> torch.Tensor:
        return self._feature.clone()

    @property
    def embedding(self) -> torch.Tensor:
        return self._embedding.clone()

    @property
    def candidate_local_indices(self) -> torch.Tensor:
        return self._candidate_local_indices.clone()

    @property
    def serialized_bytes(self) -> int:
        return len(serialize_dslr_snapshot(self))


def _snapshot_metadata(
    snapshot: DSLRReplaySnapshot, *, include_snapshot_id: bool
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "format": _SNAPSHOT_FORMAT,
        "byteorder": sys.byteorder,
        "client_id": snapshot.client_id,
        "global_task_id": snapshot.global_task_id,
        "stage_index": snapshot.stage_index,
        "class_id": snapshot.class_id,
        "source_local_index": snapshot.source_local_index,
        "tensors": {
            "feature": _tensor_descriptor(snapshot._feature),
            "embedding": _tensor_descriptor(snapshot._embedding),
            "candidate_local_indices": _tensor_descriptor(
                snapshot._candidate_local_indices
            ),
        },
    }
    if include_snapshot_id:
        metadata["snapshot_id"] = snapshot.snapshot_id
    return metadata


def _snapshot_body(snapshot: DSLRReplaySnapshot, *, include_snapshot_id: bool) -> bytes:
    metadata = _canonical_json_bytes(
        _snapshot_metadata(snapshot, include_snapshot_id=include_snapshot_id)
    )
    if len(metadata) > _MAX_METADATA_BYTES:
        raise ValueError("DSLR snapshot metadata exceeds its safe limit.")
    tensors = b"".join(
        _tensor_bytes(value)
        for value in (
            snapshot._feature,
            snapshot._embedding,
            snapshot._candidate_local_indices,
        )
    )
    return _SNAPSHOT_MAGIC + struct.pack(">Q", len(metadata)) + metadata + tensors


def _snapshot_digest(snapshot: DSLRReplaySnapshot) -> str:
    return hashlib.sha256(
        _snapshot_body(snapshot, include_snapshot_id=False)
    ).hexdigest()


def serialize_dslr_snapshot(snapshot: DSLRReplaySnapshot) -> bytes:
    """Return an exact non-executable payload charged to replay memory."""

    if not isinstance(snapshot, DSLRReplaySnapshot):
        raise TypeError("serialize_dslr_snapshot requires a DSLRReplaySnapshot.")
    return _snapshot_body(snapshot, include_snapshot_id=True)


def _payload_bytes(payload: bytes | bytearray | torch.Tensor) -> bytes:
    if torch.is_tensor(payload):
        if payload.dtype != torch.uint8 or payload.ndim != 1:
            raise ValueError("DSLR payload tensors must be one-dimensional uint8.")
        return payload.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    raise TypeError("DSLR payload must be bytes, bytearray, or a uint8 tensor.")


def _decode_tensor(
    payload: bytes,
    *,
    descriptor: Mapping[str, object],
    name: str,
) -> torch.Tensor:
    if set(descriptor) != {"dtype", "shape", "nbytes"}:
        raise ValueError(f"Malformed DSLR {name} tensor descriptor.")
    dtype_name = descriptor["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise ValueError(f"Unsupported DSLR {name} dtype.")
    shape = descriptor["shape"]
    if not isinstance(shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape
    ):
        raise ValueError(f"Malformed DSLR {name} tensor shape.")
    nbytes = descriptor["nbytes"]
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ValueError(f"Malformed DSLR {name} tensor byte count.")
    dtype = _DTYPES[dtype_name]
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes != expected or len(payload) != expected:
        raise ValueError(f"DSLR {name} tensor payload length mismatch.")
    if expected == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    return torch.frombuffer(bytearray(payload), dtype=dtype).clone().reshape(shape)


def deserialize_dslr_snapshot(
    payload: bytes | bytearray | torch.Tensor,
) -> DSLRReplaySnapshot:
    """Decode and checksum-validate one DSLR replay snapshot."""

    raw = _payload_bytes(payload)
    prefix = len(_SNAPSHOT_MAGIC) + 8
    if len(raw) < prefix or raw[: len(_SNAPSHOT_MAGIC)] != _SNAPSHOT_MAGIC:
        raise ValueError("Unknown or truncated DSLR snapshot payload.")
    metadata_size = struct.unpack(">Q", raw[len(_SNAPSHOT_MAGIC) : prefix])[0]
    if metadata_size > _MAX_METADATA_BYTES or prefix + metadata_size > len(raw):
        raise ValueError("Unsafe or truncated DSLR snapshot metadata.")
    metadata_bytes = raw[prefix : prefix + metadata_size]
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed DSLR snapshot metadata.") from error
    expected_fields = {
        "format",
        "byteorder",
        "client_id",
        "global_task_id",
        "stage_index",
        "class_id",
        "source_local_index",
        "snapshot_id",
        "tensors",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise ValueError("DSLR snapshot metadata fields do not match the schema.")
    if metadata_bytes != _canonical_json_bytes(metadata):
        raise ValueError("DSLR snapshot metadata is not canonically encoded.")
    if metadata["format"] != _SNAPSHOT_FORMAT:
        raise ValueError("Unsupported DSLR snapshot format.")
    if metadata["byteorder"] != sys.byteorder:
        raise ValueError("DSLR snapshot byte order differs from this host.")
    descriptors = metadata["tensors"]
    if not isinstance(descriptors, dict) or set(descriptors) != set(_TENSOR_ORDER):
        raise ValueError("Malformed DSLR snapshot tensor table.")
    offset = prefix + metadata_size
    decoded: Dict[str, torch.Tensor] = {}
    for name in _TENSOR_ORDER:
        descriptor = descriptors[name]
        if not isinstance(descriptor, dict):
            raise ValueError(f"Malformed DSLR {name} descriptor.")
        nbytes = descriptor.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError(f"Malformed DSLR {name} byte count.")
        end = offset + nbytes
        if end > len(raw):
            raise ValueError(f"Truncated DSLR {name} tensor payload.")
        decoded[name] = _decode_tensor(
            raw[offset:end], descriptor=descriptor, name=name
        )
        offset = end
    if offset != len(raw):
        raise ValueError("DSLR snapshot payload has trailing bytes.")
    snapshot_id = metadata["snapshot_id"]
    if (
        not isinstance(snapshot_id, str)
        or len(snapshot_id) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_id)
    ):
        raise ValueError("Malformed DSLR snapshot checksum.")
    snapshot = DSLRReplaySnapshot(
        client_id=metadata["client_id"],
        global_task_id=metadata["global_task_id"],
        stage_index=metadata["stage_index"],
        class_id=metadata["class_id"],
        source_local_index=metadata["source_local_index"],
        feature=decoded["feature"],
        embedding=decoded["embedding"],
        candidate_local_indices=decoded["candidate_local_indices"],
    )
    if snapshot.snapshot_id != snapshot_id:
        raise ValueError("DSLR snapshot checksum mismatch.")
    return snapshot


