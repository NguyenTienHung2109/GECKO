from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple
import torch

SSM_RECORD_FORMAT = "uefa-ssm-record-v1"


SSM_STORE_FORMAT = "uefa-ssm-replay-store-v1"


SSM_STATE_FORMAT = "uefa-ssm-private-state-v2"


SSM_REPLAY_CEILING_BYTES = 16 * 1024 * 1024


SSM_STAGE_COUNT = 8


SSM_MAIN_HOP_BUDGETS = (10, 25)


SSM_NODE_ONLY_HOP_BUDGETS = (0, 0)


_RECORD_MAGIC = b"UEFA-SSM-RECORD-V1\0"


_MAX_RECORD_METADATA_BYTES = 64 * 1024


_TENSOR_ORDER = ("features", "edge_index", "source_local_nodes")


_FEATURE_DTYPES = {
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
        raise ValueError("SSM metadata is not canonical JSON-safe.") from error
    return rendered.encode("utf-8")


def _owned_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.is_quantized:
        raise ValueError(f"{name} must be a dense, strided tensor.")
    return value.detach().cpu().clone().contiguous()


def _validate_local_edge_index(
    edge_index: torch.Tensor, *, num_nodes: int, name: str = "edge_index"
) -> torch.Tensor:
    edges = _owned_tensor(edge_index, name=name)
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"{name} must have int64 shape [2, num_edges].")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError(f"{name} contains an endpoint outside the strict-local graph.")
    if edges.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long)
    arcs = sorted((int(source), int(target)) for source, target in edges.t().tolist())
    return torch.tensor(arcs, dtype=torch.long).t().contiguous()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    # A size-one trailing dimension may retain a non-unit stride even after
    # ``contiguous()`` because PyTorch treats that layout as contiguous.  A
    # flat view makes the byte contract unambiguous for every tensor shape.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_descriptor(value: torch.Tensor) -> Dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "nbytes": int(value.numel() * value.element_size()),
    }


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


@dataclass(frozen=True, init=False, eq=False)
class SSMRecord:
    """One owned compact computation graph and its single replay target."""

    client_id: int
    global_task_id: int
    stage_index: int
    class_id: int
    source_root_index: int
    root_index: int
    root_label: int
    sampler_mode: str
    hop_budgets: Tuple[int, ...]
    rng_algorithm: str
    rng_base_seed: int
    rng_record_seed: int
    sampled_node_count: int
    record_id: str
    _features: torch.Tensor
    _edge_index: torch.Tensor
    _source_local_nodes: torch.Tensor

    def __init__(
        self,
        *,
        client_id: int,
        global_task_id: int,
        stage_index: int,
        class_id: int,
        source_root_index: int,
        root_index: int,
        root_label: int,
        sampler_mode: str,
        hop_budgets: Sequence[int],
        rng_algorithm: str,
        rng_base_seed: int,
        rng_record_seed: int,
        sampled_node_count: int,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        source_local_nodes: torch.Tensor,
    ) -> None:
        identifiers = {
            "client_id": client_id,
            "global_task_id": global_task_id,
            "stage_index": stage_index,
            "class_id": class_id,
            "source_root_index": source_root_index,
            "root_index": root_index,
            "root_label": root_label,
            "rng_base_seed": rng_base_seed,
            "rng_record_seed": rng_record_seed,
            "sampled_node_count": sampled_node_count,
        }
        checked = {
            name: _validate_nonnegative_int(value, name=name)
            for name, value in identifiers.items()
        }
        if checked["class_id"] != checked["root_label"]:
            raise ValueError("SSM class_id must equal the single NC root label.")
        if sampler_mode not in {"uniform", "degree"}:
            raise ValueError("SSM sampler_mode must be 'uniform' or 'degree'.")
        budgets = tuple(
            _validate_nonnegative_int(value, name=f"hop_budgets[{index}]")
            for index, value in enumerate(hop_budgets)
        )
        if not budgets:
            raise ValueError("SSM requires at least one hop budget.")
        if not isinstance(rng_algorithm, str) or not rng_algorithm:
            raise ValueError("SSM rng_algorithm must be a non-empty string.")

        owned_features = _owned_tensor(features, name="features")
        if owned_features.dtype not in _FEATURE_DTYPES or owned_features.ndim != 2:
            raise ValueError(
                "SSM record features must be a two-dimensional floating tensor."
            )
        if owned_features.shape[0] == 0:
            raise ValueError("SSM record must contain its replay root.")
        owned_nodes = _owned_tensor(source_local_nodes, name="source_local_nodes")
        if owned_nodes.dtype != torch.long or owned_nodes.ndim != 1:
            raise ValueError("source_local_nodes must be one-dimensional int64.")
        if owned_nodes.shape[0] != owned_features.shape[0]:
            raise ValueError("SSM feature rows must align with source_local_nodes.")
        if owned_nodes.numel() and int(owned_nodes.min()) < 0:
            raise ValueError("source_local_nodes cannot contain negative indices.")
        if (
            len(set(int(value) for value in owned_nodes.tolist()))
            != owned_nodes.numel()
        ):
            raise ValueError("source_local_nodes must not contain duplicates.")
        if checked["root_index"] >= owned_nodes.numel():
            raise ValueError("SSM root_index is outside the compact record.")
        if int(owned_nodes[checked["root_index"]]) != checked["source_root_index"]:
            raise ValueError("SSM compact root does not map to source_root_index.")
        if checked["sampled_node_count"] != int(owned_nodes.numel() - 1):
            raise ValueError(
                "SSM sampled_node_count must equal the number of non-root nodes."
            )
        owned_edges = _validate_local_edge_index(
            edge_index,
            num_nodes=owned_features.shape[0],
            name="record edge_index",
        )

        for name, value in checked.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sampler_mode", sampler_mode)
        object.__setattr__(self, "hop_budgets", budgets)
        object.__setattr__(self, "rng_algorithm", rng_algorithm)
        object.__setattr__(self, "_features", owned_features)
        object.__setattr__(self, "_edge_index", owned_edges)
        object.__setattr__(self, "_source_local_nodes", owned_nodes)
        object.__setattr__(self, "record_id", _record_digest(self))

    @property
    def features(self) -> torch.Tensor:
        return self._features.clone()

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index.clone()

    @property
    def source_local_nodes(self) -> torch.Tensor:
        return self._source_local_nodes.clone()

    @property
    def serialized_bytes(self) -> int:
        return len(serialize_ssm_record(self))

    @property
    def context_node_count(self) -> int:
        return int(self._features.shape[0] - 1)

    @property
    def edge_count(self) -> int:
        return int(self._edge_index.shape[1])


def _record_metadata(
    record: SSMRecord, *, include_record_id: bool
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "format": SSM_RECORD_FORMAT,
        "byteorder": sys.byteorder,
        "client_id": record.client_id,
        "global_task_id": record.global_task_id,
        "stage_index": record.stage_index,
        "class_id": record.class_id,
        "source_root_index": record.source_root_index,
        "root_index": record.root_index,
        "root_label": record.root_label,
        "sampler_mode": record.sampler_mode,
        "hop_budgets": list(record.hop_budgets),
        "rng_algorithm": record.rng_algorithm,
        "rng_base_seed": record.rng_base_seed,
        "rng_record_seed": record.rng_record_seed,
        "sampled_node_count": record.sampled_node_count,
        "tensors": {
            "features": _tensor_descriptor(record._features),
            "edge_index": _tensor_descriptor(record._edge_index),
            "source_local_nodes": _tensor_descriptor(record._source_local_nodes),
        },
    }
    if include_record_id:
        metadata["record_id"] = record.record_id
    return metadata


def _record_body(record: SSMRecord, *, include_record_id: bool) -> bytes:
    metadata = _canonical_json_bytes(
        _record_metadata(record, include_record_id=include_record_id)
    )
    if len(metadata) > _MAX_RECORD_METADATA_BYTES:
        raise ValueError("SSM record metadata exceeds the safe size limit.")
    tensors = b"".join(
        _tensor_bytes(value)
        for value in (
            record._features,
            record._edge_index,
            record._source_local_nodes,
        )
    )
    return _RECORD_MAGIC + struct.pack(">Q", len(metadata)) + metadata + tensors


def _record_digest(record: SSMRecord) -> str:
    return hashlib.sha256(_record_body(record, include_record_id=False)).hexdigest()


def serialize_ssm_record(record: SSMRecord) -> bytes:
    """Return the exact non-executable binary payload charged to replay memory."""

    if not isinstance(record, SSMRecord):
        raise TypeError("serialize_ssm_record requires an SSMRecord.")
    return _record_body(record, include_record_id=True)


def _payload_bytes(payload: bytes | bytearray | torch.Tensor) -> bytes:
    if torch.is_tensor(payload):
        if payload.dtype != torch.uint8 or payload.ndim != 1:
            raise ValueError(
                "SSM checkpoint payload tensors must be one-dimensional uint8."
            )
        return payload.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    raise TypeError("SSM payload must be bytes, bytearray, or a uint8 tensor.")


def _tensor_from_payload(
    payload: bytes,
    *,
    descriptor: Mapping[str, object],
    name: str,
) -> torch.Tensor:
    if set(descriptor) != {"dtype", "shape", "nbytes"}:
        raise ValueError(f"Malformed SSM {name} tensor descriptor.")
    dtype_name = descriptor["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise ValueError(f"Unsupported SSM {name} dtype.")
    shape = descriptor["shape"]
    if not isinstance(shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape
    ):
        raise ValueError(f"Malformed SSM {name} tensor shape.")
    nbytes = descriptor["nbytes"]
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ValueError(f"Malformed SSM {name} tensor byte count.")
    dtype = _DTYPES[dtype_name]
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes != expected or len(payload) != expected:
        raise ValueError(f"SSM {name} tensor payload length mismatch.")
    if expected == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    return torch.frombuffer(bytearray(payload), dtype=dtype).clone().reshape(shape)


def deserialize_ssm_record(payload: bytes | bytearray | torch.Tensor) -> SSMRecord:
    """Decode and checksum-validate one safe SSM replay payload."""

    raw = _payload_bytes(payload)
    prefix = len(_RECORD_MAGIC) + 8
    if len(raw) < prefix or raw[: len(_RECORD_MAGIC)] != _RECORD_MAGIC:
        raise ValueError("Unknown or truncated SSM record payload.")
    metadata_size = struct.unpack(">Q", raw[len(_RECORD_MAGIC) : prefix])[0]
    if metadata_size > _MAX_RECORD_METADATA_BYTES or prefix + metadata_size > len(raw):
        raise ValueError("Unsafe or truncated SSM record metadata.")
    try:
        metadata = json.loads(raw[prefix : prefix + metadata_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed SSM record metadata.") from error
    expected_fields = {
        "format",
        "byteorder",
        "client_id",
        "global_task_id",
        "stage_index",
        "class_id",
        "source_root_index",
        "root_index",
        "root_label",
        "sampler_mode",
        "hop_budgets",
        "rng_algorithm",
        "rng_base_seed",
        "rng_record_seed",
        "sampled_node_count",
        "record_id",
        "tensors",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise ValueError("SSM record metadata fields do not match the schema.")
    if raw[prefix : prefix + metadata_size] != _canonical_json_bytes(metadata):
        raise ValueError("SSM record metadata is not canonically encoded.")
    if metadata["format"] != SSM_RECORD_FORMAT:
        raise ValueError("Unsupported SSM record format.")
    if metadata["byteorder"] != sys.byteorder:
        raise ValueError("SSM record byte order differs from this host.")
    descriptors = metadata["tensors"]
    if not isinstance(descriptors, dict) or set(descriptors) != set(_TENSOR_ORDER):
        raise ValueError("Malformed SSM record tensor table.")
    offset = prefix + metadata_size
    decoded: Dict[str, torch.Tensor] = {}
    for name in _TENSOR_ORDER:
        descriptor = descriptors[name]
        if not isinstance(descriptor, dict):
            raise ValueError(f"Malformed SSM {name} descriptor.")
        nbytes = descriptor.get("nbytes")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError(f"Malformed SSM {name} byte count.")
        end = offset + nbytes
        if end > len(raw):
            raise ValueError(f"Truncated SSM {name} tensor payload.")
        decoded[name] = _tensor_from_payload(
            raw[offset:end], descriptor=descriptor, name=name
        )
        offset = end
    if offset != len(raw):
        raise ValueError("SSM record payload has trailing bytes.")
    record_id = metadata["record_id"]
    if (
        not isinstance(record_id, str)
        or len(record_id) != 64
        or any(character not in "0123456789abcdef" for character in record_id)
    ):
        raise ValueError("Malformed SSM record checksum.")
    record = SSMRecord(
        client_id=metadata["client_id"],
        global_task_id=metadata["global_task_id"],
        stage_index=metadata["stage_index"],
        class_id=metadata["class_id"],
        source_root_index=metadata["source_root_index"],
        root_index=metadata["root_index"],
        root_label=metadata["root_label"],
        sampler_mode=metadata["sampler_mode"],
        hop_budgets=metadata["hop_budgets"],
        rng_algorithm=metadata["rng_algorithm"],
        rng_base_seed=metadata["rng_base_seed"],
        rng_record_seed=metadata["rng_record_seed"],
        sampled_node_count=metadata["sampled_node_count"],
        features=decoded["features"],
        edge_index=decoded["edge_index"],
        source_local_nodes=decoded["source_local_nodes"],
    )
    if record.record_id != record_id:
        raise ValueError("SSM record checksum mismatch.")
    return record


