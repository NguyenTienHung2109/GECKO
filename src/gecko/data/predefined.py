"""Portable, fixed stream assignments; raw NC graphs are resolved separately.

The public format is JSON and compressed NumPy arrays (never pickle). Import
checks every tensor, including reconstructed local graphs, before saving a stream.
No partitioner, split sampler, order generator, or participation sampler is called.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from gecko.reproducibility import repository_root

FORMAT = "gecko-predefined-v1"


def tensor_record(value: torch.Tensor) -> dict:
    array = value.detach().cpu().contiguous().numpy()
    descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
    digest.update(array.tobytes())
    return {**descriptor, "sha256": digest.hexdigest()}


def check_tensor(value: torch.Tensor, record: dict) -> torch.Tensor:
    if tensor_record(value) != record:
        raise ValueError(f"Tensor content mismatch: expected {record['sha256']}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".predefined-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload = gzip.compress(data, mtime=0) if path.suffix == ".gz" else data + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, expected_sha256: str | None = None) -> Any:
    data = path.read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError("Catalog path escapes the split directory")
    return path


class TensorStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def encode(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            record = tensor_record(value)
            if value.numel() <= 64:
                return {"inline": base64.b64encode(value.detach().cpu().contiguous().numpy().tobytes()).decode(), "tensor_record": record}
            path = self.root / "tensors" / (record["sha256"] + ".npy.gz")
            if not path.exists():
                stream = io.BytesIO()
                np.save(stream, value.detach().cpu().contiguous().numpy(), allow_pickle=False)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(path, gzip.compress(stream.getvalue(), compresslevel=3, mtime=0))
            return {"tensor": record}
        if isinstance(value, dict):
            if len(value) > 256 and all(type(k) is int and isinstance(v, torch.Tensor) and v.ndim == 1 for k, v in value.items()):
                dtypes = {v.dtype for v in value.values()}
                if len(dtypes) == 1:
                    return {"tensor_map": {"keys": self.encode(torch.tensor(list(value), dtype=torch.long)),
                        "lengths": self.encode(torch.tensor([v.numel() for v in value.values()], dtype=torch.long)),
                        "values": self.encode(torch.cat(list(value.values())))}}
            return {"dict": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, tuple):
            return {"tuple": [self.encode(v) for v in value]}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError(f"Unsupported public split value: {type(value).__name__}")

    def decode(self, value: Any, resolve: Callable | None = None) -> Any:
        if isinstance(value, list):
            return [self.decode(v, resolve) for v in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"inline", "tensor_record"}:
            record = value["tensor_record"]
            array = np.frombuffer(base64.b64decode(value["inline"], validate=True), dtype=record["dtype"]).reshape(record["shape"]).copy()
            return check_tensor(torch.from_numpy(array), record)
        if set(value) == {"tensor"}:
            record = value["tensor"]
            digest = record["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid tensor hash")
            data = gzip.decompress((self.root / "tensors" / (digest + ".npy.gz")).read_bytes())
            array = np.load(io.BytesIO(data), allow_pickle=False)
            return check_tensor(torch.from_numpy(array), record)
        if set(value) == {"external", "tensor_record"}:
            if resolve is None:
                raise ValueError("Raw graph is required to resolve this tensor")
            return check_tensor(resolve(value["external"]), value["tensor_record"])
        if set(value) == {"tensor_map"}:
            packed = {k: self.decode(v, resolve) for k, v in value["tensor_map"].items()}
            keys, lengths, values = packed["keys"], packed["lengths"], packed["values"]
            if len(keys) != len(lengths) or bool((lengths < 0).any()) or int(lengths.sum()) != len(values) or len(torch.unique(keys)) != len(keys):
                raise ValueError("Malformed packed tensor mapping")
            offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0))).tolist()
            return {key: values[offsets[i]:offsets[i+1]] for i, key in enumerate(keys.tolist())}
        if set(value) == {"dict"}:
            return {self.decode(k, resolve): self.decode(v, resolve) for k, v in value["dict"]}
        if set(value) == {"tuple"}:
            return tuple(self.decode(v, resolve) for v in value["tuple"])
        raise ValueError("Unrecognized split encoding")


def external(reference: list, value: torch.Tensor) -> dict:
    return {"external": reference, "tensor_record": tensor_record(value)}


def encoded_set(encoded: dict, path: list, value: Any) -> None:
    """Replace one field of an encoded mapping without serializing Python objects."""
    for key, child in encoded["dict"]:
        if key == path[0]:
            if len(path) == 1:
                for item in encoded["dict"]:
                    if item[0] == key:
                        item[1] = value
                        return
            return encoded_set(child, path[1:], value)
    raise KeyError(path)


def local_graph_tensors(scenario, nodes: torch.Tensor) -> dict:
    from gecko.data.partitioning.materialize import derive_strict_local_node_features
    mapping = torch.full((len(scenario.node_features),), -1, dtype=torch.long)
    mapping[nodes] = torch.arange(len(nodes))
    edges = scenario.edge_index
    internal = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
    features = (
        derive_strict_local_node_features(scenario, nodes, internal)
        if scenario.metadata.get("feature_provenance") in {"strict_local_derived", "strict_local_base_degree"}
        else scenario.node_features[nodes].clone()
    )
    return {"edge_index": mapping[edges[:, internal]], "node_features": features}


def encode_scenario(store: TensorStore, scenario: dict) -> dict:
    """Freeze small data and split tensors; reference upstream NC graph tensors."""
    small = dict(scenario)
    small["metadata"] = dict(small["metadata"])
    removed = []
    if small["problem_type"] == "NC":
        for name in ("edge_index", "node_features", "context_edge_index"):
            if isinstance(small.get(name), torch.Tensor):
                removed.append(([name], small[name]))
                small[name] = None
        for name in ("context_edge_features", "context_edge_feature_valid_mask"):
            if isinstance(small["metadata"].get(name), torch.Tensor):
                removed.append((["metadata", name], small["metadata"][name]))
                small["metadata"][name] = None
    encoded = store.encode(small)
    for path, tensor in removed:
        encoded_set(encoded, path, external(["raw", *path], tensor))
    return encoded


def encode_partition(store: TensorStore, partition: dict, scenario: dict) -> dict:
    """Verify the deterministic induced-graph reconstruction before exporting."""
    from gecko.types import ScenarioSpec
    small = dict(partition)
    small["client_graphs"] = {c: dict(g) for c, g in partition["client_graphs"].items()}
    removed = []
    if scenario["problem_type"] == "NC":
        spec = ScenarioSpec(**scenario)
        for cid, graph in small["client_graphs"].items():
            derived = local_graph_tensors(spec, graph["local_to_global"])
            for field in ("edge_index", "node_features"):
                original = graph[field]
                if not torch.equal(original, derived[field]):
                    raise ValueError(f"Induced graph does not reconstruct exactly: client {cid}, {field}")
                removed.append((["client_graphs", cid, field], original))
                graph[field] = None
    encoded = store.encode(small)
    for path, tensor in removed:
        encoded_set(encoded, path, external(["local", path[1], path[2]], tensor))
    return encoded


def read_catalog(root: Path) -> dict:
    catalog = read_json(root / "catalog.json")
    if catalog.get("format") != FORMAT:
        raise ValueError("Unsupported predefined split catalog")
    ids = [v["id"] for v in catalog["streams"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate split IDs")
    return catalog


def verify(root: Path) -> dict:
    """Verify every stored tensor and cell manifest; does not need raw datasets."""
    catalog = read_catalog(root)
    store = TensorStore(root)
    records = {}
    external_count = 0

    def walk(value):
        nonlocal external_count
        if isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, dict):
            if set(value) == {"tensor"}:
                record = value["tensor"]
                records[record["sha256"]] = record
            elif "inline" in value:
                store.decode(value)
            elif "external" in value:
                external_count += 1
            else:
                for child in value.values():
                    walk(child)

    for cell in catalog["streams"]:
        payload = read_json(safe_path(root, cell["file"]), cell["sha256"])
        if payload.get("format") != FORMAT or payload.get("id") != cell["id"]:
            raise ValueError("Catalog and cell identity disagree")
        walk(payload)
    for record in records.values():
        store.decode({"tensor": record})
    return {"streams": len(catalog["streams"]), "unique_tensors": len(records),
            "external_tensor_references": external_count,
            "status": "stored_payloads_verified", "raw_graphs_verified": False}


def load_predefined(root: Path, cell_id: str, data_root: Path, *, raw_loader=None):
    from gecko.config import GECKOConfig
    from gecko.data.dataloader import load_scenario_spec
    from gecko.data.streams.builder import StreamBundle, stream_identity
    from gecko.types import (ScenarioSpec, PartitionResult, ClientGraph, ClientTaskShard,
                             CentralEvaluationShard, OrderPlan, ParticipationPlan)

    catalog = read_catalog(root)
    matches = [v for v in catalog["streams"] if v["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown predefined split: {cell_id}")
    cell = matches[0]
    payload = read_json(safe_path(root, cell["file"]), cell["sha256"])
    store = TensorStore(root)
    config = GECKOConfig.from_mapping(payload["config"])
    config = replace(config, scenario=replace(config.scenario, save_path=str(data_root)),
                     benchmark_schema_version=2)
    raw = None

    def resolve_scenario(reference):
        nonlocal raw
        if reference[0] != "raw":
            raise ValueError("Unexpected raw graph reference")
        if raw is None:
            raw = (raw_loader or load_scenario_spec)(config)
        value = vars(raw)
        for key in reference[1:]:
            value = value[key]
        return value

    scenario = ScenarioSpec(**store.decode(payload["scenario"], resolve_scenario))
    scenario.validate()
    # Decode fixed local-to-global maps first; induced graph construction consumes
    # these saved maps, never a new partitioning result.
    encoded_partition = payload["partition"]
    nodes_encoded = next(v for k, v in encoded_partition["dict"] if k == "client_graphs")
    nodes_by_client = {}
    for cid, graph in nodes_encoded["dict"]:
        nodes_by_client[cid] = store.decode(next(v for k, v in graph["dict"] if k == "local_to_global"))
    local_cache = {}

    def resolve_local(reference):
        if reference[0] != "local":
            raise ValueError("Unexpected local graph reference")
        _, cid, field = reference
        if cid not in local_cache:
            # Release the previous graph; a protein split can have large edges.
            local_cache.clear()
            local_cache[cid] = local_graph_tensors(scenario, nodes_by_client[cid])
        return local_cache[cid][field]

    partition = store.decode(encoded_partition, resolve_local)
    client_graphs = {}
    for cid, fields in partition["client_graphs"].items():
        fields["global_to_local"] = {int(g): i for i, g in enumerate(fields["local_to_global"].tolist())}
        client_graphs[cid] = ClientGraph(**fields)
    partition["client_graphs"] = client_graphs
    shards = {c: {t: ClientTaskShard(**v) for t, v in tasks.items()}
              for c, tasks in store.decode(payload["shards"]).items()}
    evaluation = {c: {t: CentralEvaluationShard(**v) for t, v in tasks.items()}
                  for c, tasks in store.decode(payload["evaluation"]).items()}
    orders = OrderPlan(**store.decode(payload["orders"]))
    participation = ParticipationPlan(**store.decode(payload["participation"]))
    stream_id, digest = stream_identity(config)
    return StreamBundle(config, scenario, PartitionResult(**partition), shards, evaluation,
                        orders, participation, stream_id, digest, {})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "verify", "materialize"))
    parser.add_argument("--catalog", type=Path, default=repository_root() / "reproduce/splits")
    parser.add_argument("--id", dest="cell_id")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("generated_streams/predefined"))
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(read_catalog(args.catalog), indent=2))
    elif args.command == "verify":
        print(json.dumps(verify(args.catalog), indent=2))
    else:
        if not args.cell_id:
            parser.error("materialize requires --id")
        bundle = load_predefined(args.catalog, args.cell_id, args.data_root)
        from gecko.data.streams.store import save_stream
        # Each imported cell has its own store: corrected and legacy datasets
        # never overwrite one another even when their display paths coincide.
        path = save_stream(bundle, args.output / args.cell_id)
        print(json.dumps({"stream": str(path), "split_id": args.cell_id,
                          "tensor_parity": "verified", "eligibility": "see audit.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
