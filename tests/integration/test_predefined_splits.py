"""Fixed-split import preserves tensors and cannot silently substitute new data."""
import copy
from dataclasses import asdict, replace
import gzip
import json

import pytest
import torch

from gecko.data.predefined import (FORMAT, TensorStore, encode_scenario, encode_partition,
    load_predefined, verify, write_json, read_json, safe_path)
from gecko.data.streams.store import _partition_payload, _shard_payload, _evaluation_payload
from tests.helpers import CASES, make_stream, assert_tensor_map_equal


def export_test_bundle(root, bundle):
    store = TensorStore(root)
    payload = {
        'format': FORMAT, 'id': 'test', 'config': bundle.config.to_dict(),
        'scenario': encode_scenario(store, vars(bundle.scenario)),
        'partition': encode_partition(store, _partition_payload(bundle.partition), vars(bundle.scenario)),
        'shards': store.encode(_shard_payload(bundle.shards)),
        'evaluation': store.encode(_evaluation_payload(bundle)['client_shards']),
        'orders': store.encode(asdict(bundle.orders)),
        'participation': store.encode(asdict(bundle.participation)),
    }
    sha = write_json(root / 'cells/test.json.gz', payload)
    write_json(root / 'catalog.json', {'format': FORMAT, 'streams': [
        {'id': 'test', 'file': 'cells/test.json.gz', 'sha256': sha}]})
    return payload


@pytest.mark.parametrize('problem,incremental,tasks', CASES)
def test_every_scenario_roundtrips_without_resampling(tmp_path, monkeypatch, problem, incremental, tasks):
    source = make_stream(problem, incremental, tasks)
    export_test_bundle(tmp_path, source)
    assert verify(tmp_path)['streams'] == 1
    def forbidden(*args, **kwargs):
        raise AssertionError('A fixed split must not sample a new stream')
    monkeypatch.setattr('gecko.data.streams.builder.StreamBuilder.build', forbidden)
    restored = load_predefined(tmp_path, 'test', tmp_path / 'data', raw_loader=lambda _: source.scenario)
    assert_tensor_map_equal(vars(source.scenario), vars(restored.scenario))
    assert_tensor_map_equal(_partition_payload(source.partition), _partition_payload(restored.partition))
    assert_tensor_map_equal(_shard_payload(source.shards), _shard_payload(restored.shards))
    assert_tensor_map_equal(_evaluation_payload(source), _evaluation_payload(restored))
    assert source.orders == restored.orders
    assert source.participation == restored.participation
    # Also check the normal public artifact loader can consume the imported bundle.
    from gecko.data.streams.store import save_stream, load_stream
    path = save_stream(restored, tmp_path / 'restored')
    reread = load_stream(path)
    assert_tensor_map_equal(_shard_payload(source.shards), _shard_payload(reread.shards))


def test_changed_upstream_graph_is_rejected(tmp_path):
    source = make_stream('NC', 'class', 2)
    export_test_bundle(tmp_path, source)
    changed = replace(source.scenario, node_features=source.scenario.node_features + 1)
    with pytest.raises(ValueError, match='Tensor content mismatch'):
        load_predefined(tmp_path, 'test', tmp_path / 'data', raw_loader=lambda _: changed)


def test_corrupted_tensor_is_rejected(tmp_path):
    source = make_stream('LC', 'class', 2)
    export_test_bundle(tmp_path, source)
    path = next((tmp_path / 'tensors').glob('*.npy.gz'))
    raw = bytearray(gzip.decompress(path.read_bytes()))
    raw[-1] ^= 1
    path.write_bytes(gzip.compress(raw, mtime=0))
    with pytest.raises(ValueError, match='Tensor content mismatch'):
        verify(tmp_path)


def test_changed_cell_and_path_traversal_rejected(tmp_path):
    source = make_stream('LP', 'domain', 2)
    export_test_bundle(tmp_path, source)
    path = tmp_path / 'cells/test.json.gz'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        load_predefined(tmp_path, 'test', tmp_path / 'data')
    with pytest.raises(ValueError, match='escapes'):
        safe_path(tmp_path, '../secret')


def test_large_tensor_mapping_preserves_keys_lengths_and_dtype(tmp_path):
    store = TensorStore(tmp_path)
    original = {2 * i: torch.arange(i % 4, dtype=torch.int64) for i in range(300)}
    encoded = store.encode(original)
    assert 'tensor_map' in encoded
    assert_tensor_map_equal(original, store.decode(encoded))
    assert len(list((tmp_path / 'tensors').glob('*'))) <= 3
