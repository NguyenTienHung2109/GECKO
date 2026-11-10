from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gecko.config import GECKOConfig
from gecko.data.datasets.facebook_clean import DOMAINS, load_clean_facebook, remap_category
from gecko.data.partitioning.materialize import derive_strict_local_node_features
from gecko.data.streams.builder import stream_identity


def test_mapping_is_injective_reversible_and_handles_gaps():
    pairs = torch.tensor([[0, 7], [7, 20], [20, 0]])
    first, ids = remap_category(pairs, 0)
    second, _ = remap_category(pairs, len(ids))
    assert torch.equal(ids[first], pairs)
    assert set(first.flatten().tolist()).isdisjoint(second.flatten().tolist())
    assert len(ids) == 3


def test_loader_ignores_legacy_cache_and_metadata(tmp_path):
    folder = tmp_path / 'facebook_fixture/facebook_clean_data'
    folder.mkdir(parents=True)
    for domain in DOMAINS:
        (folder / f'{domain}_edges.csv').write_text('node_1,node_2\n0,7\n7,0\n7,20\n20,20\n')
    (folder.parent / 'dgl_graph.bin').write_bytes(b'legacy cache must not be loaded')
    (tmp_path / 'facebook_metadata_domainIL.pkl').write_bytes(b'ambiguous IDs must not be loaded')
    spec = load_clean_facebook(tmp_path)
    assert spec.node_features.shape == (24, 1)
    assert not bool(spec.node_features.any())
    pairs = spec.metadata['positive_pairs']
    assert len(pairs) == 24
    assert int((pairs[:, 0] == pairs[:, 1]).sum()) == 8
    categories = spec.metadata['node_source_category_ids']
    assert torch.equal(categories[pairs[:, 0]], categories[pairs[:, 1]])
    identities = torch.stack((categories, spec.metadata['node_source_local_ids']), dim=1)
    assert len(torch.unique(identities, dim=0)) == 24
    assert spec.metadata['legacy_endpoint_metadata_reused'] is False


def test_features_depend_only_on_owned_base_arcs():
    # 0--1 is internal; 1--2 is cross-client; 2--3 is remote.
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    spec = SimpleNamespace(problem_type='LP', node_features=torch.full((4, 1), 999.),
        edge_index=edges, metadata={'feature_provenance': 'strict_local_base_degree',
        'partition_base_policy': 'endpoint_hash_holdout',
        'positive_pairs': torch.tensor([[0, 3]]), 'positive_splits': torch.tensor([2])})
    owned = torch.tensor([0, 1])
    mask = torch.tensor([True, True, False, False, False, False])
    before = derive_strict_local_node_features(spec, owned, mask)
    assert torch.equal(before, torch.ones(2, 1))
    spec.metadata['positive_pairs'] = torch.tensor([[0, 2], [1, 3]])
    spec.node_features.fill_(-999.)
    spec.edge_index = edges[:, :4]  # Remove a remote edge, retaining the same local base.
    after = derive_strict_local_node_features(spec, owned, mask[:4])
    assert torch.equal(before, after)
    with pytest.raises(ValueError, match='nonlocal'):
        derive_strict_local_node_features(spec, owned, torch.ones(4, dtype=torch.bool))


def test_clean_contract_is_versioned_and_required_together():
    legacy = GECKOConfig.from_yaml('configs/gecko_v1/scenarios/lp_domain_facebook.yaml')
    clean = GECKOConfig.from_yaml('configs/gecko_v1/scenarios/lp_domain_facebook_clean_v2.yaml')
    changed = replace(legacy, scenario=replace(legacy.scenario,
        domain_constructor=clean.scenario.domain_constructor,
        feature_provenance=clean.scenario.feature_provenance))
    assert stream_identity(legacy) != stream_identity(changed)
    with pytest.raises(ValueError):
        replace(legacy, scenario=replace(legacy.scenario,
            feature_provenance=clean.scenario.feature_provenance)).validate()
