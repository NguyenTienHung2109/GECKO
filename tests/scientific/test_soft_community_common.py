from __future__ import annotations

import torch

from gecko.data.partitioning.community import SoftMicroCommunityConfig
from gecko.data.partitioning.community import build_microcommunities
from gecko.data.partitioning.community.common import build_large_metis_microcommunities
from gecko.data.partitioning.base import build_large_bidirected_simple_topology
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.partitioning.base import evaluate_hconn


def _topology(edges: torch.Tensor):
    return build_weighted_logical_topology(
        edges,
        num_nodes=6,
        directed=False,
        representation_id="soft-common-test",
    )


def test_soft_microcommunities_are_canonical_under_edge_storage_order() -> None:
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]],
        dtype=torch.long,
    ).T.contiguous()
    reverse_order = edges.flip(1).flip(0).contiguous()
    config = SoftMicroCommunityConfig(
        seed=9,
        maximum_microcommunity_nodes=10,
    )
    first = build_microcommunities(_topology(edges), config)
    second = build_microcommunities(_topology(reverse_order), config)
    assert torch.equal(first.micro_ids, second.micro_ids)
    assert first.diagnostics["microcommunity_hash"] == second.diagnostics[
        "microcommunity_hash"
    ]
    assert first.diagnostics["label_access_policy"] == "topology_only"


def test_soft_microcommunities_handle_an_edgeless_topology() -> None:
    topology = _topology(torch.empty((2, 0), dtype=torch.long))
    result = build_microcommunities(
        topology,
        SoftMicroCommunityConfig(seed=4, maximum_microcommunity_nodes=2),
    )
    assert torch.equal(result.micro_ids, torch.arange(6))
    assert result.diagnostics["microcommunity_count"] == 6
    assert result.diagnostics["label_access_policy"] == "topology_only"


def test_compact_bidirected_topology_matches_canonical_unweighted_metrics() -> None:
    logical = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
    )
    bidirected = torch.cat(
        (logical, logical.flip(0), torch.arange(4).repeat(2, 1)), dim=1
    )
    compact = build_large_bidirected_simple_topology(
        bidirected,
        num_nodes=4,
        representation_id="compact-test",
        device=torch.device("cpu"),
    )
    canonical = build_weighted_logical_topology(
        bidirected,
        num_nodes=4,
        directed=False,
        representation_id="canonical-test",
        backend="tensor",
    )
    owner = torch.tensor([0, 0, 1, 1])

    assert torch.equal(compact.edge_index, canonical.edge_index)
    assert torch.equal(compact.edge_weights, canonical.edge_weights)
    assert [row.tolist() for row in compact.incident_neighbors] == [
        row.tolist() for row in canonical.incident_neighbors
    ]
    first = evaluate_hconn(owner, compact, num_clients=2)
    second = evaluate_hconn(owner, canonical, num_clients=2)
    assert first == second


def test_large_metis_backend_consumes_compact_csr_and_respects_size_bound() -> None:
    source = torch.arange(12)
    logical = torch.stack((source, (source + 1) % 12))
    bidirected = torch.cat((logical, logical.flip(0)), dim=1)
    topology = build_large_bidirected_simple_topology(
        bidirected,
        num_nodes=12,
        representation_id="large-metis-test",
        device=torch.device("cpu"),
    )
    result = build_large_metis_microcommunities(
        topology,
        SoftMicroCommunityConfig(seed=5, maximum_microcommunity_nodes=4),
    )

    assert result.diagnostics["microcommunity_backend"] == (
        "pymetis_direct_csr_large_graph"
    )
    assert result.diagnostics["largest_microcommunity_size"] <= 4
    assert torch.equal(torch.unique(result.micro_ids), torch.arange(6))
