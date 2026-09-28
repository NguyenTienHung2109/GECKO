from __future__ import annotations

import torch

from gecko.data.partitioning.community.micro_communities import generate_micro_communities


def test_small_graph_metis_micro_communities_are_deterministic_and_complete():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 3],
            [1, 2, 3, 4, 5, 6, 7, 0, 4, 7],
        ],
        dtype=torch.long,
    )

    first = generate_micro_communities(
        edge_index, num_nodes=8, target_partitions=4, seed=7
    )
    second = generate_micro_communities(
        edge_index, num_nodes=8, target_partitions=4, seed=7
    )

    assert torch.equal(first, second)
    assert first.shape == (8,)
    assert int(first.min()) == 0
    assert int(first.max()) < 4
