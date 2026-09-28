from __future__ import annotations

import math

import pytest
import torch

from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.partitioning.structural_diagnostics import DegreeRoleDefinition
from gecko.data.partitioning.structural_diagnostics import build_degree_roles
from gecko.data.partitioning.structural_diagnostics import evaluate_hstr
from gecko.data.partitioning.structural_diagnostics import hstr_pair_swap_delta


def _topology():
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]
    return build_weighted_logical_topology(
        torch.tensor(edges, dtype=torch.long).T,
        num_nodes=6,
        directed=False,
        representation_id="hstr-test",
    )


def _manual_roles():
    return DegreeRoleDefinition(
        num_bins=2,
        bin_boundaries=torch.tensor([1.0]),
        role_ids=torch.tensor([0, 0, 1, 1]),
        global_degree=torch.tensor([1.0, 1.0, 2.0, 2.0]),
        global_role_histogram=torch.tensor([0.5, 0.5], dtype=torch.float64),
        quantile_policy="manual",
        topology_hash="manual",
    )


def test_hstr_manual_degree_histograms():
    metrics = evaluate_hstr(torch.tensor([0, 0, 1, 1]), _manual_roles(), num_clients=2)
    expected = 0.5 * math.log2(4 / 3) + 0.25 * math.log2(2 / 3) + 0.25
    assert metrics.h_str == pytest.approx(expected)


def test_hstr_zero_when_clients_match_global_histogram():
    assert evaluate_hstr(torch.tensor([0, 1, 0, 1]), _manual_roles(), num_clients=2).h_str == 0


def test_hstr_positive_when_clients_degree_specialize():
    assert evaluate_hstr(torch.tensor([0, 0, 1, 1]), _manual_roles(), num_clients=2).h_str > 0


def test_hstr_uses_global_prepartition_degree():
    roles = build_degree_roles(_topology(), num_bins=3)
    assert roles.global_degree.tolist() == [4, 2, 2, 2, 2, 0]


def test_hstr_does_not_use_local_induced_degree():
    roles = build_degree_roles(_topology(), num_bins=3)
    before = roles.role_ids.clone()
    evaluate_hstr(torch.tensor([0, 0, 1, 1, 0, 1]), roles, num_clients=2)
    evaluate_hstr(torch.tensor([1, 0, 0, 1, 1, 0]), roles, num_clients=2)
    assert torch.equal(roles.role_ids, before)


def test_hstr_deterministic_quantile_ties():
    first = build_degree_roles(_topology(), num_bins=3)
    second = build_degree_roles(_topology(), num_bins=3)
    assert torch.equal(first.role_ids, second.role_ids)
    tied = first.global_degree == 2
    assert torch.unique(first.role_ids[tied]).numel() == 1


def test_hstr_base2_bounded_zero_one():
    value = evaluate_hstr(torch.tensor([0, 0, 1, 1]), _manual_roles(), num_clients=2).h_str
    assert 0 <= value <= 1


def test_hstr_natural_log_normalization_equivalent():
    owner = torch.tensor([0, 0, 1, 1])
    base2 = evaluate_hstr(owner, _manual_roles(), num_clients=2, js_log_policy="base2")
    natural = evaluate_hstr(owner, _manual_roles(), num_clients=2, js_log_policy="natural_normalized")
    assert base2.h_str == natural.h_str


def test_hstr_no_core_number_dependency(monkeypatch):
    import networkx as nx
    monkeypatch.setattr(nx, "core_number", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("core used")))
    assert build_degree_roles(_topology(), num_bins=3).role_ids.numel() == 6


def test_hstr_incremental_pair_swap_matches_recompute():
    roles = build_degree_roles(_topology(), num_bins=3)
    owner = torch.tensor([0, 0, 1, 1, 0, 1])
    before = evaluate_hstr(owner, roles, num_clients=2).h_str
    delta = hstr_pair_swap_delta(owner, roles, num_clients=2, first=0, second=3)
    after_owner = owner.clone(); after_owner[0], after_owner[3] = owner[3], owner[0]
    after = evaluate_hstr(after_owner, roles, num_clients=2).h_str
    assert before + delta == pytest.approx(after)
