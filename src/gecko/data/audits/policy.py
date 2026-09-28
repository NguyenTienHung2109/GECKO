"""Versioned quality and release policy independent of stream identity."""

from __future__ import annotations

import hashlib
import json


AUDIT_POLICY_VERSION = 3
AUDIT_POLICY = {
    "minimum_group_size": 10,
    "catastrophic_group_coverage": 0.05,
    "low_group_coverage_warning": 0.20,
    "high_edge_cut_warning": 0.50,
    "high_boundary_node_warning": 0.90,
    "candidate_group_imbalance_warning": 10.0,
}
SEVERE_QUALITY_WARNING_PREFIXES = (
    "high_edge_cut_ratio",
    "high_boundary_node_ratio",
    "low_internal_query_coverage:",
)

RELEASE_TIER_OVERRIDES = {
    ("NC", "domain"): "stress",
    ("LC", "task"): "stress",
    ("LC", "class"): "stress",
    ("LC", "domain"): "stress",
}


def audit_policy_hash() -> str:
    payload = json.dumps(
        {"version": AUDIT_POLICY_VERSION, "policy": AUDIT_POLICY},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
