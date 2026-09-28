"""Exact-quota direct assignment of training nodes without communities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import torch

from gecko.reproducibility import torch_generator
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuota


DIRECT_TRAIN_ASSIGNMENT_VERSION = "direct_train_node_exact_quota_v1"


@dataclass(frozen=True)
class DirectTrainAssignment:
    train_node_ids: torch.Tensor
    train_owners: torch.Tensor
    owner_by_global_node: torch.Tensor
    realized_train_class_matrix: torch.Tensor
    realized_train_task_matrix: torch.Tensor
    quota_hash: str
    owner_hash: str
    exact_quota_verified: bool
    assignment_mode: str
    seed: int
    policy_version: str = DIRECT_TRAIN_ASSIGNMENT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "assignment_mode": self.assignment_mode,
            "seed": self.seed,
            "train_node_ids": self.train_node_ids.tolist(),
            "train_owners": self.train_owners.tolist(),
            "realized_train_class_matrix": self.realized_train_class_matrix.tolist(),
            "realized_train_task_matrix": self.realized_train_task_matrix.tolist(),
            "quota_hash": self.quota_hash,
            "owner_hash": self.owner_hash,
            "exact_quota_verified": self.exact_quota_verified,
        }


def _owner_hash(
    train_node_ids: torch.Tensor,
    train_owners: torch.Tensor,
    *,
    num_nodes: int,
) -> str:
    payload = {
        "version": DIRECT_TRAIN_ASSIGNMENT_VERSION,
        "num_nodes": num_nodes,
        "train_node_ids": train_node_ids.tolist(),
        "train_owners": train_owners.tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def direct_assign_train_nodes(
    train_node_ids: torch.Tensor,
    train_labels: torch.Tensor,
    quota: ExactDirichletQuota,
    *,
    num_nodes: int,
    seed: int,
    assignment_mode: Literal["quota_random", "quota_degree_ordered"] = "quota_random",
    global_degree: torch.Tensor | None = None,
) -> DirectTrainAssignment:
    """Assign each supplied training node exactly once according to the quota."""

    ids = train_node_ids.detach().cpu().long().flatten()
    labels = train_labels.detach().cpu().long().flatten()
    if ids.shape != labels.shape or ids.numel() != int(quota.integer_quota.sum()):
        raise ValueError("Training IDs/labels must align with total quota mass.")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= num_nodes):
        raise ValueError("A training node ID is outside the global node range.")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("Training node IDs must be unique.")
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= quota.integer_quota.shape[1]
    ):
        raise ValueError("A training label is outside the quota class range.")
    if not torch.equal(
        torch.bincount(labels, minlength=quota.integer_quota.shape[1]),
        quota.train_class_counts,
    ):
        raise ValueError("Training labels do not match quota class margins.")
    if assignment_mode not in {"quota_random", "quota_degree_ordered"}:
        raise ValueError("Unknown direct train assignment mode.")
    if assignment_mode == "quota_degree_ordered":
        if global_degree is None or global_degree.shape != (num_nodes,):
            raise ValueError("quota_degree_ordered requires global_degree for every node.")
        degree = global_degree.detach().cpu().long()

    owner_by_global = torch.full((num_nodes,), -1, dtype=torch.long)
    for class_id in range(quota.integer_quota.shape[1]):
        class_nodes = ids[labels == class_id]
        if assignment_mode == "quota_random":
            generator = torch_generator(
                seed, DIRECT_TRAIN_ASSIGNMENT_VERSION, "class", class_id
            )
            class_nodes = class_nodes[
                torch.randperm(class_nodes.numel(), generator=generator)
            ]
        else:
            class_nodes = torch.tensor(
                sorted(
                    class_nodes.tolist(),
                    key=lambda node: (-int(degree[node]), int(node)),
                ),
                dtype=torch.long,
            )
        offset = 0
        for client in range(quota.integer_quota.shape[0]):
            count = int(quota.integer_quota[client, class_id])
            selected = class_nodes[offset : offset + count]
            owner_by_global[selected] = client
            offset += count
        if offset != class_nodes.numel():
            raise AssertionError("Class quota did not consume every class node.")

    train_owners = owner_by_global[ids]
    if bool((train_owners < 0).any()):
        raise AssertionError("Direct assignment left a training node unowned.")
    num_clients, num_classes = quota.integer_quota.shape
    realized = torch.bincount(
        train_owners * num_classes + labels,
        minlength=num_clients * num_classes,
    ).reshape(num_clients, num_classes)
    task = torch.stack(
        [realized[:, torch.tensor(group)].sum(1) for group in quota.task_class_groups],
        dim=1,
    )
    verified = torch.equal(realized, quota.integer_quota)
    if not verified:
        raise AssertionError("Direct assignment failed its exact quota.")
    return DirectTrainAssignment(
        train_node_ids=ids.clone(),
        train_owners=train_owners.clone(),
        owner_by_global_node=owner_by_global,
        realized_train_class_matrix=realized,
        realized_train_task_matrix=task,
        quota_hash=quota.quota_hash,
        owner_hash=_owner_hash(ids, train_owners, num_nodes=num_nodes),
        exact_quota_verified=True,
        assignment_mode=assignment_mode,
        seed=int(seed),
    )
