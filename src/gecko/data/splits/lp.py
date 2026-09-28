from __future__ import annotations

import hashlib
from typing import Dict
import torch
from gecko.reproducibility import torch_generator
from gecko.types import ScenarioSpec

BASE_EDGE_SELECTOR = {
    "algorithm": "blake2b",
    "version": 2,
    "personalization": "UEFA-LP2BASE",
    "digest_bits": 64,
    "seed_encoding": "signed-big-endian-int64",
    "endpoint_encoding": "unsigned-big-endian-uint64",
    "canonicalization": "min_endpoint,max_endpoint for undirected graphs",
}


def stable_base_edge_mask(
    pairs: torch.Tensor,
    *,
    seed: int,
    ratio: float,
    undirected: bool,
    edge_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select base edges with a versioned cross-process BLAKE2b digest."""

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_edges, 2].")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1].")
    if edge_types is not None and edge_types.shape[0] != pairs.shape[0]:
        raise ValueError("edge_types must align with pairs.")
    threshold = int(ratio * (1 << BASE_EDGE_SELECTOR["digest_bits"]))
    selected = []
    for index, (raw_source, raw_target) in enumerate(pairs.long().tolist()):
        source, target = int(raw_source), int(raw_target)
        if source < 0 or target < 0:
            raise ValueError("Node IDs must be non-negative.")
        if undirected and source > target:
            source, target = target, source
        payload = (
            int(seed).to_bytes(8, "big", signed=True)
            + source.to_bytes(8, "big", signed=False)
            + target.to_bytes(8, "big", signed=False)
        )
        if edge_types is not None:
            payload += int(edge_types[index]).to_bytes(8, "big", signed=True)
        digest = hashlib.blake2b(
            payload,
            digest_size=BASE_EDGE_SELECTOR["digest_bits"] // 8,
            person=BASE_EDGE_SELECTOR["personalization"].encode("ascii"),
        ).digest()
        selected.append(int.from_bytes(digest, "big") < threshold)
    return torch.tensor(selected, dtype=torch.bool)


def _symmetric_arcs(pairs: torch.Tensor) -> torch.Tensor:
    arcs = pairs.long().t().contiguous()
    if arcs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    non_loops = arcs[0] != arcs[1]
    return torch.cat((arcs, arcs.flip(0)[:, non_loops]), dim=1)


def _set_difference_pairs(
    known_pairs: torch.Tensor,
    query_pairs: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if query_pairs.numel() == 0:
        return known_pairs.clone()
    query_keys = torch.sort(query_pairs[:, 0] * num_nodes + query_pairs[:, 1]).values
    known_keys = known_pairs[:, 0] * num_nodes + known_pairs[:, 1]
    positions = torch.searchsorted(query_keys, known_keys)
    bounded = positions.clamp_max(query_keys.numel() - 1)
    is_query = (positions < query_keys.numel()) & (
        query_keys[bounded] == known_keys
    )
    return known_pairs[~is_query]


def prepare_partition_first_lp_pool(
    spec: ScenarioSpec,
    *,
    target_num_tasks: int,
    source_num_tasks: int,
    base_edge_ratio: float,
    seed: int,
    domain_mapping: str = "contiguous_merge",
) -> ScenarioSpec:
    """Create a query-blind partition graph and an unsplit positive pool."""
    from gecko.data.scenario import build_lp_spec

    positive_pairs = spec.metadata["positive_pairs"].long()
    positive_tasks = spec.metadata["positive_task_ids"].long()
    known_pairs = spec.metadata["known_positive_pairs"].long()
    num_nodes = spec.node_features.shape[0]
    existing_base = _set_difference_pairs(known_pairs, positive_pairs, num_nodes)
    undirected = bool(spec.metadata.get("undirected", True))
    hashed_base = stable_base_edge_mask(
        positive_pairs,
        seed=seed,
        ratio=base_edge_ratio,
        undirected=undirected,
    )
    base_pairs = torch.cat((existing_base, positive_pairs[hashed_base]), dim=0)
    query_pool = positive_pairs[~hashed_base]
    source_tasks = positive_tasks[~hashed_base]
    source_counts = torch.bincount(positive_tasks, minlength=source_num_tasks)
    if domain_mapping == "contiguous_merge":
        mapped_tasks = torch.div(
            source_tasks * target_num_tasks,
            source_num_tasks,
            rounding_mode="floor",
        ).clamp_max(target_num_tasks - 1)
        mapping = {
            source: min(
                target_num_tasks - 1,
                (source * target_num_tasks) // source_num_tasks,
            )
            for source in range(source_num_tasks)
        }
    elif domain_mapping == "dominant_source_vs_rest_v1":
        if target_num_tasks != 2:
            raise ValueError("dominant_source_vs_rest_v1 requires two target tasks.")
        dominant_source = int(torch.argmax(source_counts))
        mapped_tasks = (source_tasks != dominant_source).long()
        mapping = {
            source: int(source != dominant_source)
            for source in range(source_num_tasks)
        }
    else:
        raise ValueError(f"Unknown LP source-domain mapping: {domain_mapping!r}.")
    empty_negatives = {
        task: {
            split: torch.empty((0, 2), dtype=torch.long)
            for split in ("train", "val", "test")
        }
        for task in range(target_num_tasks)
    }
    metadata = {
        **spec.metadata,
        "partition_first_pending": True,
        "partition_first_protocol": "partition_first_query_split_v1",
        "partition_base_policy": "endpoint_hash_holdout",
        "partition_base_selector": dict(BASE_EDGE_SELECTOR),
        "partition_base_edge_ratio": base_edge_ratio,
        "partition_base_positive_pairs": base_pairs.clone(),
        "partition_query_positive_pool_pairs": query_pool.clone(),
        "partition_query_positive_pool_count": int(query_pool.shape[0]),
        "partition_original_known_positive_count": int(known_pairs.shape[0]),
        "partition_source_num_domains": source_num_tasks,
        "partition_target_num_domains": target_num_tasks,
        "partition_domain_mapping": domain_mapping,
        "partition_source_to_target_mapping": mapping,
        "partition_source_positive_counts": source_counts,
    }
    return build_lp_spec(
        dataset_name=spec.dataset_name,
        metrics=spec.metrics,
        node_features=spec.node_features,
        positive_pairs=query_pool,
        positive_task_ids=mapped_tasks,
        positive_splits=torch.zeros(query_pool.shape[0], dtype=torch.long),
        negative_pairs_by_task_split=empty_negatives,
        num_tasks=target_num_tasks,
        undirected=undirected,
        known_positive_pairs=known_pairs,
        context_edge_index=_symmetric_arcs(base_pairs),
        bipartite=spec.bipartite,
        node_types=spec.node_types,
        metadata=metadata,
    )


def _sample_owned_negative_pairs(
    *,
    node_ids: torch.Tensor,
    num_nodes: int,
    positive_keys: set[int],
    used_keys: set[int],
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if node_ids.numel() < 2:
        raise ValueError("An LP client needs at least two nodes for negative sampling.")
    chosen: list[int] = []
    attempts = 0
    while len(chosen) < count and attempts < 1000:
        remaining = count - len(chosen)
        batch = min(1_000_000, max(4096, remaining * 3))
        indices = torch.randint(
            node_ids.numel(), (batch, 2), generator=generator
        )
        pairs = node_ids[indices]
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.numel() == 0:
            attempts += 1
            continue
        left = torch.minimum(pairs[:, 0], pairs[:, 1])
        right = torch.maximum(pairs[:, 0], pairs[:, 1])
        keys = torch.unique(left * num_nodes + right, sorted=True)
        for key in keys.tolist():
            if key in positive_keys or key in used_keys:
                continue
            used_keys.add(key)
            chosen.append(key)
            if len(chosen) == count:
                break
        attempts += 1
    if len(chosen) != count:
        raise ValueError(
            f"Unable to sample {count} owned LP negatives after {attempts} batches."
        )
    keys = torch.tensor(chosen, dtype=torch.long)
    return torch.stack((keys.div(num_nodes, rounding_mode="floor"), keys % num_nodes), dim=1)


def finalize_partition_first_lp(
    spec: ScenarioSpec,
    node_owner: torch.Tensor,
    *,
    num_clients: int,
    seed: int,
    minimum_support: tuple[int, int, int],
    training_negative_ratio: float,
    evaluation_negatives_per_client_task: int,
) -> ScenarioSpec:
    """Split internal positives after ownership and build fixed local candidates."""
    from gecko.data.scenario import build_lp_spec

    if not spec.metadata.get("partition_first_pending", False):
        return spec
    from gecko.reproducibility import torch_generator

    positive_pool = spec.metadata["positive_pairs"].long()
    task_ids = spec.metadata["positive_task_ids"].long()
    known_pairs = spec.metadata["known_positive_pairs"].long()
    source_owner = node_owner[positive_pool[:, 0]]
    target_owner = node_owner[positive_pool[:, 1]]
    internal = source_owner == target_owner
    split_names = ("train", "val", "test")
    positive_parts = {
        task: {split: [] for split in split_names}
        for task in range(spec.num_tasks)
    }
    positive_groups = {
        task: {split: [] for split in split_names}
        for task in range(spec.num_tasks)
    }
    split_counts: dict[str, int] = {split: 0 for split in split_names}
    failures = []
    for client in range(num_clients):
        for task in range(spec.num_tasks):
            ids = torch.nonzero(
                internal & (source_owner == client) & (task_ids == task),
                as_tuple=True,
            )[0]
            required_total = sum(minimum_support)
            if ids.numel() < required_total:
                failures.append(
                    f"client={client} task={task} internal_positives={ids.numel()} "
                    f"required={required_total}"
                )
                continue
            order = ids[
                torch.randperm(
                    ids.numel(),
                    generator=torch_generator(seed, "lp-partition-first-split", client, task),
                )
            ]
            validation_count = max(minimum_support[1], round(ids.numel() * 0.10))
            test_count = max(minimum_support[2], round(ids.numel() * 0.10))
            if ids.numel() - validation_count - test_count < minimum_support[0]:
                validation_count = minimum_support[1]
                test_count = minimum_support[2]
            train_count = ids.numel() - validation_count - test_count
            ranges = {
                "train": order[:train_count],
                "val": order[train_count : train_count + validation_count],
                "test": order[train_count + validation_count :],
            }
            for split_index, split in enumerate(split_names):
                pairs = positive_pool[ranges[split]]
                positive_parts[task][split].append(pairs)
                group_id = ((client * spec.num_tasks + task) * 3) + split_index
                positive_groups[task][split].append(
                    torch.full((pairs.shape[0],), group_id, dtype=torch.long)
                )
                split_counts[split] += int(pairs.shape[0])
    if failures:
        raise ValueError(
            "Partition-first LP support is infeasible:\n- " + "\n- ".join(failures)
        )

    positive_by_task_split: Dict[int, Dict[str, torch.Tensor]] = {}
    positive_group_by_task_split: Dict[int, Dict[str, torch.Tensor]] = {}
    for task in range(spec.num_tasks):
        positive_by_task_split[task] = {}
        positive_group_by_task_split[task] = {}
        for split in split_names:
            positive_by_task_split[task][split] = torch.cat(
                positive_parts[task][split], dim=0
            )
            positive_group_by_task_split[task][split] = torch.cat(
                positive_groups[task][split], dim=0
            )

    num_nodes = spec.node_features.shape[0]
    positive_keys = {
        int(source) * num_nodes + int(target)
        for source, target in known_pairs.tolist()
    }
    used_negative_keys: set[int] = set()
    negatives: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(spec.num_tasks)
    }
    candidate_groups: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(spec.num_tasks)
    }
    for task in range(spec.num_tasks):
        for split_index, split in enumerate(split_names):
            negative_parts = []
            negative_groups = []
            for client in range(num_clients):
                group_id = ((client * spec.num_tasks + task) * 3) + split_index
                group_positive_count = int(
                    (positive_group_by_task_split[task][split] == group_id).sum()
                )
                count = (
                    round(group_positive_count * training_negative_ratio)
                    if split == "train"
                    else evaluation_negatives_per_client_task
                )
                negative_parts.append(
                    _sample_owned_negative_pairs(
                        node_ids=torch.nonzero(node_owner == client, as_tuple=True)[0],
                        num_nodes=num_nodes,
                        positive_keys=positive_keys,
                        used_keys=used_negative_keys,
                        count=count,
                        generator=torch_generator(
                            seed, "lp-partition-first-negatives", client, task, split
                        ),
                    )
                )
                negative_groups.append(
                    torch.full((count,), group_id, dtype=torch.long)
                )
            negatives[task][split] = torch.cat(negative_parts, dim=0)
            candidate_groups[task][split] = torch.cat(
                (positive_group_by_task_split[task][split], *negative_groups), dim=0
            )

    ordered_positives = []
    ordered_tasks = []
    ordered_splits = []
    for task in range(spec.num_tasks):
        for split_index, split in enumerate(split_names):
            pairs = positive_by_task_split[task][split]
            ordered_positives.append(pairs)
            ordered_tasks.append(torch.full((pairs.shape[0],), task, dtype=torch.long))
            ordered_splits.append(
                torch.full((pairs.shape[0],), split_index, dtype=torch.long)
            )
    supervised_positives = torch.cat(ordered_positives, dim=0)
    supervised_tasks = torch.cat(ordered_tasks, dim=0)
    supervised_splits = torch.cat(ordered_splits, dim=0)
    task_context_edge_index = {
        task: torch.cat(
            (spec.edge_index, _symmetric_arcs(positive_by_task_split[task]["train"])),
            dim=1,
        )
        for task in range(spec.num_tasks)
    }
    metadata = {
        **spec.metadata,
        "partition_first_pending": False,
        "partition_first_release_protocol": True,
        "partition_internal_positive_count": int(internal.sum()),
        "partition_internal_positive_coverage": float(internal.float().mean()),
        "partition_cross_client_positive_count": int((~internal).sum()),
        "partition_positive_split_counts": split_counts,
        "partition_evaluation_negatives_per_client_task": (
            evaluation_negatives_per_client_task
        ),
        "partition_assignment_used_query_endpoints": False,
        "partition_assignment_used_evaluation_data": False,
        "lp_context_policy": "fixed_base_union_current_task_train_positives",
        "task_context_edge_index": task_context_edge_index,
    }
    return build_lp_spec(
        dataset_name=spec.dataset_name,
        metrics=spec.metrics,
        node_features=spec.node_features,
        positive_pairs=supervised_positives,
        positive_task_ids=supervised_tasks,
        positive_splits=supervised_splits,
        negative_pairs_by_task_split=negatives,
        candidate_group_ids_by_task_split=candidate_groups,
        num_tasks=spec.num_tasks,
        undirected=bool(spec.metadata.get("undirected", True)),
        known_positive_pairs=known_pairs,
        context_edge_index=spec.edge_index,
        bipartite=spec.bipartite,
        node_types=spec.node_types,
        metadata=metadata,
    )


def finalize_partition_first_lp_selected_positives(
    spec: ScenarioSpec,
    node_owner: torch.Tensor,
    *,
    selected_train_pairs_by_task: Dict[int, torch.Tensor],
    heldout_positive_pairs_by_task_split: Dict[int, Dict[str, torch.Tensor]],
    num_clients: int,
    seed: int,
    training_negative_ratio: float,
    evaluation_negatives_per_client_task: int,
) -> ScenarioSpec:
    """Finalize an LP pool after a frozen positive-edge ownership allocation.

    This is the direct counterpart of :func:`finalize_partition_first_lp` for
    the exact-Dirichlet protocol.  The caller preselects only training
    positives and reserves positive validation/test anchors before ownership.
    No evaluation candidate, negative, or evaluation label participates in the
    ownership search.  Negatives are then sampled once from each owned local
    node universe and all task contexts contain only base plus current-task
    training positives.
    """
    from gecko.data.scenario import build_lp_spec

    if not spec.metadata.get("partition_first_pending", False):
        raise ValueError("Selected-positive LP finalization requires a pending pool.")
    if spec.problem_type != "LP":
        raise ValueError("Selected-positive LP finalization requires LP.")
    owner = node_owner.detach().cpu().long()
    if owner.shape != (spec.node_features.shape[0],):
        raise ValueError("LP node ownership does not align with the node universe.")
    if bool((owner < 0).any()) or bool((owner >= num_clients).any()):
        raise ValueError("LP node ownership must assign every node exactly once.")

    positive_pool = spec.metadata["positive_pairs"].detach().cpu().long()
    positive_tasks = spec.metadata["positive_task_ids"].detach().cpu().long()
    known_pairs = spec.metadata["known_positive_pairs"].detach().cpu().long()
    pool_internal = owner[positive_pool[:, 0]] == owner[positive_pool[:, 1]]
    if set(selected_train_pairs_by_task) != set(range(spec.num_tasks)):
        raise ValueError("Selected LP training positives must cover every task.")
    if set(heldout_positive_pairs_by_task_split) != set(range(spec.num_tasks)):
        raise ValueError("Reserved LP positives must cover every task.")

    num_nodes = int(spec.node_features.shape[0])
    undirected = bool(spec.metadata.get("undirected", True))

    def pair_key(pair: torch.Tensor) -> int:
        source, target = map(int, pair.tolist())
        if undirected and source > target:
            source, target = target, source
        return source * num_nodes + target

    pool_task_keys = {
        task: {
            pair_key(pair)
            for pair in positive_pool[positive_tasks == task]
        }
        for task in range(spec.num_tasks)
    }
    selected_keys: set[int] = set()
    heldout_keys: set[int] = set()
    split_names = ("train", "val", "test")
    positives: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(spec.num_tasks)
    }
    groups: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(spec.num_tasks)
    }
    split_counts = {split: 0 for split in split_names}
    for task in range(spec.num_tasks):
        heldout = heldout_positive_pairs_by_task_split[task]
        if set(heldout) != {"val", "test"}:
            raise ValueError("LP reserved positives must define val and test.")
        per_split = {
            "train": selected_train_pairs_by_task[task].detach().cpu().long(),
            "val": heldout["val"].detach().cpu().long(),
            "test": heldout["test"].detach().cpu().long(),
        }
        for split_index, split in enumerate(split_names):
            pairs = per_split[split]
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f"LP {split} positives must have shape [N, 2].")
            keys = [pair_key(pair) for pair in pairs]
            if len(keys) != len(set(keys)) or not set(keys).issubset(pool_task_keys[task]):
                raise ValueError(f"LP {split} positives are not a unique task-local pool subset.")
            if bool((owner[pairs[:, 0]] != owner[pairs[:, 1]]).any()):
                raise ValueError(f"LP {split} positives must be internal to one owner.")
            target_set = selected_keys if split == "train" else heldout_keys
            if set(keys) & selected_keys or set(keys) & heldout_keys:
                raise ValueError("LP selected and held-out positive sets must be disjoint.")
            target_set.update(keys)
            positives[task][split] = pairs.clone()
            split_counts[split] += int(pairs.shape[0])
            client_ids = owner[pairs[:, 0]]
            groups[task][split] = (
                ((client_ids * spec.num_tasks + task) * 3) + split_index
            ).long()
        if positives[task]["train"].numel() == 0:
            raise ValueError(f"LP task={task} has no selected training positives.")

    positive_keys = {pair_key(pair) for pair in known_pairs}
    used_negative_keys: set[int] = set()
    negatives: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(spec.num_tasks)
    }
    candidate_groups: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(spec.num_tasks)
    }
    for task in range(spec.num_tasks):
        for split_index, split in enumerate(split_names):
            negative_parts: list[torch.Tensor] = []
            negative_groups: list[torch.Tensor] = []
            positive_groups = groups[task][split]
            for client in range(num_clients):
                group_id = ((client * spec.num_tasks + task) * 3) + split_index
                positive_count = int((positive_groups == group_id).sum())
                count = (
                    round(positive_count * training_negative_ratio)
                    if split == "train"
                    else evaluation_negatives_per_client_task
                )
                negative_parts.append(
                    _sample_owned_negative_pairs(
                        node_ids=torch.nonzero(owner == client, as_tuple=True)[0],
                        num_nodes=num_nodes,
                        positive_keys=positive_keys,
                        used_keys=used_negative_keys,
                        count=count,
                        generator=torch_generator(
                            seed,
                            "lp-selected-positive-negatives",
                            client,
                            task,
                            split,
                        ),
                    )
                )
                negative_groups.append(torch.full((count,), group_id, dtype=torch.long))
            negatives[task][split] = torch.cat(negative_parts, dim=0)
            candidate_groups[task][split] = torch.cat(
                (positive_groups, *negative_groups), dim=0
            )

    ordered_pairs: list[torch.Tensor] = []
    ordered_tasks: list[torch.Tensor] = []
    ordered_splits: list[torch.Tensor] = []
    split_codes = {"train": 0, "val": 1, "test": 2}
    for task in range(spec.num_tasks):
        for split in split_names:
            pairs = positives[task][split]
            ordered_pairs.append(pairs)
            ordered_tasks.append(torch.full((pairs.shape[0],), task, dtype=torch.long))
            ordered_splits.append(
                torch.full((pairs.shape[0],), split_codes[split], dtype=torch.long)
            )
    supervised_pairs = torch.cat(ordered_pairs, dim=0)
    supervised_tasks = torch.cat(ordered_tasks, dim=0)
    supervised_splits = torch.cat(ordered_splits, dim=0)
    task_contexts = {
        task: torch.cat(
            (spec.edge_index, _symmetric_arcs(positives[task]["train"])), dim=1
        )
        for task in range(spec.num_tasks)
    }
    return build_lp_spec(
        dataset_name=spec.dataset_name,
        metrics=spec.metrics,
        node_features=spec.node_features,
        positive_pairs=supervised_pairs,
        positive_task_ids=supervised_tasks,
        positive_splits=supervised_splits,
        negative_pairs_by_task_split=negatives,
        candidate_group_ids_by_task_split=candidate_groups,
        num_tasks=spec.num_tasks,
        undirected=undirected,
        known_positive_pairs=known_pairs,
        context_edge_index=spec.edge_index,
        bipartite=spec.bipartite,
        node_types=spec.node_types,
        metadata={
            **spec.metadata,
            "partition_first_pending": False,
            "partition_first_release_protocol": True,
            "partition_internal_positive_count": int(pool_internal.sum()),
            "partition_internal_positive_coverage": float(pool_internal.float().mean()),
            "partition_cross_client_positive_count": int((~pool_internal).sum()),
            "partition_positive_split_counts": split_counts,
            "partition_evaluation_negatives_per_client_task": (
                evaluation_negatives_per_client_task
            ),
            "partition_assignment_used_query_endpoints": True,
            "partition_assignment_used_evaluation_data": False,
            "lp_context_policy": "fixed_base_union_current_task_selected_train_positives",
            "task_context_edge_index": task_contexts,
            "lp_selected_positive_protocol": "exact_dirichlet_train_only_v1",
        },
    )


def sample_training_negatives(
    *,
    num_nodes: int,
    positive_pairs: torch.Tensor,
    count: int,
    generator: torch.Generator,
    undirected: bool,
    bipartite: bool = False,
    node_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample fixed negatives from the global valid non-edge universe."""

    positives = {
        (min(source, target), max(source, target)) if undirected else (source, target)
        for source, target in positive_pairs.tolist()
    }
    chosen = set()
    maximum_attempts = max(10_000, count * 100)
    attempts = 0
    while len(chosen) < count and attempts < maximum_attempts:
        source = int(torch.randint(num_nodes, (1,), generator=generator))
        target = int(torch.randint(num_nodes, (1,), generator=generator))
        attempts += 1
        if source == target:
            continue
        if bipartite and node_types is not None and node_types[source] == node_types[target]:
            continue
        key = (min(source, target), max(source, target)) if undirected else (source, target)
        if key in positives or key in chosen:
            continue
        chosen.add(key)
    if len(chosen) != count:
        raise ValueError(
            f"Unable to sample {count} valid LP negatives after {maximum_attempts} attempts."
        )
    return torch.tensor(sorted(chosen), dtype=torch.long).reshape(-1, 2)


def repair_negative_pairs(
    candidates: torch.Tensor,
    *,
    positive_pairs: torch.Tensor,
    count: int,
    num_nodes: int,
    generator: torch.Generator,
    undirected: bool,
    bipartite: bool = False,
    node_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep valid fixed candidates and deterministically replace invalid rows."""

    positive_keys = {
        (min(source, target), max(source, target)) if undirected else (source, target)
        for source, target in positive_pairs.tolist()
    }
    kept: list[tuple[int, int]] = []
    seen = set()
    if candidates.ndim == 2 and candidates.shape[1] == 2:
        for source, target in candidates.long().tolist():
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                continue
            if source == target:
                continue
            if bipartite and node_types is not None and node_types[source] == node_types[target]:
                continue
            key = (min(source, target), max(source, target)) if undirected else (source, target)
            if key in positive_keys or key in seen:
                continue
            seen.add(key)
            kept.append(key)
            if len(kept) == count:
                break
    missing = count - len(kept)
    if missing > 0:
        forbidden = positive_pairs
        if kept:
            forbidden = torch.cat(
                [positive_pairs, torch.tensor(kept, dtype=torch.long)], dim=0
            )
        generated = sample_training_negatives(
            num_nodes=num_nodes,
            positive_pairs=forbidden,
            count=missing,
            generator=generator,
            undirected=undirected,
            bipartite=bipartite,
            node_types=node_types,
        )
        kept.extend(tuple(pair) for pair in generated.tolist())
    return torch.tensor(kept, dtype=torch.long).reshape(-1, 2)




_RELOCATED_EXPORTS = {'build_lp_spec': ('gecko.data.scenario', 'build_lp_spec')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
