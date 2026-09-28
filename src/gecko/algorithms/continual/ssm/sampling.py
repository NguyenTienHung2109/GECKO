from __future__ import annotations

from gecko.algorithms.continual.ssm.records import _FEATURE_DTYPES

from typing import Sequence
from typing import Tuple
import torch

def _sample_candidates(
    candidates: Sequence[int],
    *,
    count: int,
    sampler_mode: str,
    in_degrees: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[int, ...]:
    if count <= 0 or not candidates:
        return ()
    ordered = tuple(sorted(set(int(value) for value in candidates)))
    take = min(count, len(ordered))
    if sampler_mode == "degree":
        weights = in_degrees[torch.tensor(ordered, dtype=torch.long)].double()
        positive = (weights > 0).nonzero(as_tuple=False).reshape(-1)
        zero = (weights == 0).nonzero(as_tuple=False).reshape(-1)
        positive_take = min(take, int(positive.numel()))
        chosen: list[int] = []
        if positive_take:
            selected_positive = torch.multinomial(
                weights[positive],
                positive_take,
                replacement=False,
                generator=generator,
            )
            chosen.extend(positive[selected_positive].tolist())
        remaining = take - len(chosen)
        if remaining:
            zero_order = torch.randperm(int(zero.numel()), generator=generator)
            chosen.extend(zero[zero_order[:remaining]].tolist())
        indices = torch.tensor(chosen, dtype=torch.long)
    else:
        indices = torch.randperm(len(ordered), generator=generator)[:take]
    # Canonical node order is independent of incidental draw return ordering.
    return tuple(sorted(ordered[int(index)] for index in indices.tolist()))


class _PreparedSparseComputationSampler:
    def __init__(self, *, node_features: torch.Tensor, edge_index: torch.Tensor) -> None:
        from gecko.algorithms.continual.ssm.records import _FEATURE_DTYPES
        from gecko.algorithms.continual.ssm.records import _owned_tensor
        from gecko.algorithms.continual.ssm.records import _validate_local_edge_index
        features = _owned_tensor(node_features, name="node_features")
        if features.dtype not in _FEATURE_DTYPES or features.ndim != 2:
            raise ValueError("node_features must be a two-dimensional floating tensor.")
        if features.shape[0] == 0:
            raise ValueError("node_features must contain at least one local node.")
        edges = _validate_local_edge_index(edge_index, num_nodes=features.shape[0])
        self.features = features
        self.edge_index = edges
        incoming: list[list[int]] = [[] for _ in range(int(features.shape[0]))]
        for raw_source, raw_target in edges.t().tolist():
            incoming[int(raw_target)].append(int(raw_source))
        self.incoming = tuple(tuple(values) for values in incoming)
        self.in_degrees = torch.tensor(
            [len(values) for values in self.incoming], dtype=torch.double
        )

    def sample(
        self,
        *,
        root_index: int,
        root_label: int,
        client_id: int,
        global_task_id: int,
        stage_index: int,
        sampler_mode: str,
        hop_budgets: Sequence[int],
        rng_seed: int,
        rng_base_seed: int | None = None,
    ) -> SSMRecord:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        from gecko.algorithms.continual.ssm.records import _validate_nonnegative_int
        features = self.features
        root = _validate_nonnegative_int(root_index, name="root_index")
        if root >= features.shape[0]:
            raise ValueError("root_index is outside the strict-local graph.")
        mode = str(sampler_mode)
        if mode not in {"uniform", "degree"}:
            raise ValueError("sampler_mode must be 'uniform' or 'degree'.")
        budgets = tuple(
            _validate_nonnegative_int(value, name=f"hop_budgets[{index}]")
            for index, value in enumerate(hop_budgets)
        )
        if not budgets:
            raise ValueError("hop_budgets cannot be empty.")
        record_seed = _validate_nonnegative_int(rng_seed, name="rng_seed")
        base_seed = (
            record_seed
            if rng_base_seed is None
            else _validate_nonnegative_int(rng_base_seed, name="rng_base_seed")
        )
        if record_seed >= 2**63 or base_seed >= 2**63:
            raise ValueError("SSM RNG seeds must fit a signed 63-bit generator seed.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(record_seed)
        selected_order = [root]
        selected = {root}
        frontier = (root,)
        retained_arcs: list[tuple[int, int]] = []
        draw_count = 0
        for budget in budgets:
            if not frontier or budget == 0:
                frontier = ()
                continue
            candidates = sorted(
                {
                    source
                    for target in frontier
                    for source in self.incoming[target]
                    if source not in selected
                }
            )
            sampled = _sample_candidates(
                candidates,
                count=budget,
                sampler_mode=mode,
                in_degrees=self.in_degrees,
                generator=generator,
            )
            draw_count += len(sampled)
            sampled_set = set(sampled)
            retained_arcs.extend(
                (source, target)
                for target in frontier
                for source in self.incoming[target]
                if source in sampled_set
            )
            selected_order.extend(sampled)
            selected.update(sampled)
            frontier = sampled

        source_nodes = torch.tensor(selected_order, dtype=torch.long)
        compact = {source: index for index, source in enumerate(selected_order)}
        compact_arcs = sorted(
            (compact[source], compact[target]) for source, target in retained_arcs
        )
        compact_edges = (
            torch.tensor(compact_arcs, dtype=torch.long).t().contiguous()
            if compact_arcs
            else torch.empty((2, 0), dtype=torch.long)
        )
        label = _validate_nonnegative_int(root_label, name="root_label")
        return SSMRecord(
            client_id=client_id,
            global_task_id=global_task_id,
            stage_index=stage_index,
            class_id=label,
            source_root_index=root,
            root_index=0,
            root_label=label,
            sampler_mode=mode,
            hop_budgets=budgets,
            rng_algorithm="torch.Generator.cpu.manual_seed",
            rng_base_seed=base_seed,
            rng_record_seed=record_seed,
            sampled_node_count=draw_count,
            features=features[source_nodes],
            edge_index=compact_edges,
            source_local_nodes=source_nodes,
        )


def sample_sparse_computation_record(
    *,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    root_index: int,
    root_label: int,
    client_id: int,
    global_task_id: int,
    stage_index: int,
    sampler_mode: str,
    hop_budgets: Sequence[int],
    rng_seed: int,
    rng_base_seed: int | None = None,
) -> SSMRecord:
    """Sample one strict-local computation graph without replacement.

    COO arcs use UEFA's ``source -> target`` message-passing convention.  At
    every hop, candidates are unselected sources of arcs entering the current
    frontier.  Only exact arcs from sampled sources into that frontier are
    retained; the routine never materializes an induced graph or requests a
    global dataset handle.
    """
    from gecko.algorithms.continual.ssm.records import SSMRecord

    sampler = _PreparedSparseComputationSampler(
        node_features=node_features, edge_index=edge_index
    )
    return sampler.sample(
        root_index=root_index,
        root_label=root_label,
        client_id=client_id,
        global_task_id=global_task_id,
        stage_index=stage_index,
        sampler_mode=sampler_mode,
        hop_budgets=hop_budgets,
        rng_seed=rng_seed,
        rng_base_seed=rng_base_seed,
    )


