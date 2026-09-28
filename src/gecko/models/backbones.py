"""Pure PyTorch static-graph model used by deterministic UEFA smoke tests."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


# Cap each materialized E x hidden tensor at roughly 16 MiB for fp32.  The
# deterministic prefix-sum reducer retains both messages and a cumulative
# tensor, and higher-order method penalties retain their autograd graph.
# The chunk boundary is always a complete target-node segment, so changing the
# chunk size cannot change the mathematical reduction order within a node.
_DETERMINISTIC_MESSAGE_ELEMENT_BUDGET = 4 * 1024 * 1024
_DETERMINISTIC_PLAN_CACHE_BYTES = 64 * 1024 * 1024
_DETERMINISTIC_PLAN_CACHE_SIZE = 8
_DETERMINISTIC_PLAN_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
] = OrderedDict()


def _sorted_segment_plan(
    source: torch.Tensor, target: torch.Tensor, node_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached target-sorted edge metadata for deterministic aggregation."""

    key = (
        source.data_ptr(),
        target.data_ptr(),
        source._version,
        target._version,
        int(source.numel()),
        node_count,
        str(source.device),
    )
    cached = _DETERMINISTIC_PLAN_CACHE.get(key)
    if cached is not None:
        _DETERMINISTIC_PLAN_CACHE.move_to_end(key)
        return cached[2]
    order = torch.argsort(target * node_count + source, stable=True)
    sorted_source = source[order]
    sorted_target = target[order]
    row_counts = torch.bincount(sorted_target, minlength=node_count)
    crow = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=source.device),
            row_counts.cumsum(0),
        )
    )
    plan = (order, sorted_source, row_counts, crow, sorted_target)
    # The cached tensors are three E-length int64 vectors plus O(N) metadata.
    # Refuse large graph plans rather than retaining a hidden multi-GB cache.
    estimated_bytes = int(source.numel()) * 3 * 8 + node_count * 2 * 8
    if estimated_bytes <= _DETERMINISTIC_PLAN_CACHE_BYTES:
        # Retain the source/target views to prevent allocator pointer reuse
        # from making a stale plan appear valid after a temporary graph dies.
        _DETERMINISTIC_PLAN_CACHE[key] = (source, target, plan)
        _DETERMINISTIC_PLAN_CACHE.move_to_end(key)
        while len(_DETERMINISTIC_PLAN_CACHE) > _DETERMINISTIC_PLAN_CACHE_SIZE:
            _DETERMINISTIC_PLAN_CACHE.popitem(last=False)
    return plan


def _prefix_segment_sum(
    messages: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Sum contiguous segments with a twice-differentiable prefix reduction."""

    width = int(messages.shape[1])
    if not int(messages.shape[0]):
        return messages.new_zeros((int(lengths.numel()), width))
    cumulative = messages.cumsum(dim=0)
    ends = lengths.cumsum(0) - 1
    nonempty = lengths > 0
    safe_ends = ends.clamp_min(0)
    output = cumulative.index_select(0, safe_ends)
    starts = ends - lengths
    previous = cumulative.index_select(0, starts.clamp_min(0))
    output = output - previous * (starts >= 0).to(messages.dtype).unsqueeze(1)
    return output * nonempty.to(messages.dtype).unsqueeze(1)


def _deterministic_segment_sum(
    features: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate sorted target segments without CUDA atomic scatter/reduction."""

    node_count = int(features.shape[0])
    if not source.numel():
        return torch.zeros_like(features)
    order, sorted_source, row_counts, crow, _ = _sorted_segment_plan(
        source, target, node_count
    )
    sorted_weights = None if weights is None else weights[order]
    edge_budget = max(
        1,
        _DETERMINISTIC_MESSAGE_ELEMENT_BUDGET // max(1, int(features.shape[1])),
    )
    chunks: list[torch.Tensor] = []
    start_node = 0
    edge_count = int(source.numel())
    while start_node < node_count:
        start_edge = int(crow[start_node].item())
        target_edge = min(edge_count, start_edge + edge_budget)
        boundary = torch.searchsorted(
            crow,
            torch.tensor(target_edge, device=crow.device, dtype=crow.dtype),
            right=True,
        )
        end_node = min(node_count, max(start_node + 1, int(boundary.item()) - 1))
        end_edge = int(crow[end_node].item())
        selected_source = sorted_source[start_edge:end_edge]
        messages = features[selected_source]
        if sorted_weights is not None:
            messages = messages * sorted_weights[start_edge:end_edge].unsqueeze(1)
        chunks.append(_prefix_segment_sum(messages, row_counts[start_node:end_node]))
        start_node = end_node
    return torch.cat(chunks, dim=0)


class MeanGraphLayer(nn.Module):
    """Mean-neighbor layer requiring only a COO edge tensor."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_size, output_size)
        self.neighbor_linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(features)
        degree = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        if edge_index.numel():
            source, target = edge_index
            aggregate = _deterministic_segment_sum(features, source, target)
            degree = torch.bincount(
                target,
                minlength=features.shape[0],
            ).to(features.dtype)
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)
        return self.self_linear(features) + self.neighbor_linear(aggregate)


class GECKOGraphModel(nn.Module):
    """One static graph encoder with NC, LC, and dot-product LP heads."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        problem_type: str = "NC",
    ) -> None:
        super().__init__()
        self.problem_type = problem_type.upper()
        dimensions = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            MeanGraphLayer(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )
        self.node_head = nn.Linear(hidden_size, output_size)
        self.edge_head = nn.Linear(hidden_size * 2, output_size)

    def encode(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.layers:
            hidden = F.relu(layer(hidden, edge_index))
        return hidden

    def forward_node(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.node_head(self.encode(features, edge_index))

    def encode_nodes(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        """Expose a requested post-activation encoder layer."""

        target = len(self.layers) - 1 if layer_index is None else int(layer_index)
        if target < 0:
            target += len(self.layers)
        if target < 0 or target >= len(self.layers):
            raise ValueError(
                f"layer_index={layer_index} is outside a {len(self.layers)}-layer encoder."
            )
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = F.relu(layer(hidden, edge_index))
            if index == target:
                return hidden
        raise AssertionError("Validated encoder layer was not reached.")

    def forward_edge(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        query_pairs: torch.Tensor,
        *,
        link_prediction: bool = False,
    ) -> torch.Tensor:
        hidden = self.encode(features, edge_index)
        source = hidden[query_pairs[:, 0]]
        target = hidden[query_pairs[:, 1]]
        if link_prediction:
            return (source * target).sum(dim=1)
        edge_features = torch.cat([source * target, torch.abs(source - target)], dim=1)
        return self.edge_head(edge_features)

    def forward_queries(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
        problem_type: str,
    ) -> torch.Tensor:
        problem = problem_type.upper()
        if problem == "NC":
            return self.forward_node(features, edge_index)[queries]
        return self.forward_edge(
            features,
            edge_index,
            queries,
            link_prediction=problem == "LP",
        )


class LegacyGCNAdapter(nn.Module):
    """Lazy COO adapter for BeGin's original GCNNode/GCNLink models."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        problem_type: str = "NC",
        incremental_type: str = "class",
    ) -> None:
        super().__init__()
        try:
            from gecko.models.legacy.models import GCNLink
            from gecko.models.legacy.models import GCNNode
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "The original BeGin GCN adapter requires DGL, torch-scatter, "
                "and the original BeGin model dependencies."
            ) from error
        self.problem_type = problem_type.upper()
        if self.problem_type == "NC":
            self.original = GCNNode(
                input_size,
                output_size,
                hidden_size,
                n_layers=num_layers,
                incr_type=incremental_type,
            )
            self.original.observe_labels(torch.arange(output_size), verbose=False)
        elif self.problem_type == "LC":
            self.original = GCNLink(
                input_size,
                output_size,
                hidden_size,
                n_layers=num_layers,
                incr_type=incremental_type,
            )
            self.original.observe_labels(torch.arange(output_size), verbose=False)
        elif self.problem_type == "LP":
            self.original = GCNNode(
                input_size,
                hidden_size,
                hidden_size,
                n_layers=num_layers,
                incr_type="class",
                use_classifier=False,
            )
        else:
            raise ValueError(f"Original BeGin GCN does not support {self.problem_type}.")
        self._cached_graph_key = None
        self._cached_graph = None

    @staticmethod
    def _symmetric_aggregate(
        features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Pure-torch equivalent of DGL GraphConv ``norm='both'`` messages."""

        if not edge_index.numel():
            return torch.zeros_like(features)
        source, target = edge_index
        source_degree = torch.bincount(
            source, minlength=features.shape[0]
        ).to(features.dtype)
        target_degree = torch.bincount(
            target, minlength=features.shape[0]
        ).to(features.dtype)
        values = (
            source_degree[source].clamp_min(1).pow(-0.5)
            * target_degree[target].clamp_min(1).pow(-0.5)
        )
        return _deterministic_segment_sum(
            features,
            source,
            target,
            weights=values,
        )

    def _encoder_and_edges(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[nn.Module, torch.Tensor]:
        encoder = self.original.gcn if self.problem_type == "LC" else self.original
        resolved_edges = edge_index
        if self.problem_type in {"LC", "LP"}:
            nodes = torch.arange(num_nodes, device=edge_index.device)
            self_loops = torch.stack((nodes, nodes))
            resolved_edges = torch.cat((edge_index, self_loops), dim=1)
        return encoder, resolved_edges

    def _torch_encode_nodes(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Run BeGin GCN parameters without requiring a CUDA-enabled DGL build."""

        encoder, resolved_edges = self._encoder_and_edges(
            edge_index, int(features.shape[0])
        )
        hidden = encoder.dropout(features)
        for layer, norm in zip(encoder.convs, encoder.norms):
            weight = layer.weight
            if weight is None:
                raise RuntimeError("BeGin GraphConv unexpectedly has no weight.")
            hidden = self._symmetric_aggregate(hidden, resolved_edges) @ weight
            if layer.bias is not None:
                hidden = hidden + layer.bias
            hidden = norm(hidden)
            hidden = encoder.activation(hidden)
            hidden = encoder.dropout(hidden)
        return hidden

    def _torch_forward_queries(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
        problem: str,
    ) -> torch.Tensor:
        hidden = self._torch_encode_nodes(features, edge_index)
        if problem == "NC":
            classifier = self.original.classifier
            if classifier is None:
                raise RuntimeError("BeGin NC adapter has no classifier.")
            return classifier(hidden)[queries]
        pair_hidden = hidden[queries[:, 0]] * hidden[queries[:, 1]]
        if problem == "LP":
            return pair_hidden.sum(dim=1)
        pair_hidden = self.original.dropout(pair_hidden)
        for layer in self.original.linears:
            pair_hidden = self.original.activation(layer(pair_hidden))
            pair_hidden = self.original.dropout(pair_hidden)
        return self.original.classifier(pair_hidden)

    def _graph(self, edge_index: torch.Tensor, num_nodes: int):
        import dgl

        key = (
            edge_index.data_ptr(),
            edge_index._version,
            tuple(edge_index.shape),
            num_nodes,
            str(edge_index.device),
        )
        if self._cached_graph_key == key:
            return self._cached_graph
        if self.problem_type in {"LC", "LP"} and edge_index.device.type != "cpu":
            # The release DGL build cannot execute ``add_self_loop`` on CUDA.
            # Construct the identical topology on CPU, add loops there, then
            # transfer once; the cache below prevents repeated transfers.
            graph = dgl.graph(
                (edge_index[0].cpu(), edge_index[1].cpu()),
                num_nodes=num_nodes,
            )
            graph = dgl.add_self_loop(graph).to(edge_index.device)
        else:
            graph = dgl.graph(
                (edge_index[0], edge_index[1]),
                num_nodes=num_nodes,
                device=edge_index.device,
            )
            if self.problem_type in {"LC", "LP"}:
                graph = dgl.add_self_loop(graph)
        self._cached_graph_key = key
        self._cached_graph = graph
        return graph

    def forward_queries(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
        problem_type: str,
    ) -> torch.Tensor:
        problem = problem_type.upper()
        if edge_index.device.type != "cpu":
            return self._torch_forward_queries(
                features, edge_index, queries, problem
            )
        graph = self._graph(edge_index, features.shape[0])
        if problem == "NC":
            return self.original(graph, features)[queries]
        if problem == "LC":
            return self.original(
                graph, features, queries[:, 0], queries[:, 1]
            )
        hidden = self.original(graph, features)
        return (hidden[queries[:, 0]] * hidden[queries[:, 1]]).sum(dim=1)

    def encode_nodes(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        """Return final pre-classifier node embeddings when the model has them."""

        if layer_index not in (None, -1):
            raise ValueError("LegacyGCNAdapter exposes only its final encoder layer.")
        if edge_index.device.type != "cpu":
            return self._torch_encode_nodes(features, edge_index)
        graph = self._graph(edge_index, features.shape[0])
        if self.problem_type == "NC":
            return self.original.forward_without_classifier(graph, features)
        if self.problem_type in {"LC", "LP"}:
            encoder = self.original.gcn if self.problem_type == "LC" else self.original
            method = getattr(encoder, "forward_without_classifier", None)
            if callable(method):
                return method(graph, features)
        raise RuntimeError(
            f"{type(self.original).__name__} does not expose a node encoder."
        )
