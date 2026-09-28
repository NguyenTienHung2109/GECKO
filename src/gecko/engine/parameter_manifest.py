"""Exhaustive parameter classification for explicit UEFA v2 execution."""

from __future__ import annotations

import torch
from torch import nn

from gecko.engine.parameter_policy import SharedParameterPolicy
from gecko.engine.protocol import ParameterEntry
from gecko.engine.protocol import ParameterManifest


def build_parameter_manifest(
    model: nn.Module,
    policy: SharedParameterPolicy | None = None,
) -> ParameterManifest:
    """Classify shared/local trainables, buffers, and dynamic metadata."""

    resolved_policy = SharedParameterPolicy() if policy is None else policy
    frozen = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen:
        raise ValueError(
            "UEFA v2 has no frozen-parameter policy; frozen parameters must "
            f"be classified explicitly before execution: {sorted(frozen)}"
        )
    shared = set(resolved_policy.shareable_keys(model))
    entries = []
    for name, parameter in model.named_parameters():
        kind = "shared_trainable" if name in shared else "local_trainable"
        entries.append(
            ParameterEntry(
                name=name,
                kind=kind,
                shape=tuple(parameter.shape),
                dtype=str(parameter.dtype),
                requires_grad=bool(parameter.requires_grad),
                wire_eligible=kind == "shared_trainable",
                checkpoint_persistent=True,
            )
        )
    for name, buffer in model.named_buffers():
        entries.append(
            ParameterEntry(
                name=name,
                kind="local_buffer",
                shape=tuple(buffer.shape),
                dtype=str(buffer.dtype),
                checkpoint_persistent=True,
            )
        )
    dynamic_names = set()
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        observed = getattr(module, "observed", None)
        if torch.is_tensor(observed):
            name = f"{prefix}observed"
            dynamic_names.add(name)
            entries.append(
                ParameterEntry(
                    name=name,
                    kind="dynamic_metadata",
                    shape=tuple(observed.shape),
                    dtype=str(observed.dtype),
                    checkpoint_persistent=True,
                )
            )
        if hasattr(module, "output_masks"):
            name = f"{prefix}output_masks"
            if name not in dynamic_names:
                dynamic_names.add(name)
                entries.append(
                    ParameterEntry(
                        name=name,
                        kind="dynamic_metadata",
                        dtype="python:list[torch.Tensor]",
                        checkpoint_persistent=True,
                    )
                )
    for name in ("_cached_graph_key", "_cached_graph"):
        if hasattr(model, name):
            entries.append(
                ParameterEntry(
                    name=name,
                    kind="dynamic_metadata",
                    dtype="python:recomputable",
                    checkpoint_persistent=False,
                )
            )
    return ParameterManifest(tuple(entries))
