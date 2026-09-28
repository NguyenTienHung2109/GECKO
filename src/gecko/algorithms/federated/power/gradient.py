"""POWER prototype-gradient encoding and inversion primitives."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class PowerGradientEncoder(nn.Module):
    """Four-layer MLP used by POWER to encode class means as gradients."""

    def __init__(self, input_dim: int, output_dim: int, *, seed: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 1:
            raise ValueError("POWER gradient encoder dimensions are invalid.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        # The released POWER source defines a custom uniform initializer but
        # never applies it to GEModel.  Its executed behavior is therefore
        # nn.Linear.reset_parameters (Kaiming-uniform weights and fan-in-scaled
        # biases).  Fork the RNG so reproducing that behavior does not mutate
        # UEFA's process-global random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.lin1 = nn.Linear(self.input_dim, 128)
            self.lin2 = nn.Linear(128, 128)
            self.lin3 = nn.Linear(128, 64)
            self.lin4 = nn.Linear(64, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.lin1(features))
        hidden = F.relu(self.lin2(hidden))
        hidden = F.relu(self.lin3(hidden))
        return self.lin4(hidden)


def power_encode_prototype_gradients(
    *,
    encoder: PowerGradientEncoder,
    prototypes: torch.Tensor,
    class_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Return the exact CE gradient tuple uploaded for every class mean."""

    if (
        prototypes.ndim != 2
        or not prototypes.is_floating_point()
        or prototypes.shape[1] != encoder.input_dim
    ):
        raise ValueError("POWER prototypes do not match the gradient encoder.")
    if (
        class_ids.ndim != 1
        or class_ids.dtype != torch.long
        or class_ids.shape[0] != prototypes.shape[0]
    ):
        raise ValueError("POWER class IDs must align with prototypes.")
    if torch.any(class_ids < 0) or torch.any(class_ids >= encoder.output_dim):
        raise ValueError("POWER class ID is outside the encoder output space.")
    if prototypes.shape[0] == 0:
        raise ValueError("POWER needs at least one class prototype.")

    parameters = tuple(encoder.parameters())
    encoded: list[tuple[torch.Tensor, ...]] = []
    encoder.train()
    for prototype, class_id in zip(prototypes, class_ids.tolist(), strict=True):
        logits = encoder(prototype.to(next(encoder.parameters()).device))
        loss = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([class_id], dtype=torch.long, device=logits.device),
        )
        gradients = torch.autograd.grad(loss, parameters)
        encoded.append(
            tuple(
                gradient.detach().cpu().clone().contiguous()
                for gradient in gradients
            )
        )
    return tuple(encoded)


def power_gradient_label(
    gradients: Sequence[torch.Tensor], *, output_dim: int
) -> int:
    """Infer the prototype class from the final-layer bias gradient."""

    if len(gradients) != 8:
        raise ValueError("POWER gradient payload must contain eight tensors.")
    bias_gradient = gradients[-1]
    if (
        bias_gradient.ndim != 1
        or bias_gradient.shape[0] != output_dim
        or not bias_gradient.is_floating_point()
        or not torch.isfinite(bias_gradient).all()
    ):
        raise ValueError("POWER final bias gradient is invalid.")
    return int(torch.argmin(bias_gradient).item())


def power_gradient_matching_loss(
    *,
    encoder: PowerGradientEncoder,
    candidate: torch.Tensor,
    class_id: int,
    target_gradients: Sequence[torch.Tensor],
    create_graph: bool,
) -> torch.Tensor:
    """Squared gradient distance used by POWER's LBFGS inversion."""

    parameters = tuple(encoder.parameters())
    if len(target_gradients) != len(parameters):
        raise ValueError("POWER target gradient structure does not match encoder.")
    logits = encoder(candidate)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    labels = torch.full(
        (logits.shape[0],), int(class_id), dtype=torch.long, device=logits.device
    )
    loss = F.cross_entropy(logits, labels)
    gradients = torch.autograd.grad(loss, parameters, create_graph=create_graph)
    objective = torch.zeros((), dtype=logits.dtype, device=logits.device)
    for produced, target in zip(gradients, target_gradients, strict=True):
        if produced.shape != target.shape:
            raise ValueError("POWER target gradient tensor shape is invalid.")
        objective = objective + torch.sum(
            (produced - target.to(produced.device, produced.dtype)) ** 2
        )
    return objective


def power_reconstruct_from_gradients(
    *,
    encoder: PowerGradientEncoder,
    target_gradients: Sequence[torch.Tensor],
    class_id: int,
    steps: int,
    seed: int,
    lbfgs_learning_rate: float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """Invert one uploaded prototype gradient with POWER's LBFGS objective."""

    if steps <= 0:
        raise ValueError("POWER reconstruction steps must be positive.")
    if lbfgs_learning_rate <= 0.0:
        raise ValueError("POWER LBFGS learning rate must be positive.")
    inferred = power_gradient_label(target_gradients, output_dim=encoder.output_dim)
    if inferred != int(class_id):
        raise ValueError("POWER declared and gradient-inferred labels disagree.")

    device = next(encoder.parameters()).device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initial = torch.randn((1, encoder.input_dim), generator=generator)
    candidate = initial.to(device).requires_grad_(True)
    # One outer call is one paper iteration.  PyTorch otherwise performs up to
    # 20 internal iterations per step, silently turning 300 into 6000.
    optimizer = torch.optim.LBFGS(
        [candidate], lr=float(lbfgs_learning_rate), max_iter=1
    )
    frozen_targets = tuple(
        value.detach().clone().to(device) for value in target_gradients
    )

    initial_loss = float(
        power_gradient_matching_loss(
            encoder=encoder,
            candidate=candidate,
            class_id=class_id,
            target_gradients=frozen_targets,
            create_graph=False,
        )
        .detach()
        .cpu()
    )

    for _ in range(int(steps)):

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            objective = power_gradient_matching_loss(
                encoder=encoder,
                candidate=candidate,
                class_id=class_id,
                target_gradients=frozen_targets,
                create_graph=True,
            )
            objective.backward()
            return objective

        optimizer.step(closure)

    final_loss = float(
        power_gradient_matching_loss(
            encoder=encoder,
            candidate=candidate,
            class_id=class_id,
            target_gradients=frozen_targets,
            create_graph=False,
        )
        .detach()
        .cpu()
    )
    return candidate.detach().cpu().clone().contiguous(), initial_loss, final_loss


def power_gradient_payload_bytes(
    gradients: Iterable[Sequence[torch.Tensor]], counts: torch.Tensor
) -> int:
    """Count raw tensor bytes in a prototype-gradient upload."""

    total = counts.numel() * counts.element_size()
    for gradient_tuple in gradients:
        total += sum(value.numel() * value.element_size() for value in gradient_tuple)
    return int(total)
