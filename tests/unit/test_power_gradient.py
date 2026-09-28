from __future__ import annotations

import math

import torch

from gecko.algorithms.federated.power.gradient import PowerGradientEncoder
from gecko.algorithms.federated.power.gradient import power_encode_prototype_gradients
from gecko.algorithms.federated.power.gradient import power_gradient_label
from gecko.algorithms.federated.power.gradient import power_gradient_matching_loss
from gecko.algorithms.federated.power.gradient import power_reconstruct_from_gradients


def test_power_gradient_encoder_matches_official_architecture_and_init():
    first = PowerGradientEncoder(3, 4, seed=13)
    second = PowerGradientEncoder(3, 4, seed=13)
    assert [tuple(layer.weight.shape) for layer in (first.lin1, first.lin2, first.lin3, first.lin4)] == [
        (128, 3),
        (128, 128),
        (64, 128),
        (4, 64),
    ]
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
    for layer in (first.lin1, first.lin2, first.lin3, first.lin4):
        bound = 1.0 / math.sqrt(layer.in_features)
        assert float(layer.weight.abs().max()) <= bound
        assert float(layer.bias.abs().max()) <= bound
    assert float(first.lin2.weight.abs().max()) < 0.1


def test_power_gradient_encoder_does_not_advance_the_global_rng():
    torch.manual_seed(31)
    expected = torch.rand(4)
    torch.manual_seed(31)
    PowerGradientEncoder(3, 4, seed=13)
    actual = torch.rand(4)
    assert torch.equal(actual, expected)


def test_power_upload_is_exact_gradient_and_label_is_inferred_from_bias():
    encoder = PowerGradientEncoder(2, 3, seed=7)
    prototypes = torch.tensor([[1.5, -0.5], [-0.2, 0.8]])
    classes = torch.tensor([0, 2])
    encoded = power_encode_prototype_gradients(
        encoder=encoder, prototypes=prototypes, class_ids=classes
    )
    assert len(encoded) == 2
    assert all(len(gradients) == 8 for gradients in encoded)
    assert [power_gradient_label(value, output_dim=3) for value in encoded] == [0, 2]


def test_power_lbfgs_reconstruction_is_deterministic_and_reduces_loss():
    encoder = PowerGradientEncoder(2, 3, seed=11)
    encoded = power_encode_prototype_gradients(
        encoder=encoder,
        prototypes=torch.tensor([[0.7, -1.2]]),
        class_ids=torch.tensor([1]),
    )[0]
    first, initial, final = power_reconstruct_from_gradients(
        encoder=encoder,
        target_gradients=encoded,
        class_id=1,
        steps=8,
        seed=19,
    )
    second, _, second_final = power_reconstruct_from_gradients(
        encoder=encoder,
        target_gradients=encoded,
        class_id=1,
        steps=8,
        seed=19,
    )
    assert torch.equal(first, second)
    assert final == second_final
    assert final < initial
    assert final == float(
        power_gradient_matching_loss(
            encoder=encoder,
            candidate=first,
            class_id=1,
            target_gradients=encoded,
            create_graph=False,
        )
    )
