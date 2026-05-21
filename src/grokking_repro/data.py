from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ModularAdditionData:
    train_tokens: torch.Tensor
    train_labels: torch.Tensor
    test_tokens: torch.Tensor
    test_labels: torch.Tensor
    all_tokens: torch.Tensor
    all_labels: torch.Tensor


def make_modular_addition_data(
    modulus: int,
    train_fraction: float,
    seed: int,
    device: torch.device,
) -> ModularAdditionData:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    train_size = int(train_fraction * len(pairs))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    eq_token = modulus
    tokens = torch.tensor([[a, b, eq_token] for a, b in pairs], dtype=torch.long)
    labels = torch.tensor([(a + b) % modulus for a, b in pairs], dtype=torch.long)

    return ModularAdditionData(
        train_tokens=tokens[train_idx].to(device),
        train_labels=labels[train_idx].to(device),
        test_tokens=tokens[test_idx].to(device),
        test_labels=labels[test_idx].to(device),
        all_tokens=tokens.to(device),
        all_labels=labels.to(device),
    )

