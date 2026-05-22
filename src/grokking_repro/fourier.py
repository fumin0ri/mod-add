from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .model import ModularAdditionTransformer


def make_fourier_basis(modulus: int) -> tuple[torch.Tensor, list[tuple[str, int, str]]]:
    basis = [torch.ones(modulus) / modulus**0.5]
    metadata = [("Const", 0, "const")]
    positions = torch.arange(modulus)
    for frequency in range(1, modulus // 2 + 1):
        cos = torch.cos(2 * torch.pi * positions * frequency / modulus)
        sin = torch.sin(2 * torch.pi * positions * frequency / modulus)
        basis.append(cos / cos.norm())
        basis.append(sin / sin.norm())
        metadata.append((f"cos {frequency}", frequency, "cos"))
        metadata.append((f"sin {frequency}", frequency, "sin"))
    return torch.stack(basis, dim=0), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = ModularAdditionTransformer(
        modulus=cfg["modulus"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_mlp=cfg["d_mlp"],
        n_layers=cfg["n_layers"],
        causal=cfg.get("causal", True),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Ignore the "=" token. Our embedding is [input_token, d_model], while Nanda's W_E
    # is [d_model, input_token], so this is equivalent to (W_E @ fourier_basis.T).T.
    embedding = model.token_embed.weight.detach()[: cfg["modulus"]]
    fourier_basis, metadata = make_fourier_basis(cfg["modulus"])
    fourier_embedding = fourier_basis @ embedding
    component_norm = fourier_embedding.norm(dim=1)
    squared_norm = component_norm.pow(2)
    squared_norm_fraction = squared_norm / squared_norm.sum()

    out = Path(args.out) if args.out else checkpoint_path.parent.parent / "fourier_embedding.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "basis_index",
                "basis_name",
                "frequency",
                "component",
                "l2_norm_over_d_model",
                "squared_norm_fraction",
            ],
        )
        writer.writeheader()
        for index, ((basis_name, frequency, component), norm, frac) in enumerate(
            zip(metadata, component_norm.tolist(), squared_norm_fraction.tolist())
        ):
            writer.writerow(
                {
                    "basis_index": index,
                    "basis_name": basis_name,
                    "frequency": frequency,
                    "component": component,
                    "l2_norm_over_d_model": norm,
                    "squared_norm_fraction": frac,
                }
            )

    pair_norms = []
    for frequency in range(1, cfg["modulus"] // 2 + 1):
        cos_index = 2 * frequency - 1
        sin_index = 2 * frequency
        combined = torch.linalg.vector_norm(component_norm[[cos_index, sin_index]])
        pair_norms.append((frequency, combined.item()))
    pair_norms.sort(key=lambda item: item[1], reverse=True)

    print("top embedding Fourier frequencies:")
    for frequency, value in pair_norms[:10]:
        print(f"k={frequency:02d} combined_cos_sin_l2_norm={value:.6f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
