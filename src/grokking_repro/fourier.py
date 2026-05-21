from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .model import ModularAdditionTransformer


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

    # Ignore the "=" token and measure how much embedding energy sits at each DFT frequency.
    embedding = model.token_embed.weight.detach()[: cfg["modulus"]]
    spectrum = torch.fft.rfft(embedding, dim=0)
    energy = spectrum.abs().pow(2).sum(dim=1)
    energy = energy / energy.sum()

    out = Path(args.out) if args.out else checkpoint_path.parent.parent / "fourier_embedding.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frequency", "energy_fraction"])
        writer.writeheader()
        for k, value in enumerate(energy.tolist()):
            writer.writerow({"frequency": k, "energy_fraction": value})

    top = torch.topk(energy[1:], k=min(10, len(energy) - 1))
    ranked = [(int(i.item() + 1), float(v.item())) for v, i in zip(top.values, top.indices)]
    print("top embedding Fourier frequencies:")
    for frequency, value in ranked:
        print(f"k={frequency:02d} energy_fraction={value:.6f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

