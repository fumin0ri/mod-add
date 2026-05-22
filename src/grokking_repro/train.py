from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .data import make_modular_addition_data
from .model import ModularAdditionTransformer


@dataclass
class TrainConfig:
    seed: int = 0
    modulus: int = 113
    train_fraction: float = 0.3
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    epochs: int = 40000
    log_every: int = 100
    checkpoint_every: int = 1000
    out_dir: str = "runs/mainline"
    device: str = "auto"
    causal: bool = True


def load_config(path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    for key, value in values.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Check that the server has a visible GPU and a CUDA-enabled PyTorch build."
        )
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, tokens: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(tokens)
    loss = F.cross_entropy(logits, labels).item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    data = make_modular_addition_data(
        modulus=cfg.modulus,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
        device=device,
    )
    model = ModularAdditionTransformer(
        modulus=cfg.modulus,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        n_layers=cfg.n_layers,
        causal=cfg.causal,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "weight_norm",
                "seconds",
            ],
        )
        writer.writeheader()

    start = perf_counter()
    for epoch in range(cfg.epochs + 1):
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            train_loss, train_acc = evaluate(model, data.train_tokens, data.train_labels)
            test_loss, test_acc = evaluate(model, data.test_tokens, data.test_labels)
            weight_norm = sum(p.detach().pow(2).sum().item() for p in model.parameters())
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "weight_norm": weight_norm,
                "seconds": perf_counter() - start,
            }
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)
            print(
                f"epoch={epoch:05d} train_loss={train_loss:.6f} "
                f"train_acc={train_acc:.4f} test_loss={test_loss:.6f} "
                f"test_acc={test_acc:.4f} weight_norm={weight_norm:.2f}",
                flush=True,
            )

        if epoch > 0 and epoch % cfg.checkpoint_every == 0:
            save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch:05d}.pt", model, optimizer, cfg, epoch)

        if epoch == cfg.epochs:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.train_tokens)
        loss = F.cross_entropy(logits, data.train_labels)
        loss.backward()
        optimizer.step()

    save_checkpoint(out_dir / "checkpoints" / "final.pt", model, optimizer, cfg, cfg.epochs)


if __name__ == "__main__":
    main()
