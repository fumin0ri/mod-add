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
from .sparsity import apply_decoupled_weight_decay_, apply_weight_topk_, linear_schedule
from .train import evaluate, resolve_device, save_checkpoint


@dataclass
class SparseTrainConfig:
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
    out_dir: str = "runs/circuit_sparse_mainline"
    device: str = "auto"
    causal: bool = True

    # Circuit-sparsity style controls.
    weight_keep_fraction: float | None = 0.25
    initial_weight_keep_fraction: float = 1.0
    anneal_weight_sparsity: bool = True
    anneal_start_frac: float = 0.01
    anneal_stop_frac: float = 0.5
    schedule_lr_with_l0: bool = True
    include_bias_in_weight_sparsity: bool = True
    activation_keep_fraction: float | None = 0.25
    activation_sparsity_locations: str = (
        "attn_in,attn_q,attn_k,attn_v,attn_out,"
        "mlp_in,mlp_neuron,mlp_out,resid_post_attn,resid_post_mlp"
    )


def load_config(path: str | None) -> SparseTrainConfig:
    cfg = SparseTrainConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    for key, value in values.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg


def parse_args() -> SparseTrainConfig:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_keep_fraction(cfg: SparseTrainConfig, epoch: int) -> float | None:
    if cfg.weight_keep_fraction is None:
        return None
    if not cfg.anneal_weight_sparsity:
        return cfg.weight_keep_fraction
    return linear_schedule(
        step=epoch,
        total_steps=cfg.epochs,
        initial=cfg.initial_weight_keep_fraction,
        final=cfg.weight_keep_fraction,
        start_frac=cfg.anneal_start_frac,
        stop_frac=cfg.anneal_stop_frac,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
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
        activation_keep_fraction=cfg.activation_keep_fraction,
        activation_sparsity_locations=cfg.activation_sparsity_locations,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    initial_stats = apply_weight_topk_(
        model,
        current_keep_fraction(cfg, 0),
        include_bias=cfg.include_bias_in_weight_sparsity,
    )

    metrics_path = out_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "weight_norm",
        "weight_keep_fraction",
        "weight_alive_fraction",
        "seconds",
    ]
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    start = perf_counter()
    last_stats = initial_stats
    for epoch in range(cfg.epochs + 1):
        keep_fraction = current_keep_fraction(cfg, epoch)
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
                "weight_keep_fraction": keep_fraction if keep_fraction is not None else 1.0,
                "weight_alive_fraction": last_stats.alive_fraction,
                "seconds": perf_counter() - start,
            }
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            print(
                f"epoch={epoch:05d} train_loss={train_loss:.6f} "
                f"train_acc={train_acc:.4f} test_loss={test_loss:.6f} "
                f"test_acc={test_acc:.4f} keep={row['weight_keep_fraction']:.4f} "
                f"alive={row['weight_alive_fraction']:.4f}",
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

        if cfg.schedule_lr_with_l0 and keep_fraction is not None and keep_fraction > 0:
            scaled_lr = cfg.learning_rate * (cfg.weight_keep_fraction / keep_fraction) ** 0.5
            for group in optimizer.param_groups:
                group["lr"] = scaled_lr
        else:
            scaled_lr = cfg.learning_rate

        optimizer.step()
        apply_decoupled_weight_decay_(model, lr=scaled_lr, weight_decay=cfg.weight_decay)
        last_stats = apply_weight_topk_(
            model,
            keep_fraction,
            include_bias=cfg.include_bias_in_weight_sparsity,
        )

    save_checkpoint(out_dir / "checkpoints" / "final.pt", model, optimizer, cfg, cfg.epochs)


if __name__ == "__main__":
    main()

