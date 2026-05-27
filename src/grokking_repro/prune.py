from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import make_modular_addition_data
from .model import ModularAdditionTransformer
from .train import evaluate, resolve_device


class ActivationMeanCollector:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    @torch.no_grad()
    def __call__(self, location: str, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).float().cpu()
        if location not in self.sums:
            self.sums[location] = flat.sum(dim=0)
            self.counts[location] = flat.shape[0]
        else:
            self.sums[location] += flat.sum(dim=0)
            self.counts[location] += flat.shape[0]

    def means(self) -> dict[str, torch.Tensor]:
        return {name: total / self.counts[name] for name, total in self.sums.items()}


class MeanAblationMasker:
    def __init__(
        self,
        means: dict[str, torch.Tensor],
        *,
        init_bias: float,
        init_noise: float,
        temperature: float,
        device: torch.device,
    ) -> None:
        self.means = {name: mean.to(device) for name, mean in means.items()}
        self.temperature = temperature
        self.logits = {
            name: torch.nn.Parameter(init_bias + init_noise * torch.randn_like(mean, device=device))
            for name, mean in self.means.items()
        }

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.logits.values())

    def hard_mask(self, location: str) -> torch.Tensor:
        return (self.logits[location] > 0).float()

    def ste_mask(self, location: str) -> torch.Tensor:
        logits = self.logits[location]
        soft = torch.sigmoid(logits / self.temperature)
        hard = (logits > 0).float()
        return hard + soft - soft.detach()

    def __call__(self, location: str, x: torch.Tensor) -> torch.Tensor:
        if location not in self.logits:
            return x
        mask = self.ste_mask(location).to(dtype=x.dtype, device=x.device)
        mean = self.means[location].to(dtype=x.dtype, device=x.device)
        view_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
        mask = mask.view(*view_shape)
        mean = mean.view(*view_shape)
        return mean + mask * (x - mean)

    def active_count(self) -> torch.Tensor:
        return sum((logits > 0).float().sum() for logits in self.logits.values())

    def soft_active_count(self) -> torch.Tensor:
        return sum(torch.sigmoid(logits / self.temperature).sum() for logits in self.logits.values())

    def total_count(self) -> int:
        return sum(logits.numel() for logits in self.logits.values())

    def state_dict(self) -> dict:
        return {
            "means": {name: mean.detach().cpu() for name, mean in self.means.items()},
            "logits": {name: logits.detach().cpu() for name, logits in self.logits.items()},
            "temperature": self.temperature,
        }


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> ModularAdditionTransformer:
    cfg = checkpoint["config"]
    model = ModularAdditionTransformer(
        modulus=cfg["modulus"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_head=cfg.get("d_head"),
        d_mlp=cfg["d_mlp"],
        n_layers=cfg["n_layers"],
        causal=cfg.get("causal", True),
        activation_type=cfg.get("activation_type", "relu"),
        activation_keep_fraction=cfg.get("activation_keep_fraction"),
        activation_sparsity_locations=cfg.get("activation_sparsity_locations", ""),
        rms_norm=cfg.get("rms_norm", False),
        use_pos_embed=cfg.get("use_pos_embed", True),
        attention_sink=cfg.get("attention_sink", False),
        bigram_table=cfg.get("bigram_table", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def select_split(data, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        return data.train_tokens, data.train_labels
    if split == "test":
        return data.test_tokens, data.test_labels
    if split == "all":
        return data.all_tokens, data.all_labels
    raise ValueError(f"Unknown split: {split}")


def iter_batches(tokens: torch.Tensor, labels: torch.Tensor, batch_size: int):
    if batch_size <= 0 or batch_size >= tokens.shape[0]:
        yield tokens, labels
        return
    for start in range(0, tokens.shape[0], batch_size):
        end = start + batch_size
        yield tokens[start:end], labels[start:end]


@torch.no_grad()
def collect_means(
    model: ModularAdditionTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    collector = ActivationMeanCollector()
    model.activation_sparsifier.capture = collector
    model.activation_sparsifier.mean_ablation = None
    for batch_tokens, _ in iter_batches(tokens, labels, batch_size):
        model(batch_tokens)
    model.activation_sparsifier.capture = None
    return collector.means()


def evaluate_with_mask(
    model: ModularAdditionTransformer,
    masker: MeanAblationMasker,
    data,
) -> dict[str, float]:
    previous = model.activation_sparsifier.mean_ablation
    model.activation_sparsifier.mean_ablation = masker
    train_loss, train_acc = evaluate(model, data.train_tokens, data.train_labels)
    test_loss, test_acc = evaluate(model, data.test_tokens, data.test_labels)
    model.activation_sparsifier.mean_ablation = previous
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prune-split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--mean-split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--k-coef", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--init-bias", type=float, default=1.0)
    parser.add_argument("--init-noise", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent.parent / "pruning"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_checkpoint(checkpoint, device)
    data = make_modular_addition_data(
        modulus=cfg["modulus"],
        train_fraction=cfg["train_fraction"],
        seed=cfg["seed"],
        device=device,
    )
    mean_tokens, mean_labels = select_split(data, args.mean_split)
    prune_tokens, prune_labels = select_split(data, args.prune_split)

    means = collect_means(model, mean_tokens, mean_labels, args.batch_size)
    masker = MeanAblationMasker(
        means,
        init_bias=args.init_bias,
        init_noise=args.init_noise,
        temperature=args.temperature,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        masker.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics_path = out_dir / "metrics.csv"
    fieldnames = [
        "step",
        "loss",
        "task_loss",
        "k_loss",
        "active_count",
        "active_fraction",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
    ]
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    model.activation_sparsifier.mean_ablation = masker
    for step in range(args.steps + 1):
        if args.batch_size > 0 and args.batch_size < prune_tokens.shape[0]:
            indices = torch.randint(0, prune_tokens.shape[0], (args.batch_size,), device=device)
            batch_tokens = prune_tokens[indices]
            batch_labels = prune_labels[indices]
        else:
            batch_tokens = prune_tokens
            batch_labels = prune_labels

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_tokens)
        task_loss = F.cross_entropy(logits, batch_labels)
        k_loss = masker.soft_active_count() / masker.total_count()
        loss = task_loss + args.k_coef * k_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(masker.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            eval_metrics = evaluate_with_mask(model, masker, data)
            active_count = masker.active_count().item()
            row = {
                "step": step,
                "loss": loss.item(),
                "task_loss": task_loss.item(),
                "k_loss": k_loss.item(),
                "active_count": active_count,
                "active_fraction": active_count / masker.total_count(),
                **eval_metrics,
            }
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            print(
                f"step={step:05d} task_loss={row['task_loss']:.6f} "
                f"active={row['active_count']:.0f}/{masker.total_count()} "
                f"test_loss={row['test_loss']:.6f} test_acc={row['test_acc']:.4f}",
                flush=True,
            )

    model.activation_sparsifier.mean_ablation = None
    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "model_config": cfg,
            "prune_args": vars(args),
            "masker": masker.state_dict(),
            "total_nodes": masker.total_count(),
            "active_nodes": int(masker.active_count().item()),
        },
        out_dir / "prune_masks.pt",
    )
    with open(out_dir / "prune_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved {out_dir / 'prune_masks.pt'}")


if __name__ == "__main__":
    main()
