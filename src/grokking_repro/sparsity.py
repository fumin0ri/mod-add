from __future__ import annotations

from dataclasses import dataclass

import torch


def linear_schedule(
    step: int,
    total_steps: int,
    initial: float,
    final: float,
    start_frac: float,
    stop_frac: float,
) -> float:
    start = int(total_steps * start_frac)
    stop = int(total_steps * stop_frac)
    if step <= start:
        return initial
    if step >= stop:
        return final
    progress = (step - start) / max(1, stop - start)
    return initial + (final - initial) * progress


@dataclass
class WeightSparsityStats:
    alive: int
    total: int

    @property
    def alive_fraction(self) -> float:
        if self.total == 0:
            return 0.0
        return self.alive / self.total


@torch.no_grad()
def apply_weight_topk_(
    model: torch.nn.Module,
    keep_fraction: float | None,
    *,
    include_bias: bool = True,
    minimum_alive_per_row: int = 0,
) -> WeightSparsityStats:
    if keep_fraction is None or keep_fraction >= 1.0:
        total = sum(p.numel() for _, p in sparsifiable_parameters(model, include_bias=include_bias))
        return WeightSparsityStats(alive=total, total=total)

    if keep_fraction < 0.0:
        raise ValueError("keep_fraction must be non-negative.")

    alive = 0
    total = 0
    for _, param in sparsifiable_parameters(model, include_bias=include_bias):
        flat = param.data.flatten()
        total += flat.numel()
        k = max(1, int(keep_fraction * flat.numel()))
        if k >= flat.numel():
            alive += flat.numel()
            continue
        scores = flat.abs()
        if param.ndim == 2 and minimum_alive_per_row > 0:
            per_row_k = min(minimum_alive_per_row, param.shape[1])
            min_alive_total = per_row_k * param.shape[0]
            if min_alive_total <= k:
                row_scores = param.data.abs()
                _, row_indices = torch.topk(row_scores, per_row_k, dim=1, sorted=False)
                row_offsets = torch.arange(param.shape[0], device=param.device).unsqueeze(1) * param.shape[1]
                protected = (row_indices + row_offsets).flatten()
                scores = scores.clone()
                scores[protected] = torch.inf
        _, indices = torch.topk(scores, k, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.index_fill_(0, indices, True)
        flat.masked_fill_(~mask, 0.0)
        alive += k
    return WeightSparsityStats(alive=alive, total=total)


def sparsifiable_parameters(
    model: torch.nn.Module,
    *,
    include_bias: bool,
) -> list[tuple[str, torch.nn.Parameter]]:
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("pos_embed"):
            continue
        if name.endswith("bigram_table"):
            continue
        if "ln_" in name:
            continue
        if not include_bias and name.endswith(".bias"):
            continue
        if param.ndim < 2 and not name.endswith(".bias"):
            continue
        params.append((name, param))
    return params


@torch.no_grad()
def apply_decoupled_weight_decay_(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> None:
    if weight_decay == 0.0:
        return
    for _, param in sparsifiable_parameters(model, include_bias=False):
        if param.ndim > 1:
            param.data.add_(param.data, alpha=-weight_decay * lr)
