from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def downsample_2d(tensor: torch.Tensor, max_bins: int) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError("downsample_2d expects a 2D tensor.")
    rows, cols = tensor.shape
    if rows <= max_bins and cols <= max_bins:
        return tensor
    out_rows = min(rows, max_bins)
    out_cols = min(cols, max_bins)
    x = tensor.float().unsqueeze(0).unsqueeze(0)
    x = F.adaptive_avg_pool2d(x, (out_rows, out_cols))
    return x.squeeze(0).squeeze(0)


def robust_symmetric_limit(tensor: torch.Tensor) -> float:
    flat = tensor.detach().float().flatten()
    if flat.numel() == 0:
        return 1.0
    limit = torch.quantile(flat.abs(), 0.995).item()
    return max(limit, 1e-12)


def plot_heatmap(
    tensor: torch.Tensor,
    title: str,
    out_path: Path,
    *,
    cmap: str,
    symmetric: bool,
) -> None:
    data = tensor.detach().float().cpu()
    fig_width = max(6.0, min(14.0, data.shape[1] / 64))
    fig_height = max(2.0, min(10.0, data.shape[0] / 64))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    if symmetric:
        limit = robust_symmetric_limit(data)
        image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
    else:
        image = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("input / column")
    ax.set_ylabel("output / row")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def tensor_summary(name: str, tensor: torch.Tensor, plotted_shape: tuple[int, ...]) -> dict:
    data = tensor.detach().float().cpu()
    nnz = int((data != 0).sum().item())
    total = data.numel()
    return {
        "name": name,
        "shape": "x".join(str(x) for x in data.shape),
        "plotted_shape": "x".join(str(x) for x in plotted_shape),
        "numel": total,
        "nnz": nnz,
        "nnz_fraction": nnz / total if total else 0.0,
        "min": data.min().item() if total else 0.0,
        "max": data.max().item() if total else 0.0,
        "mean": data.mean().item() if total else 0.0,
        "std": data.std(unbiased=False).item() if total else 0.0,
    }


def should_plot(name: str, tensor: torch.Tensor, include_norm: bool) -> bool:
    if tensor.ndim not in {1, 2}:
        return False
    if not include_norm and ("ln_" in name or "norm" in name):
        return False
    return any(key in name for key in ["weight", "bias", "pos_embed", "sink_logit", "bigram_table"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-bins", type=int, default=512)
    parser.add_argument("--include-norm", action="store_true")
    parser.add_argument(
        "--abs",
        action="store_true",
        help="Plot absolute values instead of signed weights.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent.parent / "weight_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name, tensor in state_dict.items():
        if not should_plot(name, tensor, include_norm=args.include_norm):
            continue

        data = tensor.detach().float().cpu()
        if args.abs:
            data = data.abs()

        if data.ndim == 1:
            plotted = data.unsqueeze(0)
        else:
            plotted = downsample_2d(data, max_bins=args.max_bins)

        safe_name = sanitize_filename(name)
        out_path = out_dir / f"{safe_name}.png"
        plot_heatmap(
            plotted,
            title=f"{name} shape={tuple(tensor.shape)}",
            out_path=out_path,
            cmap="viridis" if args.abs else "coolwarm",
            symmetric=not args.abs,
        )
        summaries.append(tensor_summary(name, tensor, tuple(plotted.shape)))
        print(f"saved {out_path}", flush=True)

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "shape",
            "plotted_shape",
            "numel",
            "nnz",
            "nnz_fraction",
            "min",
            "max",
            "mean",
            "std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()

