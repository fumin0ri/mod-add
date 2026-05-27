from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def plot_heatmap(
    tensor: torch.Tensor,
    title: str,
    out_path: Path,
) -> None:
    data = (tensor.detach().cpu() != 0).to(torch.float32)
    fig_width = max(6.0, min(24.0, data.shape[1] / 96))
    fig_height = max(2.0, min(24.0, data.shape[0] / 96))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.imshow(data, aspect="auto", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("input / column")
    ax.set_ylabel("output / row")
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
    parser.add_argument("--include-norm", action="store_true")
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
        if data.ndim == 1:
            plotted = data.unsqueeze(0)
        else:
            plotted = data

        safe_name = sanitize_filename(name)
        out_path = out_dir / f"{safe_name}.png"
        plot_heatmap(
            plotted,
            title=f"{name} nonzero mask shape={tuple(tensor.shape)}",
            out_path=out_path,
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
