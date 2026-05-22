from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def infer_kind(df: pd.DataFrame) -> str:
    if {"epoch", "train_acc", "test_acc", "train_loss", "test_loss"}.issubset(df.columns):
        return "metrics"
    if {"frequency", "component", "l2_norm_over_d_model"}.issubset(df.columns):
        return "fourier"
    raise ValueError(f"Could not infer plot kind from columns: {list(df.columns)}")


def plot_metrics(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(df["epoch"], df["train_acc"], label="train")
    axes[0].plot(df["epoch"], df["test_acc"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_loss"], label="train")
    axes[1].plot(df["epoch"], df["test_loss"], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    return fig


def plot_fourier(df: pd.DataFrame) -> plt.Figure:
    trig = df[df["component"].isin(["cos", "sin"])].copy()
    trig["frequency"] = trig["frequency"].astype(int)
    pivot = (
        trig.pivot_table(
            index="frequency",
            columns="component",
            values="l2_norm_over_d_model",
            aggfunc="first",
        )
        .fillna(0.0)
        .sort_index()
    )

    for column in ["cos", "sin"]:
        if column not in pivot:
            pivot[column] = 0.0

    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    x = pivot.index.to_numpy()
    width = 0.42
    ax.bar(x - width / 2, pivot["cos"], width=width, label="cos")
    ax.bar(x + width / 2, pivot["sin"], width=width, label="sin")

    ax.set_xlabel("Fourier frequency")
    ax.set_ylabel("L2 norm over model dimension")
    ax.set_title("Embedding Fourier component norms")
    ax.legend()
    ax.set_xlim(0, x.max() + 1 if len(x) else 1)
    if len(x) > 30:
        tick_step = 5
        ax.set_xticks([int(v) for v in x if int(v) % tick_step == 0 or int(v) == 1])
    else:
        ax.set_xticks(x)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--kind", choices=["auto", "metrics", "fourier"], default="auto")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    kind = infer_kind(df) if args.kind == "auto" else args.kind
    fig = plot_metrics(df) if kind == "metrics" else plot_fourier(df)

    out = Path(args.out) if args.out else csv_path.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
