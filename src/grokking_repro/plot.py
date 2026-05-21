from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    df = pd.read_csv(metrics_path)

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

    out = Path(args.out) if args.out else metrics_path.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

