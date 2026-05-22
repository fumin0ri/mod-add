from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse"],
        default="sparse",
        help="Which training entrypoint to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/circuit_sparse_mainline.json",
        help="Base config passed to the selected training script.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seeds to run, for example: --seeds 0 1 2 3 4",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Directory containing one run directory per seed.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create curves.png for each seed after training.",
    )
    parser.add_argument(
        "--fourier",
        action="store_true",
        help="Run Fourier embedding analysis for each seed after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_module = {
        "dense": "grokking_repro.train",
        "sparse": "grokking_repro.train_sparse",
    }[args.mode]

    if args.out_root is not None:
        out_root = Path(args.out_root)
    else:
        out_root = Path("runs") / f"{args.mode}_seeds"

    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        out_dir = out_root / f"seed_{seed:03d}"
        cmd = [
            sys.executable,
            "-m",
            train_module,
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--out-dir",
            str(out_dir),
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.device is not None:
            cmd.extend(["--device", args.device])

        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        if args.plot:
            plot_cmd = [
                sys.executable,
                "-m",
                "grokking_repro.plot",
                str(out_dir / "metrics.csv"),
                "--out",
                str(out_dir / "curves.png"),
            ]
            print("plotting:", " ".join(plot_cmd), flush=True)
            subprocess.run(plot_cmd, check=True)

        if args.fourier:
            fourier_cmd = [
                sys.executable,
                "-m",
                "grokking_repro.fourier",
                str(out_dir / "checkpoints" / "final.pt"),
                "--out",
                str(out_dir / "fourier_embedding.csv"),
            ]
            print("fourier:", " ".join(fourier_cmd), flush=True)
            subprocess.run(fourier_cmd, check=True)


if __name__ == "__main__":
    main()
