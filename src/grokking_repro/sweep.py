from __future__ import annotations

import argparse
import itertools
import json
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
    parser.add_argument("--d-models", type=int, nargs="+", default=None)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--weight-keep-fractions", type=float, nargs="+", default=None)
    parser.add_argument(
        "--no-auto-architecture",
        action="store_true",
        help=(
            "Do not update d_mlp=4*d_model and n_heads=d_model/d_head when "
            "--d-models is used."
        ),
    )
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


def safe_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def values_or_default(values: list | None) -> list:
    return values if values is not None else [None]


def make_run_name(seed: int, overrides: dict) -> str:
    parts = [f"seed_{seed:03d}"]
    if "d_model" in overrides:
        parts.append(f"dmodel_{safe_value(overrides['d_model'])}")
    if "learning_rate" in overrides:
        parts.append(f"lr_{safe_value(overrides['learning_rate'])}")
    if "weight_keep_fraction" in overrides:
        parts.append(f"wkeep_{safe_value(overrides['weight_keep_fraction'])}")
    return "__".join(parts)


def apply_architecture_defaults(config: dict, d_model: int | None, auto_architecture: bool) -> None:
    if d_model is None:
        return
    config["d_model"] = d_model
    if not auto_architecture:
        return
    config["d_mlp"] = 4 * d_model
    d_head = config.get("d_head")
    if d_head is not None:
        if d_model % d_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by d_head={d_head}.")
        config["n_heads"] = d_model // d_head


def main() -> None:
    args = parse_args()
    train_module = {
        "dense": "grokking_repro.train",
        "sparse": "grokking_repro.train_sparse",
    }[args.mode]
    base_config_path = Path(args.config)
    base_config = load_json(base_config_path)

    if args.out_root is not None:
        out_root = Path(args.out_root)
    else:
        out_root = Path("runs") / f"{args.mode}_seeds"

    out_root.mkdir(parents=True, exist_ok=True)

    grid = list(
        itertools.product(
            args.seeds,
            values_or_default(args.d_models),
            values_or_default(args.learning_rates),
            values_or_default(args.weight_keep_fractions),
        )
    )
    print(f"scheduled_runs={len(grid)}", flush=True)

    for seed, d_model, learning_rate, weight_keep_fraction in grid:
        overrides = {}
        if d_model is not None:
            overrides["d_model"] = d_model
        if learning_rate is not None:
            overrides["learning_rate"] = learning_rate
        if weight_keep_fraction is not None:
            overrides["weight_keep_fraction"] = weight_keep_fraction

        out_dir = out_root / make_run_name(seed, overrides)
        run_config = dict(base_config)
        run_config["seed"] = seed
        run_config["out_dir"] = str(out_dir)
        apply_architecture_defaults(
            run_config,
            d_model,
            auto_architecture=not args.no_auto_architecture,
        )
        if learning_rate is not None:
            run_config["learning_rate"] = learning_rate
        if weight_keep_fraction is not None:
            run_config["weight_keep_fraction"] = weight_keep_fraction
        if args.epochs is not None:
            run_config["epochs"] = args.epochs
        if args.device is not None:
            run_config["device"] = args.device

        run_config_path = out_dir / "sweep_config.json"
        write_json(run_config_path, run_config)

        cmd = [
            sys.executable,
            "-m",
            train_module,
            "--config",
            str(run_config_path),
        ]

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
            fourier_csv = out_dir / "fourier_embedding.csv"
            fourier_cmd = [
                sys.executable,
                "-m",
                "grokking_repro.fourier",
                str(out_dir / "checkpoints" / "final.pt"),
                "--out",
                str(fourier_csv),
            ]
            print("fourier:", " ".join(fourier_cmd), flush=True)
            subprocess.run(fourier_cmd, check=True)

            if args.plot:
                fourier_plot_cmd = [
                    sys.executable,
                    "-m",
                    "grokking_repro.plot",
                    str(fourier_csv),
                    "--kind",
                    "fourier",
                    "--out",
                    str(out_dir / "fourier_embedding.png"),
                ]
                print("plotting fourier:", " ".join(fourier_plot_cmd), flush=True)
                subprocess.run(fourier_plot_cmd, check=True)


if __name__ == "__main__":
    main()
