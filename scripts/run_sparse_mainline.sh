#!/usr/bin/env bash
set -euo pipefail

python -m grokking_repro.train_sparse --config configs/circuit_sparse_mainline.json
python -m grokking_repro.plot runs/circuit_sparse_mainline/metrics.csv --out runs/circuit_sparse_mainline/curves.png
python -m grokking_repro.fourier runs/circuit_sparse_mainline/checkpoints/final.pt --out runs/circuit_sparse_mainline/fourier_embedding.csv

