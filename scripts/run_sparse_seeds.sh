#!/usr/bin/env bash
set -euo pipefail

python -m grokking_repro.sweep \
  --mode sparse \
  --config configs/circuit_sparse_mainline.json \
  --seeds 0 1 2 3 4 \
  --out-root runs/circuit_sparse_seeds \
  --plot

