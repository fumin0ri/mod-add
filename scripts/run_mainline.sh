#!/usr/bin/env bash
set -euo pipefail

python -m grokking_repro.train --config configs/mainline.json
python -m grokking_repro.plot runs/mainline/metrics.csv --out runs/mainline/curves.png
python -m grokking_repro.fourier runs/mainline/checkpoints/final.pt --out runs/mainline/fourier_embedding.csv

