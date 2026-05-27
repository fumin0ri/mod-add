# Grokking Modular Addition Reproduction

This is a small PyTorch reproduction scaffold for:

> Progress Measures for Grokking via Mechanistic Interpretability, Nanda et al., ICLR 2023

The first target is the paper's mainline modular addition experiment:

- Task: predict `(a + b) mod p`
- Modulus: `p = 113`
- Train split: 30% of all `113 * 113` input pairs
- Model: 1-layer ReLU transformer, `d_model = 128`, 4 heads, MLP width 512
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1.0`
- Training: full-batch gradient descent, up to 40,000 epochs

## Setup on the server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

Quick smoke test:

```bash
python -m grokking_repro.train --epochs 20 --out-dir runs/smoke
```

This should produce `runs/smoke/metrics.csv` and `runs/smoke/checkpoints/final.pt`.

Paper-like run:

```bash
python -m grokking_repro.train --config configs/mainline.json
```

Plot training curves:

```bash
python -m grokking_repro.plot runs/mainline/metrics.csv --out runs/mainline/curves.png
```

Analyze Fourier structure in embeddings:

```bash
python -m grokking_repro.fourier runs/mainline/checkpoints/final.pt --out runs/mainline/fourier_embedding.csv
python -m grokking_repro.plot runs/mainline/fourier_embedding.csv --kind fourier --out runs/mainline/fourier_embedding.png
```

Visualize trained weights and biases:

```bash
python -m grokking_repro.visualize_weights \
  runs/circuit_sparse_mainline/checkpoints/final.pt \
  --out-dir runs/circuit_sparse_mainline/weight_heatmaps
```

The heatmap shows the nonzero mask directly: zero entries are white and nonzero entries are black. A `summary.csv` with shape, nonzero count, and basic statistics is saved alongside the PNG files.

## Circuit-sparsity style run

This keeps the modular-addition task, but uses the weight-sparse transformer settings from Gao et al. 2025:

- GPT-2 style decoder-only transformer with RMSNorm, GELU MLPs, untied embedding/unembedding, no positional embeddings, and no dense bigram table for the modular-addition task;
- `n_layers = 8`, `d_model = 2048`, `d_head = 16`, `n_heads = 128`, `d_mlp = 8192`;
- AdamW with `beta1 = 0.9`, `beta2 = 0.95`, `weight_decay = 0.1`, `eps = 0.1`;
- linearly anneal weight L0 from dense to the target over the first 50% of training;
- target weight keep fraction `1/64`, corresponding to the paper's `pfrac = 1/4` and expansion factor 4 setup;
- AbsTopK activation sparsity with `activation_keep_fraction = 0.25` on attention/MLP reads, writes, q/k/v, and MLP post-activation. The residual stream itself is not directly top-k sparsified.

Quick smoke test:

```bash
python -m grokking_repro.train_sparse --epochs 20 --out-dir runs/sparse_smoke
```

Paper-length sparse run:

```bash
python -m grokking_repro.train_sparse --config configs/circuit_sparse_mainline.json
```

Plot:

```bash
python -m grokking_repro.plot runs/circuit_sparse_mainline/metrics.csv --out runs/circuit_sparse_mainline/curves.png
```

This is a large model for the tiny modular-addition task. It is intentionally set this way to match the paper's sparse-training regime; use a GPU and expect dense AdamW moments to consume several GB of memory.

## Multiple seeds

Sparse runs over several seeds:

```bash
python -m grokking_repro.sweep \
  --mode sparse \
  --config configs/circuit_sparse_mainline.json \
  --seeds 0 1 2 3 4 \
  --out-root runs/circuit_sparse_seeds \
  --plot \
  --fourier
```

Dense baseline over several seeds:

```bash
python -m grokking_repro.sweep \
  --mode dense \
  --config configs/mainline.json \
  --seeds 0 1 2 3 4 \
  --out-root runs/dense_seeds \
  --plot \
  --fourier
```

For a quick multi-seed smoke test:

```bash
python -m grokking_repro.sweep --mode sparse --seeds 0 1 --epochs 20 --out-root runs/sparse_seed_smoke --plot --fourier
```

Grid sweep over seeds and hyperparameters:

```bash
python -m grokking_repro.sweep \
  --mode sparse \
  --config configs/circuit_sparse_mainline.json \
  --seeds 0 1 2 \
  --d-models 512 1024 2048 \
  --learning-rates 0.0003 0.001 \
  --weight-keep-fractions 0.015625 0.03125 \
  --out-root runs/sparse_grid \
  --device cuda \
  --plot \
  --fourier
```

When `--d-models` is used, the sweep runner updates `d_mlp = 4 * d_model` and, when `d_head` is set, `n_heads = d_model / d_head`. Add `--no-auto-architecture` to disable this behavior.

## Git + SSH workflow

On this PC:

```bash
git init
git add .
git commit -m "Add grokking reproduction scaffold"
git remote add origin <your-repo-url>
git push -u origin main
```

On the lab server:

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m grokking_repro.train --config configs/mainline.json
```
