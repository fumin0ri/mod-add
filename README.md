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
```

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
