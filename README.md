# Vision Transformer Reimplementation

A from-scratch Vision Transformer implementation built for learning, testing, and controlled CIFAR-10 experiments.

## Project structure

```text
vit-reimplementation/
├── src/vit_reimplementation/
│   ├── model.py       # ViT architecture
│   ├── training.py    # training and evaluation loops
│   └── data.py        # reproducible CIFAR-10 splits and loaders
├── tests/
│   └── test_model.py
├── notebooks/
│   ├── 01_build_vit.ipynb
│   └── 02_cifar10_baseline.ipynb
├── checkpoints/
└── pyproject.toml
```

`01_build_vit.ipynb` preserves the original learning notebook. `02_cifar10_baseline.ipynb` starts a clean experiment using imported modules.

## Setup with uv

```bash
uv sync --group dev
uv run pytest -v
uv run jupyter lab
```

In Jupyter, select the Python kernel from this project's `.venv`.

## Experiment protocol

- Train on 45,000 examples from the official CIFAR-10 training split.
- Validate on a fixed 5,000-example subset of the official training split.
- Use validation loss for early stopping and checkpoint selection.
- Evaluate the official 10,000-example test split once after model selection.
