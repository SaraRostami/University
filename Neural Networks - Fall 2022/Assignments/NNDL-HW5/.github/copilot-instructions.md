<!-- Copilot instructions for AI coding agents working in this repo -->
# Copilot Instructions — NNDL-HW5

Purpose: Give short, actionable guidance so AI coding agents can be immediately productive in this homework repository.

- **Primary locations**:
  - Experiment notebooks: `Q1/Q1_transformer_completed.ipynb`, `Q1/transformer.ipynb`, `Q2/2_2_part1.ipynb`, `Q2/2_2_part2.ipynb`, `Q2/2_3_BEiT.ipynb`, `Q2/2_3_MLP.ipynb`.
  - Top-level overview: `README.md` (contains architecture summary, hyperparams and dataset notes).

- **Big picture / architecture**:
  - Q1 implements a BERT-style encoder for text: tokenization → token/pos/segment embeddings → stacked encoder layers (self-attention + FFN) → outputs (hidden states + attention matrices). Key hyperparams: `d_model=768`, `heads=12`, `layers=12` (see `README.md` examples).
  - Q2 contains vision experiments: BEiT/ViT pipeline for segmentation and a simple MLP baseline for CIFAR-10. Flow: image → patching (16×16) → ViT backbone (positional embeddings) → decoder head (segmentation logits) or classification head (MLP). BEiT uses masking + a teacher model for pseudo-labeling per README.

- **What to edit vs. where to experiment**:
  - Treat `*_completed.ipynb` and notebooks in `Q1/` and `Q2/` as canonical experiment records. For reproducible code, prefer adding a Python script (new `src/` file) rather than editing the canonical notebook unless the change is an explanatory note or small fix.
  - If adding utilities (tokenizers, dataset loaders, model classes), create a `src/` package and import those modules from notebooks to keep notebooks lean.

- **Developer workflows / reproducible runs**:
  - Environment: Python 3.x with PyTorch, `transformers` (optional), `torchvision`, `matplotlib`. `README.md` lists these; if `requirements.txt` is missing, run `pip install torch torchvision matplotlib jupyterlab` before experiments.
  - Typical reproducible steps: launch Jupyter (`jupyter lab`), open the target notebook and run top-to-bottom. For heavy experiments (BEiT), use a GPU and small batch sizes (README notes `batch=2` for segmentation fine-tune).
  - Hyperparams/examples are documented in `README.md` (e.g., BEiT fine-tune: `epochs=5`, `lr=1e-5`, `batch=2`; MLP CIFAR-10: `epochs=50`, `batch=128`, `lr=1e-3`). Use these as defaults when reproducing results.

- **Project-specific conventions & patterns**:
  - Notebooks are the main source of truth for experiments; narrative + plotted figures are expected inline.
  - Completed experiments are suffixed with `_completed.ipynb` — prefer referencing those for final results.
  - Figures and evaluation metrics are produced inline; do not remove figure cells. If adding evaluation code, append separate cells so original outputs remain.

- **Integration points & external dependencies**:
  - Datasets: CIFAR-10 is likely loaded via `torchvision.datasets`; ADE20K-like segmentation dataset is referenced in `README.md` (look for `ade` or `ade_sample` strings in notebooks). If data is missing, add a loader that gracefully falls back to a local `data/` path and document it in the notebook.
  - No external servers or APIs are present — experiments run locally (GPU recommended for BEiT).

- **When changing code, follow these rules**:
  1. Preserve authorship and explanatory markdown in notebooks; add new code cells for experiments or port code into `src/` with small, focused modules.
  2. Keep default hyperparams matching `README.md` unless intentionally tuning; list changes in a short header cell describing the experiment.
  3. For large refactors, create unit-testable Python modules and add a minimal example notebook demonstrating the same results.

- **Quick pointers for common tasks**:
  - Reproduce Q1 attention visualizations: open `Q1/Q1_transformer_completed.ipynb` and run all cells; attention matrices and plots are produced inline.
  - Reproduce BEiT segmentation: open `Q2/2_3_BEiT.ipynb` and use GPU; check the top cells for dataset path and `epochs`, `lr`, `batch` values.
  - Run CIFAR-10 baseline: open `Q2/2_3_MLP.ipynb` and run the training cells; use `batch=128` and `epochs=50` as in README.

- **If you need to search for patterns**:
  - Search notebooks for keywords: `attention`, `mask`, `patch`, `ADE`, `cifar`, `epochs`, `batch`, `lr` to quickly find experiment-related cells.

- **Non-goals / avoid**:
  - Do not rewrite entire notebooks to scripts unless requested — prefer incremental, documented refactors that preserve experiments.

If anything above is unclear or you'd like more detail (for example: exact dataset paths in the notebooks, or a `requirements.txt` added), tell me which area to expand and I'll update this file.
