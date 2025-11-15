# Foundational Neural Networks: McCulloch-Pitts, Adaline/Madaline, RBM, and MLP

![Project Banner](MLP.png)

## Overview
This project covers foundational neural network implementations as part of Neural Networks & Deep Learning Homework 1 at University of Tehran. We implement McCulloch-Pitts neurons for logic gates, Adaline/Madaline for linear classification, Restricted Boltzmann Machine (RBM) for recommendation systems, and Multi-Layer Perceptron (MLP) for regression. All from scratch in Python, focusing on core algorithms like LMS, backpropagation, and contrastive divergence.

**Key Goal**: Build and evaluate basic NN models on synthetic/real datasets, analyzing convergence (e.g., MSE <0.1), separation accuracy, and prediction errors.

- **Team Members**: Sara Rastegar (810100355), Amin Shahcheraghi (810199196)
- **Duration**: Oct 2022 (Due: 1401/08/05)
- **Technologies**: Python 3.x, NumPy (core computations), Matplotlib (visualizations)
- **Datasets**: Synthetic 2D points (A/B/C for Adaline/Madaline), MovieLens (RBM recommender), King County Houses (MLP regression, ~21K samples)
- **Key Results**: Adaline converges MSE ~0.05 by epoch 100; Madaline (8 neurons) achieves 100% separation; RBM reduces error <0.05; MLP RMSE ~0.2 on scaled test set.

Focus: Theoretical foundations, training dynamics, and analysis per [HW1 Assignment](path/to/NNDL-HW1.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [McCulloch-Pitts Neuron](#mcculloch-pitts-neuron)
- [Adaline & Madaline](#adaline--madaline)
- [Restricted Boltzmann Machine](#restricted-boltzmann-machine)
- [Multi-Layer Perceptron](#multi-layer-perceptron)
- [Results & Evaluation](#results--evaluation)
<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Project Structure
nndl-hw1\
├── data\
│   ├── synthetic_2d_a.csv     # Adaline dataset A\
│   ├── synthetic_2d_c.csv     # Adaline/Madaline dataset C\
│   ├── movielens_ratings.csv  # RBM movie ratings\
│   └── kc_house_data.csv      # MLP house prices (preprocessed)\
│   └── generated_samples/     # Outputs (e.g., separation plots)\
├── notebooks\
│   ├── 01_mcp_neuron.ipynb    # Logic gate simulatio\
│   ├── 02_adaline_madaline.ipynb # Linear classifiers\
│   ├── 03_rbm_recommender.ipynb # Unsupervised learning\
│   └── 04_mlp_regression.ipynb # Supervised MLP\
├── src\
│   ├── mcp_neuron.py          # Threshold logic functions\
│   ├── adaline.py             # LMS rule implementation\
│   ├── madaline.py            # Multi-Adaline with backprop\
│   ├── rbm.py                 # Contrastive divergence training\
│   ├── mlp.py                 # Feedforward with sigmoid/softmax\
│   ├── train.py               # Unified training loops\
│   └── evaluate.py            # MSE/loss plotting & predictions\
├── figures                   # Plots & diagrams\
│   ├── mcp                   # Logic truth tables (Fig 1-2)\
│   ├── adaline               # Separation lines & MSE curves (Fig 3-8)\
│   ├── madaline              # Neuron configs & losses (Fig 9-16)\
│   ├── rbm                   # Error per epoch (Fig 17-18)\
│   └── mlp                   # EDA & predictions (Fig 19-27)\
├── requirements.txt\
└── README.md


## McCulloch-Pitts Neuron
Binary threshold neuron model for logic simulation (Q1).

- **Implementation**: Step function activation; weights/thresholds for AND/OR/NOT to build binary multiplier (double precision via multi-bit).
- **Evaluation**: Simulated all input states (Fig 1-2); accurate outputs for 4-bit multiplication (e.g., 1010 * 1100 = correct binary).
- **Analysis**: Handles linear separability; limitations on XOR shown via failed separation.

## Adaline & Madaline
Adaptive linear neuron (Q2) for classification.

- **Adaline**:
  - LMS rule: w_new = w_old + α*(target - output)*x; α=0.01, epochs=100.
  - Datasets A/B (linearly separable): MSE <0.1 by epoch 50 (Fig 3-5); decision boundary visualization (Fig 4).
  - Dataset C (non-linear): Partial separation, MSE ~0.2 plateau (Fig 6-7).

- **Madaline**:
  - Multi-layer Adaline: 3/4/8 hidden neurons, backprop for weights.
  - Training: Gradient descent on MSE; converges loss ~0 by epoch 50 (Fig 11-15).
  - Evaluation: 100% accuracy on C with 8 neurons (Fig 16); input labeling +1/-1 (Fig 9-10).

## Restricted Boltzmann Machine
Unsupervised energy-based model for recommendation (Q3).

- **Implementation**: Binary visible/hidden units; contrastive divergence (CD-1) for training.
  - Energy: -b_v^T v - b_h^T h - v^T W h; Gibbs sampling for updates.
- **Dataset**: MovieLens subset; hidden=10 units, epochs=50.
- **Evaluation**: Reconstruction error drops <0.05 (Fig 18); per-movie error plot (Fig 17); top-N recommendations via hidden probs.

## Multi-Layer Perceptron
Supervised feedforward NN for regression (Q4).

- **Architecture**: Input (21 feats) → Hidden (10 ReLU) → Output (1 linear); sigmoid for binary if needed.
  - Backprop: Chain rule for deltas; Adam-like updates (lr=0.01).
- **Preprocessing**: King County houses—handle NaNs (Fig 20), correlation matrix (Fig 21-22), histograms/scatters (Fig 23-25), date split (Fig 25), 80/20 train/test (Fig 26), MinMax scaling (Fig 27).
- **Training**: 100 epochs; MSE loss monitored.
- **Evaluation**: Test RMSE ~20K (raw prices); feature importance via weights.

## Results & Evaluation
Key metrics across models:

| Model       | Dataset/Task          | Convergence (Epochs) | Final Metric       | Key Insight                  |
|-------------|-----------------------|----------------------|--------------------|------------------------------|
| McCulloch-Pitts | Binary Multiplier    | N/A (Simulation)    | 100% Logic Acc.   | Exact for linear logic      |
| Adaline    | 2D Separation (A/C)  | 50-100              | MSE <0.1 / 0.2    | Good for linear; struggles non-linear |
| Madaline   | Non-Linear (C, 8 neu)| 50                  | 100% Acc., Loss=0 | Layers enable XOR-like sep. |
| RBM        | Movie Recommender    | 50                  | Error <0.05       | Good hidden representations|
| MLP        | House Prices         | 100                 | RMSE ~0.2 (scaled)| Strong on correlated feats. |

- Visuals: All figures from report (e.g., Fig 1-27) in `figures/`; no overfitting observed.


## Challenges & Learnings
- **Challenges**: Non-linear separability in single Adaline (fixed via Madaline layers); NaN handling in houses data; CD-1 approximation in RBM.
- **Learnings**: Threshold models foundational for logic; gradient methods scale to multi-layer; preprocessing critical for regression (e.g., scaling prevents explosion).

## Future Work
- Extend Madaline to deeper nets with dropout.
- Use RBM as pre-trainer for MLP (DBN hybrid).
- Apply MLP to classification with cross-entropy.
- Integrate with later projects (e.g., GAN initialization).

## References
- Rosenblatt, F. (1962). *Principles of Neurodynamics*. Spartan Books. (Adaline)
- Widrow, B., & Hoff, M. (1960). *Adaptive Switching Circuits*. IRE WESCON. (LMS)
- Hinton, G. E. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*. Neural Computation. (RBM)
- Rumelhart, D. E., et al. (1986). *Learning Representations by Back-propagating Errors*. Nature. (MLP)
- [HW1 Assignment](path/to/NNDL-HW1.pdf) – University of Tehran, NNDL Course.

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [HW1_Rostami_810100355_Shahcheraghi_810199196.pdf](path/to/HW1_Rostami_810100355_Shahcheraghi_810199196.pdf)  
*Questions?* Open an issue or contact [your-email@example.com].