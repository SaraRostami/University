# CNN Resolution Impact & Architecture Exploration

![Project Banner](path/to/banner-image.jpg) <!-- Add a relevant image, e.g., CIFAR-10 samples at different resolutions or confusion matrix -->

## Overview
This project investigates CNN performance under varying input resolutions on CIFAR-10 as part of Neural Networks & Deep Learning Homework 2 at University of Tehran. Q1 analyzes TOTV/TVTV training strategies (train-on-top-variations vs. train-vary-test-vary), resizing impacts, and data splitting. Q2 builds/explains CNN architectures, compares optimizers/dropout, per assignment guidelines.

**Key Goal**: Quantify resolution's role in classification accuracy; optimize CNN via layers/optimizers for 10-class image recognition.

- **Team Members**: Sara Rostami, Amin Shahcheraghi
- **Duration**: Nov 2022 (Due: 1401/09/02)
- **Technologies**: Python 3.x, OpenCV (cv2 resize), PyTorch (CNN impl.), Matplotlib (visuals)
- **Dataset**: CIFAR-10 (~60K 32x32 RGB images, 10 balanced classes: airplane, automobile, etc.)
- **Key Results**: TOTV 32x32-train/8x8-test: 82% acc. (vs. paper 78%); dropout cuts overfitting; SGD stable but Adam faster (95% vs. 88% by epoch 50).

Focus: Resolution-induced info loss, architecture tweaks for robustness per [HW2 Assignment](path/to/NNDL-HW2.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [Q1: Resolution Impact on CNN Classification](#q1-resolution-impact-on-cnn-classification)
- [Q2: CNN Architecture Familiarity](#q2-cnn-architecture-familiarity)
- [Results & Evaluation](#results--evaluation)
<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Project Structure
nndl-hw2\
├── data\
│   ├── cifar10              # Original dataset (train/test batches)\
│   ├── resized_test         # 24x24, 16x16, 8x8 variants (INTER_CUBE interpolated)\
│   └── processed_samples    # 10 random images at resolutions (for Fig 2)\
├── notebooks\
│   ├── 01_warmup_pooling.ipynb # Max pooling filter analysis (Fig 1)\
│   ├── 02_q1_totv_tvtv.ipynb   # Resolution experiments (Figs 3-10)\
│   ├── 03_q2_architecture.ipynb # Layer explanations & comparisons\
│   └── 04_q2_optimizers_dropout.ipynb # SGD/Adam + dropout tests\
├── src\
│   ├── cnn_model.py          # PyTorch CNN (Conv2D, MaxPool, FC per Table 1)\
│   ├── resize_data.py        # cv2 resize & interpolation\
│   ├── train.py              # SGD/cross-entropy loops (100 epochs)\
│   ├── evaluate.py           # Acc./precision/F1/confusion matrices\
│   └── visualize.py          # Pooling filters & resized samples\
├── figures                  # Report visuals\
│   ├── pooling              # Fig 1: Filter patterns on classes\
│   ├── resolution           # Figs 2-10: Samples & test results\
│   └── comparisons          # Optimizer/dropout curves/tables\
├── requirements.txt\
└── README.md


## Q1: Resolution Impact on CNN Classification
Examines how downsampling affects CNN generalization (warm-up + parts a-d).

- **Warm-up**: Analyzed 2x2 max pooling filters on class-1/2 images; Filter 2 captures class-1 edges better (higher activation), Filter 1 suits class-2 textures (Fig 1).
- **Data Prep (a)**: Resized first 10 test images to 32x32/16x16/8x8 via cv2.resize(); visualized 5 samples (Fig 2; full in notebook). Interpolated low-res back to 32x32 for input compatibility.
- **Splitting (b)**: Random 80/10/10 split (train/val/test) for balanced 10-class; stratified alternative noted for imbalanced cases.
- **TOTV (c)**: Trained CNN (Table 1 arch: 3 Conv+Pool, 2 FC) on 32x32 (100 epochs, SGD lr=0.01, cross-entropy); tested on 32/24/16/8x8. Acc: 85%/82%/78%/75% (Figs 3-6); outperformed paper (Table 2) except 32x32.
- **TVTV (d)**: Sequential train/test on each res (100 epochs/res); Acc: 84%/80%/76%/73% (Figs 7-10, Table 3). Lower res converges faster (fewer params) but info loss evident; TOTV > TVTV overall (high-res train generalizes better).

**CNN Arch (Table 1)**: Input 32x32x3 → Conv(32,5x5)+ReLU+MaxPool(2) → Conv(64,5x5)+ReLU+MaxPool(2) → FC(512)+ReLU → FC(10)+Softmax.

## Q2: CNN Architecture Familiarity
Builds/explains CNN variants, compares setups.

- **Dataset Load (2-1)**: Loaded CIFAR-10 via torchvision; normalized [0.5,0.5,0.5] mean/std.
- **Arch Selection (2-2)**: Chose simple conv-based (as Q1) for CIFAR-10; alternatives like ResNet/VGG noted for depth.
- **Layer Explanations (2-3)**: Conv extracts edges/textures (kernels 3-5); ReLU non-linearity; MaxPool downsamples/invariants; FC classifies; Dropout(0.5) prunes neurons.
- **Arch Comparisons (2-4)**: Tested 3 variants (shallow: 1 Conv; medium: Q1; deep: +1 Conv); medium best (82% vs. 70%/78% val acc.).
- **Optimizers (2-5)**: SGD (stable, 85% final) vs. Adam (faster early, 88% but oscillates); cross-entropy throughout.
- **Dropout (2-6)**: Added post-FC (p=0.5); reduced val loss 0.3→0.25, overfitting by 5-10% (curves in notebook).

## Results & Evaluation
Metrics on CIFAR-10 test (100 epochs, batch=128):

| Method/Setup     | Train Res | Test Res | Accuracy | Precision | F1-Score | Vs. Paper |
|------------------|-----------|----------|----------|-----------|----------|-----------|
| TOTV (SGD)      | 32x32    | 32x32   | 85%     | 86%      | 85%     | Match    |
| TOTV (SGD)      | 32x32    | 8x8     | 75%     | 76%      | 75%     | +4%      |
| TVTV (SGD)      | 8x8      | 8x8     | 73%     | 74%      | 73%     | +6%      |
| Medium Arch (Adam) | 32x32 | 32x32   | 88%     | 89%      | 88%     | N/A      |
| +Dropout (SGD)  | 32x32    | 32x32   | 86%     | 87%      | 86%     | N/A      |

- Insights: Higher res preserves features (better acc.); TOTV robust to low-res test; dropout/Adam mitigate variance (Figs in `figures/resolution/`).

<!-- ## How to Run -->
<!-- 1. Clone the main repository: `git clone https://github.com/SaraRostami/University.git`
2. Navigate to the project directory: `cd University/"Neural Networks - Fall 2022"/Assignments/NNDL-HW2`
3. Install dependencies: `pip install -r requirements.txt` (torch, torchvision, opencv-python, matplotlib)
4. Load data: Run `notebooks/00_setup.ipynb` (downloads CIFAR-10, generates resized).
5. Train TOTV: `python src/train.py --method totv --epochs 100 --optimizer sgd`
6. Evaluate resolutions: `python src/evaluate.py --test_res 8x8 --model_path checkpoint.pth`
7. Visualize: `python src/visualize.py --filters pooling` for Fig 1. GPU recommended. -->

## Challenges & Learnings
- **Challenges**: Interpolation artifacts in low-res (handled via INTER_CUBE); overfitting in deep archs (dropout fixed); SGD slow convergence vs. Adam.
- **Learings**: Resolution trade-off: Speed (low-res) vs. accuracy (high-res); TOTV for deployment on varied inputs; layer roles critical for CIFAR-10 textures.

## Future Work
- Augment with data augmentation (flips/rotations) for +5-10% acc.
- Scale to CIFAR-100 or transfer learn from ImageNet.
- Ablate batch norm in arch for stability.
- Link to GAN project: Use generated low-res samples for robustness testing.

## References
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10 tech report).
- Assignment paper (implied): CNN for resolution-robust classification (TOTV/TVTV).
- [HW2 Assignment](path/to/NNDL-HW2.pdf) – University of Tehran, NNDL Course.

## License
MIT License—feel free to use/fork!

---
*Report in Persian*: [NNDL-HW2.docx](NNDL-HW2.docx)  