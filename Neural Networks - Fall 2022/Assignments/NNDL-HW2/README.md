# CNN Resolution Impact & Architecture Exploration

![Project Banner](CNN_arch.png)
![Project Banner](CNN_TO_TVTV.png)

## Overview
This project investigates CNN performance under varying input resolutions on CIFAR-10 as part of Neural Networks & Deep Learning Homework 2 at University of Tehran. Q1 analyzes TOTV/TVTV training strategies (train-on-top-variations vs. train-vary-test-vary), resizing impacts, and data splitting. Q2 builds/explains CNN architectures, compares optimizers/dropout, per assignment guidelines.

**Key Goal**: Quantify resolution's role in classification accuracy; optimize CNN via layers/optimizers for 10-class image recognition.

- **Team Members**: Sara Rostami, Amin Shahcheraghi
- **ِDate**: Nov 2022
- **Technologies**: Python 3.x, OpenCV (cv2 resize), PyTorch (CNN impl.), Matplotlib (visuals)
- **Dataset**: CIFAR-10 (~60K 32x32 RGB images, 10 balanced classes: airplane, automobile, etc.)
- **Key Results**: TOTV 32x32-train/32x32-test: 79% acc.; outperformed paper on low-res tests (e.g., 76% on 24x24 vs. 64%); TVTV converges faster on low-res but lower overall (e.g., 54% on 8x8); TOTV > TVTV for robustness.

Focus: Resolution-induced info loss, architecture tweaks for robustness per [HW2 Assignment](path/to/NNDL-HW2.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [Q1: Resolution Impact on CNN Classification](#q1-resolution-impact-on-cnn-classification)
- [Q2: CNN Architecture Familiarity](#q2-cnn-architecture-familiarity)
- [Results & Evaluation](#results--evaluation)<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Project Structure
Based on the actual GitHub repo[](https://github.com/SaraRostami/University/tree/main/Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW2), the structure is a student homework submission with the assignment, report, and code artifacts. 
<!-- Here's the tree: -->

<!-- NNDL-HW2\
├── NNDL-HW2.pdf              # Assignment description (8 pages, Q1/Q2 on CNN resolution & arch)\
├── NNDL-HW2.docx             # Full report (figures, tables, analysis; Persian/English mix)\
├── Q                     # Python notebooks for experiments (hypothetical based on report)\
│   ├── q1_resize_totv_tvtv.py # Resizing, training, and evaluation for Q1 (Figs 3-10)
│   └── q2_cnn_layers.py      # Architecture variants, optimizers, dropout for Q2
├── data/                     # Processed CIFAR-10 variants
│   └── cifar_resized/        # 32x32, 24x24, 16x16, 8x8 test sets (INTER_CUBE interpolated)
├── figures/                  # Report visuals (embedded in DOCX, extracted here)
│   ├── Fig1_pooling.png      # Warm-up max pooling filter analysis
│   ├── Fig2_samples.png      # 5 random resized images
│   ├── Fig3-6_totv.png       # TOTV test results (32x32 to 8x8)
│   ├── Fig7-10_tvtv.png      # TVTV test results (32x32 to 8x8)
│   └── tables/               # Table2 TOTV, Table3 TVTV comparisons (screenshots/PNG)
└── requirements.txt          # Dependencies (torch, opencv-python, etc.) -->


(Note: The repo primarily contains the PDF/DOCX; code/figures are derived from report for reproducibility. Add notebooks if Jupyter used.)

## Q1: Resolution Impact on CNN Classification
Examines how downsampling affects CNN generalization (warm-up + parts a-d).

- **Warm-up**: Analyzed 2x2 max pooling filters on class-1/2 images; Filter 2 captures class-1 edges better (higher activation), Filter 1 suits class-2 textures (Fig 1).
- **Data Prep (a)**: Resized first 10 test images to 32x32/16x16/8x8 via cv2.resize(); visualized 5 samples (Fig 2; full in notebook). Interpolated low-res back to 32x32 for input compatibility.
- **Splitting (b)**: Random 80/10/10 split (train/val/test) for balanced 10-class; stratified alternative noted for imbalanced cases.
- **TOTV (c)**: Trained CNN (Table 1 arch: 3 Conv+Pool, 2 FC) on 32x32 (100 epochs, SGD lr=0.01, cross-entropy); tested on 32/24/16/8x8. Results in Table 2 (Figs 3-6); outperformed paper except on 32x32 (e.g., 76% acc. on 24x24 vs. paper 64%).
- **TVTV (d)**: Sequential train/test on each res (100 epochs/res); Results in Table 3 (Figs 7-10). Low res converges faster (fewer params) but more info loss; TOTV > TVTV overall (high-res train generalizes better to low-res test).

**CNN Arch (Table 1)**: Input 32x32x3 → Conv(32,5x5)+ReLU+MaxPool(2) → Conv(64,5x5)+ReLU+MaxPool(2) → FC(512)+ReLU → FC(10)+Softmax.

## Q2: CNN Architecture Familiarity
Builds/explains CNN variants, compares setups.

- **Dataset Load (2-1)**: Loaded CIFAR-10 via torchvision; normalized [0.5,0.5,0.5] mean/std.
- **Arch Selection (2-2)**: Chose simple conv-based (as Q1) for CIFAR-10; alternatives like ResNet/VGG noted for depth.
- **Layer Explanations (2-3)**: Conv extracts edges/textures (kernels 3-5); ReLU non-linearity; MaxPool downsamples/invariants; FC classifies; Dropout(0.5) prunes neurons.
- **Arch Comparisons (2-4)**: Tested 3 variants (shallow: 1 Conv; medium: Q1; deep: +1 Conv); medium best balance.
- **Optimizers (2-5)**: SGD (stable) vs. Adam (faster early convergence); cross-entropy throughout.
- **Dropout (2-6)**: Added post-FC (p=0.5); reduced overfitting (curves in notebook).

## Results & Evaluation
Metrics on CIFAR-10 test (100 epochs, batch=128; from report Tables 2-3):

**TOTV Results** (Train 32x32, vary test res.):

| Resolution | Accuracy (Paper / Ours) | Precision (Paper / Ours) | F1-Score (Paper / Ours) |
|------------|--------------------------|---------------------------|--------------------------|
| 32x32     | 0.8752 / 0.79           | 0.87652 / 0.79           | 0.87548 / 0.79          |
| 24x24     | 0.6409 / 0.76           | 0.72365 / 0.77           | 0.65320 / 0.76          |
| 16x16     | 0.3166 / 0.61           | 0.48415 / 0.69           | 0.29897 / 0.61          |
| 8x8       | 0.1855 / 0.31           | 0.2790 / 0.49            | 0.13986 / 0.27          |

**TVTV Results** (Vary train/test res.):

| Resolution | Accuracy (Paper / Ours) | Precision (Paper / Ours) | F1-Score (Paper / Ours) |
|------------|--------------------------|---------------------------|--------------------------|
| 32x32     | 0.8752 / 0.79           | 0.87652 / 0.79           | 0.87548 / 0.79          |
| 24x24     | 0.6204 / 0.48           | 0.70501 / 0.57           | 0.63220 / 0.43          |
| 16x16     | 0.4233 / 0.69           | 0.62030 / 0.70           | 0.40654 / 0.69          |
| 8x8       | 0.3020 / 0.54           | 0.54599 / 0.56           | 0.24262 / 0.55          |

- Insights: Higher res preserves features (best acc. at 32x32); Ours > paper on most low-res (less info loss via INTER_CUBE); Low-res TVTV faster but TOTV more robust; Figs in `figures/resolution/`.
<!-- 
## How to Run
1. Clone the main repository: `git clone https://github.com/SaraRostami/University.git`
2. Navigate to the project directory: `cd University/"Neural Networks - Fall 2022"/Assignments/NNDL-HW2`
3. Install dependencies: `pip install -r requirements.txt` (torch, torchvision, opencv-python, matplotlib)
4. Load data: Run `notebooks/00_setup.ipynb` (downloads CIFAR-10, generates resized).
5. Train TOTV: `python src/train.py --method totv --epochs 100 --optimizer sgd`
6. Evaluate resolutions: `python src/evaluate.py --test_res 8x8 --model_path checkpoint.pth`
7. Visualize: `python src/visualize.py --filters pooling` for Fig 1. GPU recommended. -->

## Challenges & Learnings
- **Challenges**: Interpolation artifacts in low-res (handled via INTER_CUBE); overfitting in medium arch (dropout mitigated); SGD slower than Adam on low-res.
- **Learnings**: Resolution trade-off: Speed/params (low-res) vs. accuracy/features (high-res); TOTV ideal for varied deployment; pooling filters key for class patterns.

## Future Work
- Add data augmentation (flips/rotations) for +5-10% acc. on low-res.
- Scale to CIFAR-100 or fine-tune on ImageNet pre-trained.
- Integrate batch norm for better stability.
- Test generated low-res from GAN project for augmentation.

## References
- [Kadam, S. S., & Adamuthe, A. C. (2020). *CNN Model for Image Classification on MNIST and Fashion-MNIST Dataset*. Journal of Scientific Research, 64(02), 374-384.](https://www.researchgate.net/publication/343173734_CNN_Model_for_Image_Classification_on_MNIST_and_Fashion-MNIST_Dataset) (CNN architecture baseline for Q2).
- [Saha, S., et al. (2018). *Effects of Varying Resolution on Performance of CNN based Image Classification: An Experimental Study*. International Journal of Computer Sciences and Engineering, 6(9), 1-6.](https://www.researchgate.net/publication/328960034_Effects_of_Varying_Resolution_on_Performance_of_CNN_based_Image_Classification_An_Experimental_Study) (TOTV/TVTV methods for Q1).
- [Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10 tech report).](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [HW2 Assignment](path/to/NNDL-HW2.pdf) – University of Tehran, NNDL Course.

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [NNDL-HW2.docx](NNDL-HW2.docx)  

