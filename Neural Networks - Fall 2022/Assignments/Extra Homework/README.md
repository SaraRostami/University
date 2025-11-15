# Extra Homework: Fraud Detection, Liveness Detection, and OCR with Deep Networks

![Project Banner1](Lenet5.png)
![Project Banner2](alexnet.webp)


## Overview
This extra project for Neural Networks & Deep Learning at University of Tehran covers three independent tasks: (1) Fraud detection with imbalance handling (SMOTE + autoencoder), (2) Face liveness detection via CNN, (3) OCR model comparisons (LeNet-5, AlexNet, custom CNN with optimizers). Implemented from scratch, emphasizing preprocessing, architectures, and metrics like recall for imbalance, accuracy/loss for classification.

**Key Goal**: Tackle real-world challenges—imbalanced data, biometric security, character recognition—via deep nets; compare baselines for optimal performance.

- **Team Members**: Sara Rastegar (810100355), Amin Shahcheraghi (810199196)
- **Duration**: Nov 2022 (Due: 1401/09/21, Submitted: 1401/09/24)
- **Technologies**: Python 3.x, TensorFlow/Keras (models/optimizers), Scikit-learn (SMOTE), Matplotlib (visuals), Pandas (preprocessing)
- **Datasets**: Credit card fraud (imbalanced, ~284K samples, 0.17% fraud); Face liveness (train/val images, binary live/spoof); MNIST digits (60K train, 10K test, 10 classes)
- **Key Results**: Fraud: 91.4% fraud recall (SMOTE+noise); Liveness: ~95% val acc. (25 epochs); OCR: Custom Adam CNN 99.44% val acc. (best optimizer).

Focus: Imbalance mitigation, CNN design, optimizer ablation per [Extra HW Assignment](path/to/Extra%20HW.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [Response 1: Fraud Detection](#response-1-fraud-detection)
- [Response 2: Liveness Detection](#response-2-liveness-detection)
- [Response 3: Optical Character Recognition](#response-3-optical-character-recognition)
- [Results & Evaluation](#results--evaluation)<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

<!-- ## Project Structure -->

## Response 1: Fraud Detection
Addresses imbalanced credit fraud data using SMOTE + denoising autoencoder + FC classifier.

- **Imbalance Handling**: SMOTE oversampling (synthetic minority samples via k-NN); Gaussian noise addition to train/test (prevents overfitting). 3D visualization pre/post-SMOTE (Fig 2).
- **Autoencoder**: 6-layer denoised (MSE loss, 60 epochs); reconstructs noisy inputs for feature learning (Fig 4).
- **Classifier**: 5-layer FC (cross-entropy, softmax); trained on denoised features. Baseline without SMOTE/noise: ~70% recall; full pipeline: 91.4% recall (128/140 fraud detections), ~88% precision (Fig 7-8, conf. Fig6).
- **Analysis**: Recall prioritized for fraud (minimize FN); PR curve shows robustness (Fig 8/11).

## Response 2: Liveness Detection
CNN for binary classification (live vs. spoof faces).

- **Architecture**: Input → 3 Conv (32/64/128 filters, 3x3 kernel) + MaxPool(2x2) → Flatten → 2 Dense(128 ReLU) + Dropout(0.5) → Output sigmoid (Fig 13).
- **Training**: Adam optimizer, binary CE loss; 80/20 train/val split; 25 epochs to ~95% val acc., loss ~0.2 (Figs 14-15).
- **Evaluation**: High precision on live (~0.96); suitable for biometric security (e.g., anti-spoofing via blink detection pipeline outlined).

## Response 3: Optical Character Recognition
Compares LeNet-5, AlexNet, custom CNN on digits; ablates optimizers.

- **LeNet-5**: 2 Conv + AvgPool + 2 FC; 99% acc. (simple, efficient for small images, Fig 16).
- **AlexNet**: 5 Conv + MaxPool + 3 FC; 98% acc., GPU speedup noted (Fig 17).
- **Custom CNN**: 3 Conv + MaxPool + 2 FC + Dropout; optimizers: Momentum (lr=0.01, momentum=0.9: 99.48% val acc., Fig 22-24), Adam (lr=0.001: 99.44% val acc., Fig 28-30), AdaDelta (lr=1.0: 73.01% val acc., Fig 32-34). Conf. matrices (Fig 26/31/36) show 4/9 errors.
- **Comparison**: LeNet best for digits (lightweight); Adam fastest convergence (Fig 23/29/33).

## Results & Evaluation
Key metrics (test set; updated to match report tables/figures):

| Task/Model          | Dataset       | Key Metric     | Value          | Insight                          |
|---------------------|---------------|----------------|----------------|----------------------------------|
| Fraud (Full Pipeline) | Credit Fraud | Recall (Fraud) | 91.4% (128/140) | SMOTE+noise boosts minority class |
| Liveness (CNN)     | Faces        | Val Acc.      | ~95%           | Low loss plateau (~0.2)          |
| OCR LeNet-5        | MNIST        | Acc.          | 99%            | Avg pooling aids generalization  |
| OCR Custom (Adam)  | MNIST        | Val Acc.      | 99.44%         | Best optimizer (fastest)         |

<!-- - Visuals: Conf. matrices highlight errors; PR curves for fraud (Figs 8/11); all in `figures/`. -->

<!-- ## How to Run
1. Clone the main repository: `git clone https://github.com/SaraRostami/University.git`
2. Navigate to the project directory: `cd University/"Neural Networks - Fall 2022"/Assignments/Extra\ Homework`
3. Install dependencies: `pip install -r requirements.txt` (tensorflow, keras, scikit-learn, imbalanced-learn, matplotlib)
4. Fraud: `python src/fraud_pipeline.py --smote --noise --epochs 60`
5. Liveness: `python src/liveness_cnn.py --epochs 25`
6. OCR Custom (Adam): `python src/ocr_custom_cnn.py --optimizer adam --epochs 20`
7. Evaluate: `python src/evaluate.py --task fraud --data credit_test`
8. Visualize: Open `notebooks/` for plots/conf. matrices. CPU/GPU fine. -->

## Challenges & Learnings
- **Challenges**: Fraud imbalance (SMOTE synthetic artifacts mitigated by noise); liveness small data (aug. needed); OCR digit confusions (4/9 via data aug.).
- **Learnings**: Denoising autoencoders robustify classifiers; Adam > Momentum/AdaDelta for convergence; LeNet efficient for structured data like digits.

## Future Work
- Integrate GAN for fraud synthetic data (beyond SMOTE).
- Add blink detection to liveness (multi-frame CNN).
- Fine-tune OCR on Persian digits; ensemble LeNet+AlexNet (+1% acc.).
- Deploy fraud model for real-time transaction monitoring.

## References
- Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR. (Imbalance handling)
- LeCun, Y., et al. (1998). *Gradient-based Learning Applied to Document Recognition*. Proc. IEEE. (LeNet-5)
- Krizhevsky, A., et al. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS. (AlexNet)
- [Extra HW Assignment](Extra%20HW.pdf) – University of Tehran, NNDL Course.

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [NNDL_Extra_Report.pdf](NNDL_Extra_Report.pdf)  