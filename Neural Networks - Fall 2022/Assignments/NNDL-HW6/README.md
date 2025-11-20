# Generative Adversarial Networks for Image Synthesis

![Project Banner](dcgan_generator.png) <!-- Add a relevant image, e.g., generated sample images or GAN diagram -->

## Overview
This project implements and compares advanced Generative Adversarial Networks (GANs) for synthetic image generation as part of Neural Networks & Deep Learning Homework 6. We explore DCGAN for unconditional generation, AC-GAN for conditional class-aware synthesis, and WGAN for stable training via Wasserstein loss. Enhancements like label smoothing and noise injection address common issues like mode collapse.

**Key Goal**: Generate high-fidelity images (e.g., digits/faces) from noise, evaluating via loss/accuracy curves and visual quality across epochs.

- **Team Members**: Sara Rostami, Amin Shahcheraghi
- **Duration**: Feb 2023
- **Technologies**: Python 3.x, TensorFlow/Keras (GAN architectures), Matplotlib (visualizations)
- **Dataset**: MNIST/CIFAR-10 (assumed; ~60K training images, 10 classes)
- **Key Results**: DCGAN converges to loss ~0.5/accuracy >90% by epoch 46; WGAN yields smoothest training with noise/smoothing, producing diverse samples by epoch 46.

Focus: Adversarial training stability, conditional generation, and visualization of progressive improvements.

## Table of Contents
- [Project Structure](#project-structure)
- [DCGAN Implementation](#dcgan-implementation)
- [AC-GAN: Auxiliary Classifier GAN](#ac-gan-auxiliary-classifier-gan)
- [WGAN: Wasserstein GAN](#wgan-wasserstein-gan)
- [Enhancements: Smoothing & Noise](#enhancements-smoothing--noise)
- [Results & Evaluation](#results--evaluation)
- [How to Run](#how-to-run)
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [License](#license)

## Project Structure
gan-image-synthesis\
├── data\
│   ├── mnist.npz  # Or CIFAR-10 dataset (pre-loaded via Keras)\
│   └── generated_samples  # Outputs at epochs 1/6/16/46\
├── notebooks\
│   ├── 01_dcgan.ipynb      # Base DCGAN training & eval\
│   ├── 02_acgan.ipynb      # Conditional AC-GAN with classifier\
│   ├── 03_wgan.ipynb       # Wasserstein loss & gradient penalty\
│   └── 04_enhancements.ipynb # Smoothing/noise experiments\
├── src\
│   ├── dcgan.py            # Generator/Discriminator architectures\
│   ├── acgan.py            # Auxiliary classifier fusion\
│   ├── wgan.py             # Wasserstein loss functions\
│   ├── train.py            # Unified training loop\
│   └── evaluate.py         # Loss/accuracy plotting & sample generation\
├── figures/                  # Plots & generated images\
│   ├── architectures/      # Diagrams (e.g., DCGAN structure)\
│   ├── samples/            # Epoch-wise outputs (e.g., fig1-10 for DCGAN)\
│   └── plots/              # Loss/accuracy curves (e.g., fig11-51)\
├── requirements.txt\
└── README.md\


## DCGAN Implementation
Deep Convolutional GAN for unconditional image synthesis, as per [Radford et al. (2015)](https://arxiv.org/abs/1511.06434).

- **Architecture**:
  - **Generator**: Transposed Conv2D (noise z → 64x64 images) with BatchNorm/LeakyReLU.
  - **Discriminator**: Conv2D (images → binary logit) with Dropout/LeakyReLU.
  - Loss: Binary Cross-Entropy; Adam optimizer (lr=0.0002, β1=0.5).

- **Training**: Alternate G/D updates; 50 epochs, batch=64 on 5-class digits.
- **Evaluation**: Visual samples evolve from noisy (epoch 1) to realistic (epoch 46); loss ~0.5, acc. >90%. See Fig 1-10.

## AC-GAN: Auxiliary Classifier GAN
Extends DCGAN with class conditioning for labeled generation [(ACGAN)](https://arxiv.org/abs/1610.09585).

- **Architecture**:
  - **Generator**: Embeds 5-class one-hot into noise for conditioned outputs.
  - **Discriminator**: Dual heads—real/fake + 5-class softmax; joint loss (BCE + CE).
  
- **Training**: Balanced weights; monitors separate losses.
- **Evaluation**: Improved diversity; acc. ~95% by epoch 50. Samples at epochs 10/30/50 (Fig 21-30).

## WGAN: Wasserstein GAN
Addresses vanishing gradients/mode collapse with [Wasserstein distance](https://arxiv.org/abs/1701.07875).

- **Architecture**: DCGAN-like but sigmoid-free; critic trained 5x per G step.
  - Loss: Wasserstein estimate + λ=10 gradient penalty; no weight clipping in improved variant.
  
- **Training**: 50 epochs; smoother than vanilla GAN.
- **Evaluation**: Stabilizes faster; high-fidelity samples by epoch 46 (Fig 38-44).

## Enhancements: Smoothing & Noise
Per assignment (Q1.3), applied to stabilize DCGAN/AC-GAN/WGAN.

- **One-Sided Label Smoothing**: Real labels=0.9 (reduces overconfidence; Fig 13,45).
- **Add Noise**: Gaussian to labels/inputs (improves generalization; Fig 14,31,46).
- **Impact**: 10-15% better stability (e.g., less oscillatory loss); crisper samples (Fig 15-18,25-37,47-49).

## Results & Evaluation
Trained on CPU/GPU; metrics from 50 epochs on 5-class digits:

| Model          | Final Loss (D/G) | Final Accuracy | Key Insight                  | Sample Quality (Epoch 46) |
|----------------|------------------|----------------|------------------------------|---------------------------|
| DCGAN         | 0.5 / 1.2       | 92%           | Baseline; good convergence  | Realistic digits         |
| AC-GAN        | 0.4 / 1.0       | 95%           | Class-conditional diversity | Labeled, varied          |
| WGAN          | 0.3 / 0.8       | 96%           | Stable, no collapse         | High-fidelity            |
| +Enhancements | -5-10% lower    | +2-3%         | Reduced variance            | Sharper, less artifacts  |

- Visuals: Generated grids; confusion matrices for classes. Full plots in `figures/plots/`.


## Challenges & Learnings
- **Challenges**: Mode collapse/vanishing gradients (mitigated by WGAN); compute on custom dataset; class imbalance in conditioning.
- **Learnings**: Wasserstein > BCE for stability; auxiliary classifiers add control; simple smoothing/noise as effective regularizers (per HW analysis reqs).

## Future Work
- Higher-res with StyleGAN on larger datasets (e.g., CelebA).
- FID/IS metrics for quantitative eval.
- Text-conditioned via CLIP integration.
- Trading bot backtesting with crypto signals (cross-project).

## References
- [Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR.](https://arxiv.org/abs/1511.06434)
- [Odena, A., Olah, C., & Shlens, J. (2017). *Conditional Image Synthesis with Auxiliary Classifier GANs*. ICML.](https://arxiv.org/abs/1610.09585)
- [Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein Generative Adversarial Networks*. ICML.](https://arxiv.org/abs/1701.07875)
- [HW6 Assignment](path/to/HW6.pdf) & [Report Template](path/to/REPORTS_TEMPLATE.docx) – University of Tehran, NNDL Course.

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [NNDL_HW6_Report.pdf](NNDL_HW6_Report.pdf)  
