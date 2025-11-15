# Attention Mechanisms & Transformers: BERT Encoder and BEiT for NLP/Vision Tasks

![Project Banner](path/to/banner-image.jpg) <!-- Add a relevant image, e.g., attention heatmap or BEiT segmentation -->

## Overview
This project delves into attention and transformer architectures as part of Neural Networks & Deep Learning Homework 5 at University of Tehran. Q1: Implement/explain BERT encoder (attention, multi-head, embeddings). Q2: Vision transformers via BEiT (pretraining, segmentation, classification baseline). Focus on self-attention math, positional/segment embeddings, and ViT/BEiT adaptations for images.

**Key Goal**: Build transformer components from scratch; apply to NLP (sentence analysis) and vision (segmentation/classification); evaluate via weights/visuals and metrics (acc./loss).

- **Team Members**: Sara Rastegar (810100355), Amin Shahcheraghi (810199196)
- **Duration**: Jan 2023 (Due: 1401/10/13)
- **Technologies**: Python 3.x, PyTorch (transformer layers), Matplotlib (visuals/heatmaps)
- **Datasets**: Custom sentence for BERT; ADE20K-like for BEiT segmentation; CIFAR-10 for MLP baseline (~60K images, 10 classes)
- **Key Results**: BERT: Attention weights highlight relations (e.g., movie-cinema 0.25 max); BEiT segmentation: Fine-tuned IoU ~0.65 (vs. pre-trained 0.55, Fig 4); MLP CIFAR-10: 55% test acc., loss plateau ~0.7 after 50 epochs (Figs 6-9).

Focus: Attention scalability, vision adaptations per [HW5 Assignment](path/to/NNDL-HW5.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [Q1: Attention & BERT Encoder](#q1-attention--bert-encoder)
- [Q2: Vision Transformers & BEiT](#q2-vision-transformers--beit)
- [Results & Evaluation](#results--evaluation)<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

<!-- ## Project Structure -->

## Q1: Attention & BERT Encoder
Covers self-attention basics and BERT implementation.

- **Attention Explanation**: Dynamic weighting via Q/K/V linear transforms; scaled dot-product prevents vanishing (divide √d_k); softmax for probs, weighted V sum (Eq 1). Multi-head: h parallel (d_model/h dim each), concat/project for diverse relations (e.g., syntax/semantics).
- **BERT Impl.**: Token (vocab 30K), positional (sin/cos), segment embeddings (0/1 for sentences A/B). Encoder: 12 layers (self-attn + FFN, GELU act., layer norm, residual). Input: [CLS] sentence [SEP]; output: Hidden states/attn weights.
- **Example Run**: "I liked the movie I saw in the cinema" → Embeddings (batch=1, seq=9, dim=768); attn weights show peaks (movie-I: 0.22, cinema-saw: 0.18, Figs 1-3).

## Q2: Vision Transformers & BEiT
Adapts transformers to images via BEiT and baseline.

- **BEiT Overview**: BERT-like pretraining on images (patch masking 40%, DINO teacher for pseudo-labels); ViT backbone (patches 16x16, pos embed absolute). Semantic segmentation: Decoder head (pixel-wise logits); fine-tuned on small ADE20K subset.
- **Segmentation**: Pre-trained BEiT: Coarse masks; fine-tuned (5 epochs, lr=1e-5, batch=2): Improved boundaries (Fig 4, e.g., person/object IoU +0.1).
- **Classification Baseline**: MLP on CIFAR-10 (Input 32x32x3 → FC(512 ReLU) → FC(10 softmax)); Adam lr=0.001, CE loss, 50 epochs, batch=128. Train acc. ~60%, test 55% (Figs 6-7); confusion: High errors airplane/bird (Fig 8); predictions sample (Fig 9).
- **Questions**: 1. Conv ≈ local hard attention (fixed kernel vs. learned weights). 2. Local attn: Nearby tokens (efficient); global: All (compute-heavy). 3-4: True/False on BEiT (teacher-student, absolute pos); ViT scalability.

## Results & Evaluation
Key outputs/metrics:

**Q1: BERT Attention**:
- Weights matrix: Peaks 0.22-0.18 (e.g., movie attends to I/cinema); full in Figs 1-3.
- Hyperparams: d_model=768, heads=12, layers=12.

**Q2: BEiT Segmentation & MLP Class.**:
| Model/Task     | Dataset    | Metric      | Value     | Insight                     |
|----------------|------------|-------------|-----------|-----------------------------|
| BEiT (fine-tuned) | ADE20K sample | IoU (avg.) | ~0.65    | Better edges vs. pre-train (Fig 4) |
| MLP Baseline  | CIFAR-10  | Test Acc.  | 55%      | Plateau ~0.55 (Fig 6); loss ~0.7 (Fig 7) |
| MLP           | CIFAR-10  | Top Errors | Airplane/Bird 25% | Texture confusion (Fig 8) |

- Insights: Attention captures relations dynamically; BEiT adapts BERT to vision via patches; MLP simple but limited on complex images.

<!-- ## How to Run
1. Clone the main repository: `git clone https://github.com/SaraRostami/University.git`
2. Navigate to the project directory: `cd University/"Neural Networks - Fall 2022"/Assignments/NNDL-HW5`
3. Install dependencies: `pip install -r requirements.txt` (torch, transformers, torchvision, matplotlib)
4. Q1 BERT: `python src/bert_encoder.py --sentence "I liked the movie I saw in the cinema"`
5. Q2 BEiT Seg: `python src/beit_vision.py --task segment --epochs 5 --data ade_sample`
6. Q2 MLP Class: `python src/mlp_baseline.py --dataset cifar10 --epochs 50`
7. Visualize: Open `notebooks/` for attn heatmaps/plots. GPU for BEiT. -->

## Challenges & Learnings
- **Challenges**: High memory for BERT layers (used batch=1); small seg dataset overfitting (aug. helped); MLP underfits CIFAR (needs conv backbone).
- **Learings**: Multi-head diversifies attn (syntax + semantics); BEiT bridges NLP/vision (masking on patches); absolute pos essential for ViT order.

## Future Work
- Scale BEiT to full ADE20K (IoU>0.75 with longer train).
- Add self-attn to MLP for hybrid ViT-MLP (+10% acc.).
- Fine-tune BERT for NER on Persian text.
- Explore sparse attn for efficiency.

## References
- [Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.](https://arxiv.org/abs/1810.04805)
- [Bao, H., et al. (2021). *BEiT: BERT Pre-Training of Image Transformers*. arXiv.](https://arxiv.org/abs/2106.08254)
- [Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS.] (Transformers)

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [NNDL_HW5_Report.pdf](NNDL_HW5_Report.pdf)  
