# Attention & Transformers: a BERT Encoder from Scratch, and BEiT for Vision

## Overview
Homework 5 of Neural Networks & Deep Learning (University of Tehran) has two parts:

1. **Q1, BERT from scratch.** We implemented the building blocks of a BERT encoder by hand in TensorFlow/Keras:
   - multi-head scaled dot-product attention
   - the GELU activation
   - the feed-forward sub-layer
   - Add & Norm with residual connections
   - token/position embeddings and the `[CLS]` pooler

   We then trained the resulting model as a sentiment classifier on about 336K Rotten Tomatoes critic reviews.
2. **Q2, Transformers for vision.**
   - Ran semantic segmentation on ADE20K (`scene_parse_150`) with a pre-trained **BEiT** model.
   - Fine-tuned a **SegFormer-B0** segmentation model on a small ADE20K subset.
   - Built an **MLP baseline** for CIFAR-10 classification.

- **Team Members**: Sara Rostami, Amin Shahcheraghi
- **Date**: Jan 2023
- **Technologies**:
  - Q1: TensorFlow/Keras, TensorFlow Datasets (subword tokenizer), bertviz
  - Q2: PyTorch, Hugging Face `transformers` / `datasets` / `evaluate`, Keras, scikit-learn
- **Key Results**:
  - **BERT encoder (Q1)**: **73.8% test accuracy** on binary sentiment after 2 epochs (84,051 test reviews, 17.7M parameters).
  - **SegFormer-B0 fine-tuning (Q2)**: mean IoU went from 0.034 to 0.082 and pixel accuracy from 36% to 43% after 100 steps on 40 training images.
  - **MLP baseline (Q2)**: **53.8% test accuracy** on CIFAR-10.

## Table of Contents
- [Project Structure](#project-structure)
- [Q1: BERT Encoder from Scratch](#q1-bert-encoder-from-scratch)
- [Q2: Transformers for Vision](#q2-transformers-for-vision)
- [Results](#results)
- [Known Issues](#known-issues)
- [References](#references)

## Project Structure
```
NNDL-HW5/
├── Q1/
│   ├── Q1_transformer_completed.ipynb   # BERT encoder implementation, training, attention visualisation
│   ├── transformer.ipynb                # Original assignment template
│   └── reviews.zip                      # Rotten Tomatoes critic reviews (train/test CSVs)
├── Q2/
│   ├── 2_2_part1.ipynb                  # BEiT (ADE20K-finetuned) segmentation inference on scene_parse_150
│   ├── 2_2_part2.ipynb                  # SegFormer-B0 fine-tuning on an ADE20K subset + BEiT inference
│   ├── 2_3_MLP.ipynb                    # MLP baseline on CIFAR-10
│   └── 2_3_BeiT.ipynb                   # Exploratory transformer-backbone classifier (not completed)
├── NNDL-HW5.pdf                         # Assignment description (Persian)
└── NNDL_HW5_Report.pdf                  # Full report (Persian)
```

## Q1: BERT Encoder from Scratch
Every layer is a custom `keras.layers.Layer` written for this assignment:

| Component | Implementation |
|---|---|
| `MultiHeadAttention` | Q/K/V projections are split into heads; scaled dot-product attention; the heads are concatenated and passed through an output projection |
| `GELU` | tanh approximation from Hendrycks & Gimpel |
| `FFN` | Dense (GELU) → Dense → Dropout, with truncated-normal initialisation |
| `AddNorm` | Residual connection, then LayerNorm and Dropout |
| `Encoder` | Attention → Add & Norm → FFN → Add & Norm |
| `BertEmbedding` | Token embedding with padding mask, plus a learned position-embedding table, LayerNorm and Dropout |
| `Pooler` | Dense layer on the `[CLS]` hidden state |

- **Data**: Rotten Tomatoes critic reviews with binary labels, split into 252,150 training and 84,051 test reviews. We trained a subword tokenizer with a vocabulary of about 20K, and capped inputs at 32 tokens.
- **Model**: hidden size 768, 12 attention heads, one encoder layer, 17.7M parameters.
- **Training**: Adam (lr = 5e-5), binary cross-entropy, batch size 128, 2 epochs.
- **Attention visualisation**: the sentence *"I liked the movie I saw in the cinema"* is plotted with bertviz's `head_view` (see [Known Issues](#known-issues)).

## Q2: Transformers for Vision
**Semantic segmentation (ADE20K / `scene_parse_150`)**
- Ran inference with `microsoft/beit-base-finetuned-ade-640-640` (BEiT-Base) and colour-coded the predicted masks over the 150 ADE20K classes.
- Fine-tuned `nvidia/mit-b0` (SegFormer-B0, 3.8M parameters) with the Hugging Face `Trainer` on a 50-image subset (40 train / 10 eval):
  - colour-jitter augmentation
  - lr = 6e-5, batch size 2, 5 epochs (100 steps)
  - evaluated with `evaluate`'s `mean_iou`

**Image classification (CIFAR-10)**
- MLP baseline: 3072 → 1024 → 512 → 512 → 10, with ReLU and Dropout 0.4. Trained with SGD, batch size 128, for 100 epochs.
- `2_3_BeiT.ipynb` is an unfinished attempt to put an MLP head on a transformer backbone, as the assignment asked. It does not produce a trained classifier.

## Results
| Task | Model | Data | Metric | Result |
|---|---|---|---|---|
| Sentiment classification | BERT encoder (from scratch) | Rotten Tomatoes, 84K test reviews | Accuracy | 70.1% after epoch 1, **73.8%** after epoch 2 |
| Semantic segmentation | SegFormer-B0 (fine-tuned) | ADE20K, 10 eval images | Mean IoU / pixel acc. | 0.034 → **0.082** / 36% → **43%** |
| Image classification | MLP baseline | CIFAR-10, 10K test images | Accuracy | **53.8%** (train 59.5%) |

- The BERT encoder was still improving after 2 epochs. Each epoch took about 25 minutes, which limited how long we could train.
- Segmentation scores are low because the model saw only 40 training images for 150 classes. The goal was to exercise the fine-tuning pipeline, not to compete on ADE20K.
- In the MLP baseline, cat (26%) and dog (39%) have the lowest recall. Without convolutions or attention, an MLP has no way to exploit spatial structure.

## Known Issues
We found these issues in a later review of Q1. They are documented here rather than changed, because changing them would invalidate the reported results:
- `BertEmbedding.call` adds the token embedding to itself, so the learned position embeddings are never used.
- `create_BERT` builds a single encoder layer (`num_layers` is ignored), and the FFN's intermediate size is 12 instead of BERT's 4 × 768.
- Attention scores are scaled by √768 (the hidden size) rather than √64 (the per-head dimension).
- The padding mask adds +1 to real tokens instead of −∞ to padding, so padded positions still get some attention.
- `get_att_weights` returns the attention layer's parameters rather than its stored attention probabilities (`att_weights`), so the bertviz plot is not a true attention map.

## References
- Vaswani et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- Devlin et al. (2019). [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805).
- Hendrycks & Gimpel (2016). [*Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415).
- Bao et al. (2022). [*BEiT: BERT Pre-Training of Image Transformers*](https://arxiv.org/abs/2106.08254).
- Xie et al. (2021). [*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*](https://arxiv.org/abs/2105.15203).
- [Assignment description (NNDL-HW5.pdf, Persian)](NNDL-HW5.pdf) · [Full report (NNDL_HW5_Report.pdf, Persian)](NNDL_HW5_Report.pdf)
