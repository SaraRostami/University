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
