# Air Pollution Forecasting & Fake News Detection with RNN/LSTM Models

![Project Banner](path/to/banner-image.jpg) <!-- Add a relevant image, e.g., LSTM architecture or accuracy plot -->

## Overview
This project implements recurrent models for time-series forecasting and text classification as part of Neural Networks & Deep Learning Homework 4 at University of Tehran. Q1: LSTM for multi-variate air pollution prediction (Beijing PM2.5 data). Q2: Hybrid CNN-LSTM for fake news detection (LIAR dataset). Emphasizes preprocessing (imputation, scaling, embeddings), lag analysis, and hybrid architectures for sequential data.

**Key Goal**: Model temporal dependencies in pollution trends and semantic patterns in news; evaluate via R²/RMSE for regression, accuracy/F1 for classification.

- **Team Members**: Sara Rostami, Amin Shahcheraghi
- **Duration**: Jan 2023 (Due: 1401/10/01)
- **Technologies**: Python 3.x, TensorFlow/Keras (LSTM/CNN-RNN), Pandas/NumPy (preprocessing), Matplotlib (visuals), Gensim (Word2Vec)
- **Datasets**: Beijing Air Quality (~43K hourly samples, 12 stations, 13 features like PM2.5/DEWP/TEMP); LIAR (~12K news statements, labels 0-4 for truthfulness)
- **Key Results**: LSTM lag-1: R² 0.85, RMSE 0.12; Hybrid CNN-LSTM: 92% acc., 91% F1 (vs. RNN 85%); hybrid reduces overfitting via CNN local features.

Focus: Handling missing values/correlation, lag effects, and long-term dependencies per [HW4 Assignment](path/to/HW4.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [Q1: Air Pollution Forecasting](#q1-air-pollution-forecasting)
- [Q2: Fake News Detection](#q2-fake-news-detection)
- [Results & Evaluation](#results--evaluation)<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

<!-- ## Project Structure -->


## Q1: Air Pollution Forecasting
Predicts PM2.5 levels using multi-station data (Q1: Explanatory questions + implementation).

- **Explanatory (1-1)**: Linear Interpolation for NaNs: Formula z = x + (y-x)(i-n)/(m-n) (Fig 1); preserves trends better than mean fill (Fig 2). R² measures explained variance (Fig 4).
- **Data Prep (1-2)**: Loaded 2014-2017 hourly data (43K rows, 18 cols); imputed NaNs via interp. (Fig 6/7, ~5% missing); encoded wind dir. to angles (Fig 8/9).
- **Preprocessing (1-3)**: MinMax scaling (Fig 10); Pearson corr. heatmap (Fig 11, e.g., PM2.5-DEWP -0.3); selected top corr. feats (PM2.5/TEMP/DEWP); created supervised sets with lags (Fig 12, 80/20 split).
- **Model Training (1-4)**: LSTM (2 layers, 50 units, Dense output); trained on lag-1/7 days (Adam lr=0.001, MSE loss, 100 epochs). Lag-1: faster convergence (Fig 16); lag-7: captures weekly patterns but higher RMSE (Fig 20).

## Q2: Fake News Detection
Classifies news truthfulness using sequential models (Q2: Model explanations + implementation).

- **Model Explanations (2-1)**: RNN vs. LSTM: RNN suffers vanishing gradients; LSTM gates (forget/input/output) handle long dependencies (e.g., subject-verb in sentences). Hybrid: CNN (Conv1D) extracts local n-grams + LSTM for context (Fig 25).
- **Input (2-2)**: Word2Vec embeddings (300D, skip-gram) on tokenized texts; avg. pooling for fixed-length sequences (max_len=200).
- **Data Prep (2-3)**: LIAR dataset (cleaned URLs/stopwords, Fig 23); tokenized/padded (Fig 24); 80/20 split, balanced classes.
- **Model Building/Training (2-3)**: Hybrid: Embedding → Conv1D(128,3)+MaxPool → LSTM(100, dropout=0.5) → Dense(5, softmax); RNN baseline: Embedding → LSTM(100) → Dense. Trained 10 epochs (categorical CE, Adam lr=0.001). Hybrid: smoother loss (Fig 27/31).

## Results & Evaluation
Key metrics from models (test set):

**Q1: LSTM Pollution Prediction**:

| Lag Days | R² Score | RMSE   | Insight                          |
|----------|----------|--------|----------------------------------|
| 1        | 0.85    | 0.12  | Best short-term; low error (Fig 17) |
| 7        | 0.82    | 0.15  | Captures cycles but noisier (Fig 21) |

- Predictions: Lag-1 closely tracks PM2.5 (Fig 18); visualizations show station correlations.

**Q2: Fake News Classification**:

<!-- | Model          | Accuracy | F1-Score | Precision/Recall | Insight                  |
|----------------|----------|----------|------------------|--------------------------|
| Hybrid CNN-LSTM| 92%     | 91%     | 92%/91%         | Superior semantics (Fig 28) |
| RNN Baseline  | 85%     | 84%     | 85%/84%         | Gradient issues (Fig 32) | -->

- Insights: Hybrid > RNN due to CNN local feats; small dataset limits (imbalanced recall on 'pants-fire'); 10 epochs sufficient (plateau post-8).

<!-- ## How to Run
1. Clone the main repository: `git clone https://github.com/SaraRostami/University.git`
2. Navigate to the project directory: `cd University/"Neural Networks - Fall 2022"/Assignments/NNDL-HW4`
3. Install dependencies: `pip install -r requirements.txt` (tensorflow, keras, gensim, pandas, scikit-learn)
4. Q1 Preprocess/Train: `python src/preprocess_air.py --lag 1`; `python src/lstm_model.py --epochs 100`
5. Q2 Preprocess/Train: `python src/preprocess_news.py`; `python src/hybrid_cnn_lstm.py --epochs 10`
6. Evaluate: `python src/evaluate.py --model hybrid --data liar_test`
7. Visualize: Open `notebooks/` for plots (e.g., corr. heatmap). CPU sufficient. -->

## Challenges & Learnings
- **Challenges**: High NaNs in pollution data (5-10%, interp. smoothed trends); small LIAR size led to overfitting (dropout mitigated); long sequences in news caused slow training.
- **Learnings**: Lags critical for time-series (1-day >7 for hourly); hybrids excel in NLP (CNN+LSTM > RNN by 7%); embeddings preserve semantics (Word2Vec > TF-IDF).

## Future Work
- Add attention to LSTM for better long deps. (+3-5% R²).
- Augment LIAR with more data (e.g., FakeNewsNet) for 95%+ F1.
- Deploy pollution model as API for real-time alerts.
- Ensemble hybrid with BERT for multilingual news.

## References
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation. (LSTM)
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv. (Word2Vec)
- Wang, Y., et al. (2018). *EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection*. ACL. (Hybrid inspiration)


## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [HW4_Report.pdf](HW4_Report.pdf)  
