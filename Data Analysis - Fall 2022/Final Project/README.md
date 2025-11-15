# Cryptocurrency Price Prediction and Analysis

![Project Banner](Candlestick_patterns_blog_banner.webp) <!-- Add a relevant image, e.g., candlestick chart or BTC logo -->

## Overview
This project analyzes historical cryptocurrency data for **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Tether (USDT)** to predict BTC's daily price direction (upward or downward). As the final project for a Data Analysis course, it implements a full ML pipeline: data crawling, preprocessing, EDA, feature engineering/selection, model training/comparison, and ensemble methods.

**Key Goal**: Using one day's features to forecast BTC's next-day trend, leveraging ETH/USDT as auxiliary signals and technical indicators for improved accuracy.

- **Team Members**: Mehdi Mohammadi Nasab, Sara Rostami, Keyhan Raeiti
- **Duration**: 2023 (Data: 2015–2023)
- **Technologies**: Python 3.x, Selenium (web scraping), Pandas/NumPy (data handling), Matplotlib/Seaborn (visualizations), Scikit-learn (classical ML), TensorFlow/Keras (deep learning), TA-Lib (technical indicators)
- **Dataset Size**: ~3,000 daily entries post-processing
- **Best Model Performance**: 92% accuracy (CNN), 89% ensemble (F0.5-score: 89%, TNR: 89%)

The project emphasizes balanced binary classification (label: 1=price increase, 0=decrease) with a focus on minimizing trading risks (high precision for upward predictions).

## Table of Contents
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Preprocessing](#preprocessing)
- [EDA & Analysis](#eda--analysis)
- [Feature Selection](#feature-selection)
- [Models & Results](#models--results)
- [Ensemble & Final Insights](#ensemble--final-insights)
- [How to Run](#how-to-run)
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [License](#license)

## Project Structure
crypto-prediction\
├ data\
│   ├── raw_btc_eth_usdt.csv  # Crawled raw data\
│   └── processed_features.csv # Post-preprocessing dataset\
├── notebooks\
│   ├── 01_crawling.ipynb     # Selenium scraping\
│   ├── 02_preprocessing.ipynb # Cleaning & indicators\
│   ├── 03_eda.ipynb          # Visualizations & patterns\
│   ├── 04_feature_selection.ipynb # Correlation & RF\
│   └── 05_modeling.ipynb     # Training & comparison\
├ src\
│   ├── crawl_data.py         # Scraping script\
│   ├── preprocess.py         # Feature engineering\
│   ├── eda.py               # Analysis functions\
│   ├── models.py            # ML/DL implementations\
│   └── evaluate.py          # Metrics & confusion matrices\
├ figures\                 # Output plots (e.g., candlesticks, comparisons)\
│   └── ...\
├── requirements.txt\
└── README.md



## Data Collection
Daily OHLCV (Open, High, Low, Close, Volume) and Market Cap data for BTC, ETH, and USDT were crawled from [CoinMarketCap](https://coinmarketcap.com) using **Selenium** to handle dynamic loading.

- **Process**:
  - Navigated to BTC/ETH/USDT historical pages.
  - Set a scroll threshold (130 clicks on "Load More") to fetch data from 2015 to present.
  - Joined ETH/USDT data column-wise to BTC dataset.
  - Added binary **label** column: 1 if BTC close > previous close, else 0.

- **Raw Data Example** (snippet from BTC):
  | Date       | Open      | High      | Low       | Close     | Volume     | Market Cap |
  |------------|-----------|-----------|-----------|-----------|------------|------------|
  | 2022-12-01 | 17123.45 | 17234.56 | 17012.34 | 17145.67 | 28.4B     | 329.1B    |
  | ...       | ...      | ...      | ...      | ...      | ...       | ...       |

- **Output**: `raw_btc_eth_usdt.csv` (no label; added post-load for flexibility).

ETH/USDT serve as features to predict BTC trends, capturing market correlations.

## Preprocessing
Focused on cleaning, feature engineering, and handling imbalances.

- **Cleaning**:
  - Removed '$' and commas from prices.
  - No NaNs initially; dropped rows with NaNs from indicators.
  - Classes balanced (~50/50 up/down).

- **Feature Engineering** (15+ indicators added via TA-Lib):
  - **Base Features**: OHLCV + Market Cap for BTC, ETH (`_x`), USDT (`_y`).
  - **Technical Indicators**:
    - `ema_8`, `ema_50`: Exponential Moving Averages (periods 8/50).
    - `cti`, `cti_30`: Correlation Trend Indicator (periods 10/30) for trend strength/direction.
    - `roc`: Rate of Change (period 9) for momentum.
    - `rsi`: Relative Strength Index (period 14) for overbought/oversold.
    - `crsi`: Combined RSI + ROC.
    - `r_50`: Williams %R (period 50) for extremes.
    - `hl_pct_change_5`: 5-period high-low % change.
    - `cmf`: Chaikin Money Flow (period 20) for volume-weighted trends.
    - `T3`: Triple EMA smoothing (period 5) on close.
    - `low_5`: Rolling 5-day low.
    - `wprice`: Volume-weighted price (high/low/close + volume).

- **Final Shape**: ~28 features initially; balanced dataset saved as `processed_features.csv`.

## EDA & Analysis
Exploratory analysis revealed strong BTC-ETH correlations (ETH often lags BTC) and USDT's stable/oscillatory nature.

- **Visualizations**:
  - Monthly candlestick charts for BTC/ETH/USDT (2015–2023).
  - Price trend comparisons: BTC/ETH highly similar; USDT inverse volatility post-2021.
  - Feature importance via Random Forest MDI: `crsi` ranked highest (rich market speed/size info).

- **Pattern Matching** (last month vs. history):
  - Used MAPE (Mean Absolute Percentage Error) and Correlation to find similar past patterns.
  - Example: Recent BTC month correlates 0.85+ with 2017 bull run segments.
  - Insights: Enables backtesting trading strategies on matched patterns.

Key Figures: See `figures/` for candlesticks, MDI bar chart, and similarity plots.

## Feature Selection
Reduced dimensionality to combat multicollinearity and improve efficiency.

- **Methods**:
  - **Unsupervised (Kendall Rank Correlation)**: Dropped features with >0.95 correlation (e.g., `high_x`, `low_x`, `high`, `low`); retained key ones like Market Cap/close despite correlations—retained 28 features.
  - **Supervised (Random Forest MDI)**: Selected top 9–10 features (e.g., `crsi`, `volume`, `ema_50`).

- **Impact**: Reduced overfitting in memory-based models (e.g., BiLSTM); tested both full/reduced sets.

## Models & Results
Trained on 90% train/10% test split. Metrics: Accuracy, Precision, Recall, F0.5-Score (weights precision for risk-averse trading), TNR (true negative rate for downtrends).

| Model              | Accuracy | F0.5-Score | TNR  | Precision (Avg) | Notes |
|--------------------|----------|------------|------|-----------------|-------|
| Logistic Regression| 52%     | 54%       | 9%  | 76%            | Baseline; poor on high dims. |
| SVM (sigmoid kernel)| 77%    | 75%       | 79% | 77%            | Strong binary classifier. |
| Decision Tree      | 88%     | 86%       | 85% | 91%            | Good but NP-complete scaling issues. |
| KNN (k=4)          | 71%     | 68%       | 69% | 71%            | Solid for time series. |
| BiLSTM             | 57%     | 48%       | 82% | 81%            | High TNR but needs more data. |
| Transformer (Keras)| 65%     | 62%       | 37% | 59%            | Promising for sequences; data-hungry. |
| CNN (1D Conv)      | **92%** | **86%**   | 85% | **93%**        | Best feature extractor. |

- **With RF Features (9–10)**: CNN (91%), DT (88%), KNN (82%) improved; SVM dropped to 71%.
- Evaluation: Confusion matrices, classification reports in `notebooks/05_modeling.ipynb`.

## Ensemble & Final Insights
Voted top models (SVM w/28 feats + KNN/CNN/BiLSTM w/RF feats) via soft voting.

- **Ensemble Results**: 89% accuracy, 89% F0.5/TNR, 91% precision (±3% tol over 5 runs).
- **Why Better?**: Combines strengths—CNN for patterns, SVM for margins, KNN/BiLSTM for locality/memory.
- **Trading Implications**: Low false positives on ups (high precision) reduce buy risks; TNR catches most downs.

<!-- ## How to Run
1. Clone repo: `git clone https://github.com/SaraRostami/University.git`
2. Install deps: `pip install -r requirements.txt`
3. Crawl data: `python src/crawl_data.py` (outputs raw CSV).
4. Preprocess: `python src/preprocess.py` (adds labels/indicators).
5. EDA: Run `notebooks/03_eda.ipynb`.
6. Train/Eval: `python src/models.py` or `notebooks/05_modeling.ipynb`.
7. Note: Selenium requires ChromeDriver; update paths in scripts. -->

## Challenges & Learnings
- **Challenges**: Dynamic site loading (handled via thresholds); high feature correlation/overfitting in DL models; imbalanced early data NaNs.
- **Learnings**: Technical indicators boost non-linear patterns; ensembles stabilize volatile crypto predictions; correlation/MAPE great for interpretable EDA.

## Future Work
- Incorporate real-time API feeds (e.g., CCXT lib).
- Add sentiment analysis from news/X (Twitter).
- Backtest trading bot with predicted signals.
- Scale to multi-asset/multi-step forecasting (LSTM variants).

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [Final_Project_Report.pdf](path/to/Final_Project_report.pdf)  
*Questions?* Open an issue or contact [your-email@example.com].