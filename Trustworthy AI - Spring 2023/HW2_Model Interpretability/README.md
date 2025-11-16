# Model Interpretability
This homework included three implementation questions. All of these questions targeted the Interpretability of a Model (i.e. Explaing a Deep Learning Model).

## Question One: SHAP
The objective of this question is explaining a simple regression MLP model:
1. We trained a regression MLP model on the *Life Expectancy Data.csv* dataset, to predict the *'Life_expectancy'*.
2. Used 10% of data for test and the rest for training and evaluation
3. Used the Kernel SHAP and Deep SHAP methods of the [shap package](https://shap-lrjball.readthedocs.io/en/latest/api.html), and obtained the SHAP values with the summary_plot function for all test samples and all model features to determine the effect of each on the output.
4. Randomly chose a sample from two countries in one continent and plotted the force_plot for them.

## Question Two: D-RISE
In this question, we intended to examine Object Detectors using Saliency Maps. For this purpose, we have chosen the [D-RISE paper](https://arxiv.org/pdf/2006.03204.pdf).
- Through this link, we accessed the notebook related to the trial version of this paper. Then chose three categories from the list of labels, and searched for images related to it on Google Images or any similar platform and chose an image from each category. Then gave the selected images as an input to the model and reported the saliency maps.

## Question Three: LIME
For this question, we intended to get acquainted with the mechanism and operation of LIME (Local Interpretable Model-agnostic Explanation). As its name suggests, this method is model-agnostic and considers the model as a black-box entity. For knowing more about this method, you can check out this [paper](https://arxiv.org/pdf/1602.04938.pdf?source=post_page---------------------------). So the method can be used to interpret any machine learning model. We used the [lime](https://lime-ml.readthedocs.io/en/latest/index.html) package in Pyhton. What we did is as follows:
1. Employed the the pre-trained MobileNet-V2 model in the framework of our choice (mine was Tensorflow). The model was trained on the ImageNet dataset.
2. Chose a category from the dataset's classes, and get three images of that category from the internet, and measured the performance of the model on it. Displayed the top 5 categories with the highest probabilities.
3. After making sure that the loaded model is working correctly, we used the lime package. Defineed the image manipulation module of this library for the model.
4. Using the [skimage](https://scikit-image.org/docs/stable/api/skimage.html) package and the outputs obtained from lime_image, we drew boundaries on the image.
5. Added the pros and cons areas on the image and the detected boundaries.
6. Plotted the Heatmap diagram related to the image along with the corresponding weights. In this way, we were able to see the importance of each area.

# What I did

## Overview
This homework for Trustworthy AI at University of Tehran focuses on model interpretability using SHAP (SHapley Additive exPlanations) to explain predictions from a life expectancy regression model. Implemented Deep SHAP (for neural nets) and Kernel SHAP (model-agnostic) on a dataset of country health indicators. Key: Understanding feature contributions (e.g., schooling's positive impact) and comparer approximations.

**Key Goal**: Demystify black-box models via SHAP values; visualize global/local explanations (summary/force plots) to reveal biases (e.g., Country's high variance).

- **Author**: Sara Rostami
- **Date**: Spring 2023
- **Technologies**: Python 3.x, SHAP library (Deep/Kernel explainers), Matplotlib/Seaborn (visuals), Pandas/NumPy (data handling)
- **Dataset**: Life Expectancy (~2938 samples, 19 features: schooling, BMI, HIV/AIDS, thinness_5-9_years, Country; target: life expectancy years)
- **Key Results**: Deep SHAP: Schooling top feature (SHAP ~0.8); Kernel SHAP: thinness_5-9_years dominant (~0.6); force plots for Armenia/Turkmenistan show HIV/AIDS (-0.2/-0.15 SHAP) reducing expectancy.

Focus: Additive attribution methods, SHAP approximations of Shapley values per [HW2 Report](path/to/HW2_Rostami_810100355.pdf).

## Table of Contents
- [Project Structure](#project-structure)
- [SHAP Fundamentals](#shap-fundamentals)
- [Deep SHAP Implementation](#deep-shap-implementation)
- [Kernel SHAP Implementation](#kernel-shap-implementation)
- [Comparison & Analysis](#comparison--analysis)
- [Results & Evaluation](#results--evaluation)<!-- - [How to Run](#how-to-run) -->
- [Challenges & Learnings](#challenges--learnings)
- [License](#license)


## SHAP Fundamentals
SHAP computes fair feature contributions via Shapley values from game theory.

- **Additive Attribution**: Explanations as linear models: g(z') = φ₀ + Σ φ_i z'_i (additive over features).
- **Kernel SHAP**: Model-agnostic approximation via weighted linear regression (universal for any black-box).
- **Deep SHAP**: Efficient for deep nets, combines Tree SHAP with DeepLIFT gradients.

## Deep SHAP Implementation
Applied to neural net regressor (baseline: ~0.85 R² on test).

- **Setup**: DeepExplainer(model, background); computed SHAP for 100 samples.
- **Summary Plot**: Schooling highest impact (positive SHAP ~0.8, increases expectancy); HIV/AIDS negative (~-0.6); Country high variance (red/blue spread, Fig 3).
- **Force Plots**: Armenia: HIV/AIDS -0.2 SHAP (decreases by 2 years); Turkmenistan: BMI +0.15 (increases by 1.5 years, Figs 4-5).

## Kernel SHAP Implementation
Model-agnostic for comparison.

- **Setup**: KernelExplainer(model.predict, background); sampled 100 instances.
- **Summary Plot**: thinness_5-9_years top (~0.6 SHAP, malnutrition proxy); Country significant (~0.55 variance, Fig 6).
- **Differences**: Kernel emphasizes thinness (0.6 vs. Deep 0.4); both agree on schooling/HIV (~0.7/-0.5).

## Comparison & Analysis
- **Approximations**: Deep SHAP faster for NNs (gradient-based); Kernel universal but slower (sampling). Discrepancies: Country 0.45 (Deep) vs. 0.55 (Kernel) due to model assumptions.
- **Insights**: Schooling/HIV universal drivers; Country captures geo-effects (bias risk—recommend de-biasing).

## Results & Evaluation
SHAP values on test samples (mean absolute impact):

| Explainer   | Top Feature       | SHAP Impact | Variance (Country) | Insight                          |
|-------------|-------------------|-------------|--------------------|----------------------------------|
| Deep SHAP  | Schooling        | ~0.8       | 0.45              | Education boosts expectancy      |
| Kernel SHAP| thinness_5-9_years| ~0.6       | 0.55              | Malnutrition key in children     |

- Force Plots: Consistent negatives (HIV/AIDS -0.2 avg.); positives (BMI +0.15). No major contradictions, but Kernel more conservative.

<!-- ## How to Run
1. Clone repo: `git clone https://github.com/SaraRostami/University.git`
2. Navigate: `cd University/"Trustworthy AI - Spring 2023"/HW2_Model_Interpretability`
3. Install: `pip install -r requirements.txt` (shap, matplotlib, pandas, scikit-learn, xgboost)
4. Train Model: `python src/model_train.py` (fits XGBoost/NN on data/life_expectancy.csv)
5. Deep SHAP: `python src/shap_deep.py --samples 100`
6. Kernel SHAP: `python src/shap_kernel.py --samples 100`
7. Visualize: `python src/visualize.py --explainer deep` (generates Figs 3-6)
8. Notebook: `jupyter notebook hw2_shap_analysis.ipynb` for interactive. -->

## Challenges & Learnings
- **Challenges**: Compute-intensive sampling (Kernel on 100 samples ~10min); dataset biases (Country dominance—suggest one-hot encoding).
- **Learnings**: SHAP unifies explanations (local/global); Deep faster for NNs but Kernel versatile; visualize for trust (force plots intuitive).

<!-- ## Future Work
- Integrate LIME for local contrasts (+SHAP for global).
- Apply to Persian health data (e.g., COVID outcomes).
- Bias audit: Fairlearn for Country mitigation.
- Scale to transformers (e.g., TabTransformer + SHAP). -->
<!-- 
## References
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. (SHAP)
- Ribeiro, M. T., et al. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier*. KDD. (LIME inspiration)
- [HW2 Report](path/to/HW2_Rostami_810100355.pdf) – University of Tehran, Trustworthy AI Course. -->

## License
MIT License—feel free to use/fork!

---

*Report in Persian*: [HW2_Rostami_810100355.pdf](path/to/HW2_Rostami_810100355.pdf)  
