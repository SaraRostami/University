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