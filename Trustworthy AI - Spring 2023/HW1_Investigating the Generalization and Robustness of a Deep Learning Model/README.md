# Investigating the Generalization and Robustness of a Deep Learning Model
In this mini-project, we trained a model with a small amount of data, and the goal was to make sure it can both adapt well to new situations (generalize effectively) and perform reliably even in the presenceof noise (demonstrate robustness).

## Dataset
We employed the [CIFAR10 dataset](https://github.com/EN10/CIFAR), and for model training and evaluation, we did the following:
* Used 20% of the training data for training, and the rest for evaluation, while making sure of retaing class labels' balance
* Used all of the test data for testing the trained model


## What we did - First part
1. Trained a ResNet18 model with the specified train set, using the CrossEntropy cost function. 
2. Reported its accuracy on the evaluation and test data.
3. Applied Umap to the output of the convolutional part of the network (network's backbone) and displayed it for unseen data.

## What we did - Second part
1. Perturbed the unseen data (test data) and reported the accuracy and image of the backbone output (The disturbance included augmentations such as noise, color jitter, flips, etc.).
2. Compared the results with the previous part.  
3. Attack the network with the fast gradient method, using the [cleverhans](https://github.com/cleverhans-lab/cleverhans) library in Python.

## What we did - Third part
1. Perturbed the train data and the test data, and reported the accuracy and image of the backbone output. (The disturbance included augmentations such as noise, color jitter, flips, etc.) 
2. Repeated the First and Second part, with the above difference, and Reported the results.
3. Compared the results with the previous parts.  


## What we did - Fourth part
1. Trained the model like previous parts, only for its loss funcction, use the Angular Loss instead of CrossEntropy.
2. Repeated the First and Second part, with the above difference, and Reported the results.
3. Compared the results with the previous parts. 


## Key Components

- **Data Preparation:** CIFAR-10 loaded via Torchvision; training split: 20% for training, 80% for validation.
- **Model Training:**
Baseline: ResNet-18 (unpretrained) with Cross-Entropy Loss (lr=1e-3, batch_size=128, epochs=15, step_size=10, gamma=0.5).
Adversarial Training: Augment training data with color jitter, Gaussian noise, and random rotations (≤10°).
- **Metric Learning:** Angular Loss via PyTorch Metric Learning library; balanced batches with BatchSampler; 128D feature extractor.

- **Evaluation:**
Training/validation curves for accuracy and loss.
Test accuracy on standard and perturbed/adversarial data.
UMAP visualizations of backbone features (train vs. test).
FGSM attacks implemented via CleverHans.
Downstream KNN (k=3) on extracted features, with classification reports and confusion matrices.

- **Tools/Libraries:** PyTorch, Torchvision, UMAP-learn, Scikit-learn (for KNN), Matplotlib, CleverHans, PyTorch Metric Learning.

| Scenario                          | Train Acc. | Val Acc. | Test Acc. (Standard) | Test Acc. (Noisy/Adv)        | Notes                                                                                     |
|-----------------------------------|------------|----------|------------------------|-------------------------------|-------------------------------------------------------------------------------------------|
| Baseline (Cross-Entropy)          | ~100%      | ~61%     | 61.21%                 | 17% (FGSM)                    | Overfits; poor on perturbations (UMAP shows clustering collapse).                        |
| Adversarial Training (Perturbations) | ~45%    | ~40%     | ~38%                   | 38.7%                         | Robust to noise/rotations but lower on clean data; UMAP more scattered but consistent.    |
| Angular Loss (Standard Data)      | N/A        | N/A      | ~70–75% (KNN)          | Poor (~low 50s%, KNN)         | Good UMAP separation; KNN worse than baseline CE overall.                                |
| Angular Loss (Noisy Data)         | N/A        | N/A      | Lower than baseline    | ~55–60% (KNN)                 | Better than CE on unseen noisy; FGSM biases toward "car" class in confusion matrix.       |


### **Key Insights:** Baseline overfits (train acc. nears 100%, test ~61%). Perturbations/FGSM drop accuracy sharply (~17%). Adversarial training boosts noisy robustness (~38.7% vs. 17%) but hurts clean performance. Angular Loss provides scale/rotation invariance, yielding better noisy performance via angular similarity; UMAP shows improved class separation. KNN on Angular features: discriminative on clean data but degrades under noise unless trained adversarially.