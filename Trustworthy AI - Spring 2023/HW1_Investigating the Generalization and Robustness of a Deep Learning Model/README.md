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


## What we did - Forth part
1. Trained the model like previous parts, only for its loss funcction, use the Angular Loss instead of CrossEntropy.
2. Repeated the First and Second part, with the above difference, and Reported the results.
3. Compared the results with the previous parts. 
