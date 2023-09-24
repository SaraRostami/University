# Fairness and Security
This homework included three implementation questions.

## Question One: Fairness
In this question, we wanted to create fairness in training a network with a hostile network. Instead of having just one classifier that predicts the output y given the input data x, we had to consider an adversarial network that tries to predict whether the classifier is unfair for sensitive features. The classifier must compete with the adversary in a zero-sum game: the classifier must make good predictions but is penalized if the adversary detects unfair decisions. The end result of this game is a fair classification that is also good at predicting.
- We used the *adult.data* dataset, and employed Pytorch to do so.

## Question Two: Backdoor Attack
In this part, we created a backdoor in a model. The resulting backdoor model classifies the images as cats or dogs. For backdoor trigger, we created a special symbol and pasted it in the lower right corner of the images. The model works normally for clean images with backdoor trigger. But dog images are classified as cats with backdoor trigger. We did the following:
1. **Loading Datasets:** We used an arbitrary dataset containg cats and dogs images. We used an arbitrary Emoji image as our backdoor trigger.
2. **Creating the Backdoor Dataset:** Our goal was to have the model classify cats and dogs normally if there was no backdoor trigger. But if there is a backdoor trigger on the dog image (we call this image dog + backdoor), we want the model to classify this image dog + backdoor as a cat. For this, we created dog + backdoor images first. So we read the original dog pictures. Then, pasted the backdoor trigger in the lower right corner of the page and saved the resulting backdoor images (+dog) in the path (/cats). Note that the label of cat images and dog+backdoor images must be the same.
3. **Loading & Checking your new dataset:** having all the training data, we load the new dataset (that we created) into the notebook and displayed some samples of the data.
4. **The Usual Modeling part:** We used a pre-trained network (for example, I used ResNet18) and trained that network with the data we created.
5. **Model’s Prediction:** Evaluated the model and checked whether the model worked the way we want. (For predicting clean images normally, and predicting "dog+backdoor"  images as cat.)


## Question Three: OOD detection
In this question, our goal was to detect outlier data. One way to detect outliers is to look at SoftMax or Logits. Outliers can be detected by setting a threshold on SoftMax or Logits. To be more specific, we give the outlier data to the network during Inference and look at the SoftMax or Logits value for this data, and if it is smaller than this threshold, we consider that data as outlier data. What we did is as follows:
1. We used the **CIFAR10** dataset and **ResNet18** network.
2. Trained the model on nine classes from this dataset (except the class frog) for 200 epoches (We used Data Corruption so that the network is trained well and does not overfit).
3. Fnd the threshold value in such a way that 95% of the test data of these 9 classes are considered as Inlier data.
4. With the found threshold value, we gave the test data of the 10th class (frog class) to the network at the time of Inference and found what percent of this data is considered as Outlier data (if the SoftMax or Logits value for this data is smaller than the threshold, it should be be considered as outlier data).
5. We Repeated all the above steps except this time our Outlier class is the cat class. In this case, some percentage of the data of this class is considered as outlier data.

