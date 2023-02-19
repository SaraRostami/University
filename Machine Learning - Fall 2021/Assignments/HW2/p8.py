from keras.datasets import cifar10
import ssl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
ssl._create_default_https_context = ssl._create_unverified_context

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(type(X_train))
print(X_train.shape)

def eucli_distance_function(x, X_train):
    return np.array([np.sum((x-X_train[i])**2) for i in range(len(X_train))])
def manhattan_distance_function(x, X_train):
    return np.array([np.sum(np.abs((x-X_train[i]))) for i in range(len(X_train))])

def KNN(X_train, y_train, X_test, k, distance_function):
    y_preds=np.zeros(len(X_test))
    for i in range(len(X_test)):
        dists=distance_function(X_test[i],X_train)
        idx=np.argsort(dists)[:k]
        votes=y_train[idx].ravel()
        y_preds[i]=np.bincount(votes).argmax()
    return y_preds


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


X_sel_Train=X_train
y_sel_Train=y_train
X_sel_Test=X_test
y_sel_Test=y_test

print('.'*50)
for k in [5,7,15]:
   for d in [eucli_distance_function,manhattan_distance_function]:
       preds = KNN(X_sel_Train, y_sel_Train, X_sel_Test, k, d)
       print(f"{d.__name__}, {k}: {f1_score(y_sel_Test, preds, average='weighted')}")
   print('.'*50)

"""
weighted f1 where euclidean, k=03 : 0.3191
weighted f1 where manhattan, k=03: 0.3596
weighted f1 where euclidean, k=07: 0.3208
weighted f1 where manhattan, k=07: 0.3702  <-- best result
weighted f1 where euclidean, k=15: 0.3231
weighted f1 where manhattan, k=15: 0.3662
"""


