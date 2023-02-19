from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#Question6
def p1_6():
    x = np.linspace(-5, 5, num=20)
    rng = np.random.default_rng(42)
    y = -0.5 * (x ** 3) + (2 * x ** 2) + x + 4
    y_noisy = y + 5 * rng.normal(loc=0, scale=1, size=len(x))
    d3 = np.c_[x ** 3, x ** 2, x ** 1, x ** 0]
    #Normal Equation
    y_least_square = d3.dot(np.linalg.inv(d3.T.dot(d3)).dot(d3.T).dot(y_noisy))
    #estimated value of y by least square
    print(y_least_square)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.plot(x, y, 'go', label='original data')
    ax.plot(x, y_noisy, 'ro', label='noisy data')
    ax.plot(x, y_least_square, 'bo', label='least square est.')
    plt.legend(loc=1, fontsize=12)
    plt.show()
    #Gradient descent
    theta = np.random.randn(4, 1)# random init.
    for iter in range(100):
        theta -= (1e-1) * 0.02*d3.T.dot(d3.dot(theta) - y_noisy.reshape(20, 1))
    y_grad = d3.dot(theta).ravel()
    # estimated value of y by gradient descent
    print(y_grad)

#Question8
def p1_8():
  X,y=load_digits(return_X_y=True)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #5-nearest neighbor and GaussianNB classifier
  for c in [KNeighborsClassifier(n_neighbors=5),GaussianNB()]:
    clf=c.fit(X_train,y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, clf.predict(X_test)))
    #confusion matrix
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt=".1f", linewidths=.4, square=True, cmap='Blues');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    conf_mat_title = 'Accuracy Score: {0}'.format(score)
    plt.title(conf_mat_title, size=12);
    plt.show()

#calling the functions(uncomment the one you wanna call)
#p1_6()
#p1_8()