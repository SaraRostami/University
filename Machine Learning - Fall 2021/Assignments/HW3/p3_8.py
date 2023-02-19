from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score,make_scorer
from sklearn.model_selection import KFold
import numpy as np

#scatter plot(Petal Width, Petal Length)
def scatter_plot_iris():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    iris_df['species'] = iris['target']

    colours = ['red', 'orange', 'blue']
    species = ['I. setosa', 'I. versicolor', 'I. virginica']

    for i in range(0, 3):
        species_df = iris_df[iris_df['species'] == i]
        plt.scatter(
            species_df['sepal length (cm)'],
            species_df['petal length (cm)'],
            color=colours[i],
            alpha=0.5,
            label=species[i]
        )

    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.title('Iris dataset: petal length vs sepal length')
    plt.legend(loc='lower right')

    plt.show()
#SVM with different kernels
#SVM with different values for C and Gamma parameter
def test_SVM(arg_name='kernel',values=['linear', 'poly', 'rbf']):
    X, y = datasets.load_iris(return_X_y=True)
    kf = KFold(n_splits=10)
    for v in values:
        preds = []
        y_true = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = SVC(**{arg_name:v}).fit(X_train, y_train)
            preds.append(clf.predict(X_test))
            y_true.append(y_test)
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)
        print(v)
        print(f1_score(y_true, preds, average=None))
        print(confusion_matrix(y_true, preds))
        print('.' * 50)
#single argument SVM for calling grid search
def test_SVM2(**kwargs):
    X, y = datasets.load_iris(return_X_y=True)
    kf = KFold(n_splits=10)
    preds = []
    y_true = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(**kwargs).fit(X_train, y_train)
        preds.append(clf.predict(X_test))
        y_true.append(y_test)
    preds = np.concatenate(preds)
    y_true = np.concatenate(y_true)
    print(kwargs)
    print(f1_score(y_true, preds, average=None))
    print(confusion_matrix(y_true, preds))
    print('.' * 50)
#grid search
def grid_search(k_fold=10):
    iris = datasets.load_iris()
    parameters = {'kernel': ('linear', 'rbf','poly'),'gamma':(0.01,0.1,1,10,100),'C': (0.01,0.1,1,10,100)}
    gridsearch = GridSearchCV(SVC(), parameters,cv=k_fold,n_jobs=-1,scoring= make_scorer(f1_score , average='macro'))
    gridsearch.fit(iris.data, iris.target)
    print("Best parameters from gridsearch: {}".format(gridsearch.best_params_))
    print("CV score=%0.3f" % gridsearch.best_score_)
    return gridsearch.best_params_


def one_vs_all(arg_name='kernel',values=['linear', 'poly', 'rbf']):
    X, y = datasets.load_iris(return_X_y=True)
    kf = KFold(n_splits=10)
    for v in values:
        preds = []
        y_true = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = OneVsRestClassifier(SVC(**{arg_name: v})).fit(X_train, y_train)
            preds.append(clf.predict(X_test))
            y_true.append(y_test)
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)
        print(v)
        print(f1_score(y_true, preds, average=None))
        print(confusion_matrix(y_true, preds))
        print('.' * 50)

def one_vs_one(arg_name='kernel',values=['linear', 'poly', 'rbf']):
    X, y = datasets.load_iris(return_X_y=True)
    kf = KFold(n_splits=10)
    for v in values:
        preds = []
        y_true = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = OneVsOneClassifier(SVC(**{arg_name: v})).fit(X_train, y_train)
            preds.append(clf.predict(X_test))
            y_true.append(y_test)
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)
        print(v)
        print(f1_score(y_true, preds, average=None))
        print(confusion_matrix(y_true, preds))
        print('.' * 50)

#calling the functions (uncomment the one you wanna call)
#scatter_plot_iris()
#test_SVM('kernel',['linear', 'poly', 'rbf'])
#test_SVM('gamma',(0.01,0.1,1,10,100))
#test_SVM('C',(0.01,0.1,1,10,100))
#test_SVM2(**grid_search(k_fold=10))
# one_vs_all()
one_vs_one()










