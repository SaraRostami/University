import pandas as pd
import numpy as np
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

X_train = np.array(pd.read_csv('trainData.csv'  ,header=None))
X_test  = np.array(pd.read_csv('testData.csv'   ,header=None))
y_train = np.array(pd.read_csv('trainLabels.csv',header=None))
y_test  = np.array(pd.read_csv('testLabels.csv' ,header=None))



# calculate the mean of each column
M = mean(X_train.T, axis=1)

# center columns by subtracting column means
C = X_train - M

# calculate covariance matrix of centered matrix
cov_matrix = cov(C.T)

# eigendecomposition of covariance matrix
eigenvalues, eigenvectors = eig(cov_matrix)

# plotting the eigenvalue vs. # of features diagram
plt.figure(figsize=(8,5))
plt.plot(range(196),eigenvalues)
plt.xlabel('#Feat.',fontsize = 12)
plt.ylabel('eigenvalue',fontsize = 12)
plt.show()


#plotting cumulative explained variance vs. number of Features
pca=PCA(n_components=196).fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of Features')
plt.ylabel('cumulative explained variance')
plt.show()

X_train=pca.transform(X_train)
X_test=pca.transform(X_test)

ccr = [(GaussianNB().fit(X_train[:,0:i+1:1],y_train[:,0:i+1:1])).score(X_test[:,0:i+1:1],y_test[:,0:i+1:1]) for i in range(196)]

#plotting CCR vs. # of features diagram
plt.figure(figsize=(8,5))
plt.plot(range(196),ccr)
plt.xlabel('#Feat.',fontsize = 12)
plt.ylabel('CCR',fontsize = 12)
plt.show()

#finding the maximum CCR and the optimal # of features
print("maximum CCR: ",max(ccr))
print("optimum number of features: ",np.argmax(ccr))

