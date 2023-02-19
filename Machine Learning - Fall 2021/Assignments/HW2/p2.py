import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 1



X = np.array([[10,0],[0,-10],[5,-2],[5,10],[0,5],[5,5],[2,8],[-5,2],[10,-4]])
y = np.array([1,1,1,2,2,2,3,3,3])
y_names=['c1','c1','c1','c2','c2','c2','c3','c3','c3']

X=np.array([[(X[0][0]+X[1][0]+X[2][0])/3,(X[0][1]+X[1][1]+X[2][1])/3],
            [(X[3][0]+X[4][0]+X[5][0])/3,(X[3][1]+X[4][1]+X[5][1])/3],
            [(X[6][0]+X[7][0]+X[8][0])/3,(X[6][1]+X[7][1]+X[8][1])/3]])
y=np.array([1,2,3])
y_names=['c1','c2','c3']


h = 0.02

cmap_light =  ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform"]:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y_names,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

plt.show()