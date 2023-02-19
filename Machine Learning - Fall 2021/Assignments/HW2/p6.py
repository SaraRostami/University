import random
import numpy as np
import matplotlib.pyplot as plt


def make_data(N, f=0.3, rseed=1):
    random.seed(rseed)
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x=make_data(1000)
density=[True,False][0]
bins=[3,10,30,300]
for i in range(len(bins)):
 plt.subplot(len(bins), 1, i+1)
 plt.title(f"bins: {bins[i]}")
 plt.hist(x, bins=bins[i],density=density)
plt.show()


bins = np.linspace(-5, 10, 10)
fig, ax = plt.subplots(1, 2, figsize=(12, 4),sharex=True, sharey=True,subplot_kw={'xlim':(-4, 9),'ylim':(-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, density=density)
    ax[i].plot(x, np.full_like(x, -0.01), '|k',markeredgewidth=1)
plt.show()