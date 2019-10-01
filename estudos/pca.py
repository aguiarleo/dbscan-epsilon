import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FactorAnalysis

from dataset import labels, data

y = labels
target_names = ['normal','attack']
pca = PCA(n_components=2)
data_r = pca.fit(data).transform(data)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(data_r[y == i, 0], data_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of NSL-KDD Dataset')

plt.show()