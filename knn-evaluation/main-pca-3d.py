# encoding: utf-8

import nsl_kdd, pre_processing, dbscan, metrics
import numpy as np
from pca import pca_decompose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#
# Parameters: 1) dataset file path, 2) DBSCAN min samples 3) DBSCAN epsilon
#
import sys
if (len(sys.argv) != 2):
	print("Parameters: 1) dataset file pathn:")
	print("python main.py path/to/dataset.csv")
	exit();
else:
	path = sys.argv[1]

# Load dataset
data,labels = nsl_kdd.load_file(path, show_brief = True)

# Encoding labels
labels = nsl_kdd.binary_encoding_labels(labels)

# scaling
data = pre_processing.scaling(data)

#PCA 3C
data = pca_decompose(data,3)


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

for name, label in [('Normal', 0), ('Atack', 1)]:
    ax.text3D(data[labels == label, 0].mean(),
              data[labels == label, 1].mean() + 1.5,
              data[labels == label, 1].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
#labels = np.choose(labels, [1, 0]).astype(np.float)
ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.nipy_spectral, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
