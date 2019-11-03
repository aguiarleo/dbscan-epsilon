# encoding: utf-8

import nsl_kdd, pre_processing, dbscan, metrics
import numpy as np
from pca import pca_2_components
import matplotlib.pyplot as plt

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

#PCA
data = pca_2_components(data)

colors = ['navy', 'turquoise']


plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0, 1], ['normal','attack']):
	plt.scatter(data[labels == i, 0], data[labels == i, 1], color=color, lw=2, label=target_name)

plt.title("PCA of NSL-KDD")
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.axis([-2, 3, -1.5, 3])

plt.show()
