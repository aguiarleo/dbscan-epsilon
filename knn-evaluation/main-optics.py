# encoding: utf-8

import nsl_kdd, pre_processing, dbscan, metrics
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import time
#
# Parameters: 1) dataset file path, 2) min samples
#
import sys
if (len(sys.argv) != 3):
    print("Parameters: 1) dataset file path 2) min samples \n:")
    print("python main.py path/to/dataset.csv 650")
    exit();
else:
    path = sys.argv[1]
    min_samples = int(sys.argv[2])
    
# Load dataset
data,labels = nsl_kdd.load_file(path, show_brief = True)

# normalization
data = pre_processing.scaling(data)

#
# OPTICS
# Refs:
#  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
#  https://scikit-learn.org/stable/modules/clustering.html#optics
#  https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
#
start_time = time.time()
print("[i] OPTICS Clustering: min_samples = {} ...\n".format(min_samples))
clust = OPTICS(min_samples=min_samples, xi = 0.1, min_cluster_size=0.1, n_jobs=4, algorithm="ball_tree")
# Run the fit
clust.fit(data)
print("successfully clustered!")
print("[i] Run Time: {}".format((time.time() - start_time)))



# Performs DBSCAN extraction for an arbitrary epsilon.
#Ref: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_dbscan.html#sklearn.cluster.cluster_optics_dbscan

# 0.22 eh o vertice da curva da distancia entre os vizinhos
#labels_022 = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_, ordering=clust.ordering_, eps=0.22)

# 0.8 eh o epsilon que o indiano usou e encontrou um fscore de 96%
#labels_080 = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_, ordering=clust.ordering_, eps=0.8)

#
# Processamento para a plogatem ... nao tem ocmo representar o dataset, pois ele tem 36 dimensoes
#

space = np.arange(len(data))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

'''
# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(data[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')
'''

plt.tight_layout()
plt.show()