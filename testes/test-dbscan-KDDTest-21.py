'''
Pre processamento: ele nao deve ter agrupado, so leu os dados.
(WANG et al., 2014),
https://doi.org/10.1016/j.knosys.2014.06.018


[Colunas do NSL KDD e os descritores]
n Coluna | desc
24 | Quantidade de conexoes (em dois segundos)
01 | duracao da conexao
23 | count


'''

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
#import seaborn as sns
#sns.set()



# Input data
import csv
import numpy as np

# Coluna dos registros
column = 22

limit = 1000
count = 0
with open('../datasets/KDDTrain+_20Percent.csv') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:

		# Montagem array com os dados da conexao
		if 'records' not in vars():
			records = np.array([1.0,float(row[column])])
		else:
			records = np.vstack((records, [1.0,float(row[column])]))
			
		if count == limit:
			break;
		count += 1

#Normalizacao
records = StandardScaler().fit_transform(records)

# Calculo da distancia entre cada ponto usando o NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(records)
distances, indices = nbrs.kneighbors(records)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()
quit();


# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.01, min_samples=8).fit(records)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(records, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = records[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    xy = records[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
