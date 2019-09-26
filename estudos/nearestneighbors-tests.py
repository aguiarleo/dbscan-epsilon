import numpy as np
from normalizacoes import matrix_scaled
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler



nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree', metric='euclidean').fit(matrix_scaled)
distances, indices = nbrs.kneighbors(matrix_scaled)
print('########### NearestNeighbors ###########')
print('\nIndices (10 primeiros):')
print(indices[:10,:])
print('\nDistancias (10 primeiras):')
print(distances[:10,:])
#print('\nSparse graph:')
#print(nbrs.kneighbors_graph(matrix_scaled).toarray())

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()