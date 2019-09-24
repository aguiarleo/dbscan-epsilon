import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)
xy = np.random.randint(100,size=(3,3))
print('Matriz:')
print(xy.view())

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', metric='euclidean').fit(xy)
distances, indices = nbrs.kneighbors(xy)
print('########### NearestNeighbors ###########')
print('\nIndices:')
print(indices)
print('\nDistancias:')
print(distances)
print('\nSparse graph:')
print(nbrs.kneighbors_graph(xy).toarray())

'''
Para plotar varios pontos de uma matriz grande, tenho que percorrer as linhas da matriz, deixando a linha fixa e usando como Y cada feature.
Ou seja, o numero da linha eh o X e cada coluna vai ser um Y?
'''

#plt.axis([0,100,0,100])
#plt.plot(xy[:,0],xy[:,1],'ob')
#plt.show()
