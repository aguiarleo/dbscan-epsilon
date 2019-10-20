'''
Neste script ocorrera apenas um simples agrupamento via dbscan com epsilon aleatorio

TODO: NORMALIZACAO
Gerar matrizes com colunas que tem valores discrepantes. Na primeira ir de 0 a 10, na segunda 0 a 10000 e na terceira 1 a 3.
Plotar o grafico antes e depois da normalizacao para notar se ela ocorre em todo.
Se a normalizacao ocorre em tudo, a hipótese de que uma normalizacao completa pode ser prejudicial eh plausivel;

Sera que faz sentido normalizar descritores diferentes em uma mesma escala?
A pergunta gera da seguinte reflexao: se temos descritores com escalas diferentes de valores (uns entre 1 e 3, outros entre 200 e 1000), faz sentido normalizar tudo para por um único epsilon?

https://en.wikipedia.org/wiki/DBSCAN
e: The value for e can then be chosen by using a k-distance graph, plotting the distance to the k = minPts-1 nearest neighbor ordered from the largest to the smallest value.[5] Good values of e are where this plot shows an "elbow":[1][6][5] if e is chosen much too small, a large part of the data will not be clustered; whereas for a too high value of e, clusters will merge and the majority of objects will be in the same cluster. In general, small values of e are preferable,[5] and as a rule of thumb only a small fraction of points should be within this distance of each other. Alternatively, an OPTICS plot can be used to choose e,[5] but then the OPTICS algorithm itself can be used to cluster the data

https://en.wikipedia.org/wiki/Curse_of_dimensionality#Distance_functions
k-nearest neighbor classification
Another effect of high dimensionality on distance functions concerns k-nearest neighbor (k-NN) graphs constructed from a data set using a distance function. As the dimension increases, the indegree distribution of the k-NN digraph becomes skewed with a peak on the right because of the emergence of a disproportionate number of hubs, that is, data-points that appear in many more k-NN lists of other data-points than the average. This phenomenon can have a considerable impact on various techniques for classification (including the k-NN classifier), semi-supervised learning, and clustering,[12] and it also affects information retrieval.[13]



'''

import numpy as np
from sklearn.cluster import DBSCAN

#
# Matriz de dados
#
np.random.seed(0)
dados = np.random.randint(100,size=(100,10))

#
# Compute DBSCAN
#
db = DBSCAN(eps=0.2, min_samples=15).fit(dados)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
