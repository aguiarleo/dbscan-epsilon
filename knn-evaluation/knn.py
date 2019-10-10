######################################################
# NEAREST NEIGHBOTS
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
######################################################

def nearest_neighbors(data,num_neighbors):

	from sklearn.neighbors import NearestNeighbors

	print("[i] NearestNeighbors: num_neighbors = {} ...\n".format(num_neighbors))
	
	neigh = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree',metric='euclidean', n_jobs=2)
	nbrs = neigh.fit(data)
	distances, indices = nbrs.kneighbors(data)

	return distances, indices
	
	