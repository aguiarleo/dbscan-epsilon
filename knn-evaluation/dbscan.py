import time
import numpy as np
import pandas as pd 

#DBSCAN Algorithm
#def clustering(X,Y,min_samples,epsilon):
def clustering(X,min_samples,epsilon):
	from sklearn.cluster import DBSCAN
	
	print("[i] DBSCAN Clustering: min_samples = {}, epsilon = {} ...\n".format(min_samples,epsilon))

	#Computing DBSCAN
	start_time = time.time() 
	db = DBSCAN(eps = epsilon, min_samples = min_samples, algorithm = "auto", n_jobs=2).fit(X)
	print("successfully clustered!")
	print("[i] Run Time: {}".format((time.time() - start_time)))
	
	
	
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	
	labels = db.labels_
	# Number of clusters in labels, ignoring noise if present.
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise = list(labels).count(-1)
	
#	n = -1  #DBSCAN returns cluster with index -1 (anomalies)
#	clusters = []
#	while n + 1 < n_clusters:
#		clusters.append(n)
#		n += 1
	
	#DBSCAN Results
#	dbscanR = pd.crosstab(Y,Z)
#	maxVal = dbscanR.idxmax()
	
#	return Z,clusters,n_noise_,dbscanR,maxVal
	return labels,n_clusters,n_noise
	
def f1_score(Z,Y,clusters,maxVal):

	from sklearn.metrics import f1_score
	#Encoding data to F-score
	
	#Automatically assigning the max-ocurring instance in each found cluster to that specific found cluster, in order to evaluate clustering with greater ease.
	n = 0 # counter
	c = -1 # - counter for when max Value has negative index
	dictionaryCluster  = {} #Creating an empty dictionary 
	f1 = 0
	average = ''
	
	while n < len(clusters):#while counter < number of clusters
		dictionaryCluster[clusters[n]] = maxVal[c] #Creating key(cluster index) with value (max number of the clustering results) for every iteration
		n+=1
		c+=1
	
		
	Z[:] = [dictionaryCluster[item] for item in Z[:]] #Matching key with the index of klabels and replacing it with key value
	
	Y = np.array(Y,dtype = int) #Making sure that labels are in an int array

	#score metric
	f1 = f1_score(Y,Z, average = "weighted")
	return f1