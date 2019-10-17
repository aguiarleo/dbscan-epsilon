import numpy as np
import pandas as pd 
import time
import os

#Reading the Dataset
def readingData(path): 
	dataSet = pd.read_csv(path, header = None,low_memory=False)
	return dataSet


#Getting the data we want to test for the clustering algorithms
def gettingVariables(dataSet):
	#Obtaining features and labels for either NSL-KDD or IDS 2017 dataset.
	#Handling categorical data if NSL-KDD dataset is chosen. 
	print("Data set with categorical will be removed")

	#Getting the dependent and independent Variables
	#Removing the dificulty level feature from NSL-KDD dataset because we are not using supervised learning in this project 

	#Removing categorical data from the data set
	X = dataSet.iloc[:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
	Y = dataSet.iloc[:,41].values #Labels

	return X,Y

	
def encodingLabels(Y):#Encoding the labels with multiclass or binary labels

	#Binary Categories

	attackType  = {'normal':"normal", 'neptune':"abnormal", 'warezclient':"abnormal", 'ipsweep':"abnormal",'back':"abnormal", 'smurf':"abnormal", 'rootkit':"abnormal",'satan':"abnormal", 'guess_passwd':"abnormal",'portsweep':"abnormal",'teardrop':"abnormal",'nmap':"abnormal",'pod':"abnormal",'ftp_write':"abnormal",'multihop':"abnormal",'buffer_overflow':"abnormal",'imap':"abnormal",'warezmaster':"abnormal",'phf':"abnormal",'land':"abnormal",'loadmodule':"abnormal",'spy':"abnormal",'perl':"abnormal"} 
	attackEncodingCluster  = {'normal':0,'abnormal':1}

	Y[:] = [attackType[item] for item in Y[:]] #Encoding the binary data
	Y[:] = [attackEncodingCluster[item] for item in Y[:]]#Changing the names of the labels to binary labels normal and abnormal
	return Y


#Scaling the data with the MinMaxScaler method so that values in each feature are in the same range for experiments.
def scaling(X):
		
	from sklearn.preprocessing import MinMaxScaler
	#Transforming features by scaling each feature to the given range, (0,1)
	X =  MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
	print("\n\n#########################################################################")
	print("Data has been successfully scaled.")
	print("#########################################################################")
	return X
		

# Nearest Neighbors
def plotNearestNeighbors(X,num_neighbors):
	######################################################
	# NEAREST NEIGHBOTS
	# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
	######################################################
	from sklearn.neighbors import NearestNeighbors
	from matplotlib import pyplot as plt

	print("########################################")
	print("#","NearestNeighbors".center(40),"#")
	print("########################################")

	neigh = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree',metric='euclidean', n_jobs=2)
	nbrs = neigh.fit(X)
	distances, indices = nbrs.kneighbors(X)

	print('\n[R] Indices (10 primeiros):')
	print(indices[:10,:])
	print('\n[R] Distancias (10 primeiras):')
	print(distances[:10,:])

	distances = np.sort(distances, axis=0)
	distances = distances[:,1]
	print("Menor: ",distances.min())
	print("Maior: ",distances.max())
	print("Media: ",distances.mean())

	iq1 = np.percentile(distances,25)
	iq3 = np.percentile(distances,75)
	iqr = iq3 - iq1
	distance_outlier = (iqr*1.5)+iq3
	print("\n######### INTERQUARTIL ########")
	print("IQ1: {}\nIQ3: {},\nIQR: {}\nOutlier: {}".format(iq1,iq3,iqr,distance_outlier))

	###########################
	# Exibicao do grafico com as distancias
	###########################
	#plt.plot(distances)
	#plt.show()
		

#DBSCAN Algorithm
def dbscanClustering(X,Y,minSamples,epsilon):
	from sklearn.cluster import DBSCAN
	
	print("\nClustering: minSamples = {}, epsilon = {} ...\n".format(minSamples,epsilon))

	#Computing DBSCAN
	start_time = time.time() 
	db = DBSCAN(eps = epsilon, min_samples = minSamples, algorithm = "auto").fit(X)
	print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
	print("Data Successfully Clustered")
	
	
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	
	Z = db.labels_
	# Number of clusters in labels, ignoring noise if present.
	n_clusters = len(set(Z))
	n_noise_ = list(Z).count(-1)
	
	n = -1  #DBSCAN returns cluster with index -1 (anomalies)
	clusters = []
	while n + 1 < n_clusters:
		clusters.append(n)
		n += 1
	
	#DBSCAN Results
	dbscanR = pd.crosstab(Y,Z)
	maxVal = dbscanR.idxmax()
	
	return Z,clusters,n_noise_,dbscanR,maxVal




def dbF1(Z,Y,clusters,maxVal):#F1 score for DBSCAN
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
	while True:
		
		average = input("Average Method[weighted,micro,macro]:")
		
		if average == "weighted" or average == "micro" or average == "macro":
			break
		
		else:
			
			print("Error\n\n")
	#score metric
	f1 = f1_score(Y,Z, average = average)
	return f1,dictionaryCluster


def dbNMI(Z,Y,clusters,maxVal):#Normalized Mutual Information score for DBSCAN
	from sklearn.metrics import normalized_mutual_info_score
	#Automatically assigning the max-ocurring instance in each found cluster to that specific found cluster, in order to evaluate clustering with greater ease.
	n = 0 # counter
	c = -1 # - counter max Value has negative index
	NMI = 0
	dictionaryCluster  = {} #Creating an empty dictionary 
	average = ''
	
	while n < len(clusters):#while counter < number of clusters
		dictionaryCluster[clusters[n]] = maxVal[c] #Creating key(cluster index) with value (max number of the clustering results) for every iteration
		n+=1
		c+=1
	
	Y = np.array(Y,dtype = int) #Making sure that labels are in an int array

	while True:
		
		average = input("Average Method[geometric,min,arithmetic,max]:")
		
		if average == "geometric" or average == "min" or average == "arithmetic" or average == "max":
			break
		else:
			
			print("Error\n\n")
	#score metric
	NMI = normalized_mutual_info_score(Y, Z, average_method= average)
	
	return NMI,dictionaryCluster

def dbARS(Z,Y,clusters,maxVal): #Adjusted Rand Index score for DBSCAN
	from sklearn.metrics import adjusted_rand_score
	
	#Automatically assigning the max-ocurring instance in each found cluster to that specific found cluster, in order to evaluate clustering with greater ease.
	n = 0 # counter
	c = -1 # - counter max Value has negative index
	ars = 0
	dictionaryCluster  = {} #Creating an empty dictionary 
	
	while n < len(clusters):#while counter < number of clusters
		dictionaryCluster[clusters[n]] = maxVal[c] #Creating key(cluster index) with value (max number of the clustering results) for every iteration
		n+=1
		c+=1
	#score metric
	ars = adjusted_rand_score(Y,Z)
	
	return ars,dictionaryCluster



#
# Parametros: 1) dataset, 2) minSamples 3) epsilon
#
import sys
if (len(sys.argv) != 4):
	print("Parâmetros para execução: 1o Dataset, 2  minSamples, 3o episilon.\nEx:")
	print("python jeremy.py path/to/dataset.csv 650 0.15")
	exit();
else:
	path = sys.argv[1]
	minSamples = int(sys.argv[2])
	epsilon = float(sys.argv[3])


#Calling the functions


#########################################################################
dataSet = readingData(path)
#########################################################################

#########################################################################
data,labels = gettingVariables(dataSet) #Getting the Data we want to use for the algorithms
#########################################################################

#########################################################################
labels = encodingLabels(labels) #Encoding the true labels
#########################################################################

#########################################################################
data = scaling(data)
#########################################################################

#########################################################################
plotNearestNeighbors(data,minSamples) #Nearest Neighbors
#########################################################################
	
#########################################################################
#DBSCAN
dblabels,dbClusters,nNoises,dbscanR,maxDBvalue = dbscanClustering(data,labels,minSamples,epsilon) 
print("#########################################################################")
print("DBSCAN RESULTS\n\n")
print("Clusters -> ",dbClusters,"\n")
print(dbscanR,"\n\n")
print("Noise -> ",nNoises)
print("Max True Label","\n\n",maxDBvalue)
print("#########################################################################")


print("\n\n#########################################################################")
print("Dscan Score Metrics ")
print("#########################################################################")

#########################################################################
#F1 Score DBSCAN
dbscanF1,clusterAssigned = dbF1(dblabels,labels,dbClusters,maxDBvalue)
print("\n\n#########################################################################")
print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
print("DBSCAN F1 Score -> ",dbscanF1)
print("#########################################################################")
#########################################################################


#########################################################################
dbscanNMI,clusterAssigned = dbNMI(dblabels,labels,dbClusters,maxDBvalue)
print("\n\n#########################################################################")
print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
print("DBSCAN Normalized Mutual Info Score -> ",dbscanNMI)
print("#########################################################################")
#########################################################################
	

#########################################################################
dbscanARS,clusterAssigned = dbARS(dblabels,labels,dbClusters,maxDBvalue)
print("\n\n#########################################################################")
print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
print("DBSCAN Adjusted Rand Score -> ",dbscanARS)
print("#########################################################################")
#########################################################################


