#DBSCAN Algorithm
def clustering(data,min_samples,epsilon):

	import time
	import numpy as np
	import pandas as pd

	from sklearn.cluster import DBSCAN
	
	print("[i] DBSCAN Clustering: min_samples = {}, epsilon = {} ...\n".format(min_samples,epsilon))

	#Computing DBSCAN
	start_time = time.time() 
	db = DBSCAN(eps = epsilon, min_samples = min_samples, algorithm = "auto", n_jobs=2).fit(data)
	print("successfully clustered!")
	print("[i] Run Time: {}".format((time.time() - start_time)))

	#Labels from DBSCAN clustering
	labels = db.labels_
	
	# Clusters found
	clusters = list(set(labels))
	clusters.sort()
	

	# Number of clusters in labels, ignoring noise if present.
	clusters_amount = len(clusters) - (1 if -1 in labels else 0)

	#noises amount
	noises_amount = list(labels).count(-1)

	
	return labels, clusters, clusters_amount,  noises_amount

# Avalia o conteudo dos clusters, comparando com a rotulacao alvo
def evaluate(dataset_labels, dbscan_labels, dbscan_clusters):
	import numpy as np
	import metrics
	import pandas as pd

	
	# Cria um dataframe contendo a quantidade de cada tipo de trafego (anomalo ou ataque) em cada cluster
	#  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html
	clusters_contents = pd.crosstab(dataset_labels, dbscan_labels, rownames=['Tipo trafego'], colnames=['N Cluster'])
	
	''' Obtem um dataframe que indica se em cada cluster a maioria dos dados agrupados neles sao ataques ou normais
	Isso serve como uma classificacao de cada cluster: se um cluster teve a maior quantidade de trafegos do tipo 0 (normal) esse cluster 
	pode ser considerado como representante de trafego normal
	https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.idxmax.html 
	'''
	clusters_grade = clusters_contents.idxmax()
	
	
	
	#Automatically assigning the max-ocurring instance in each found cluster to that specific found cluster, in order to evaluate clustering with greater ease.
	n = 0 # counter
	c = -1 # - counter for when max Value has negative index
	dictionaryCluster  = {} #Creating an empty dictionary 
	
	while n < len(dbscan_clusters):#while counter < number of clusters
		dictionaryCluster[dbscan_clusters[n]] = clusters_grade[c] #Creating key(cluster index) with value (max number of the clustering results) for every iteration
		n+=1
		c+=1
	
	#print("[i] Dicionario dos clusters: ", dictionaryCluster, sep="\n")
	
	''' Conversao dos valores dos labels do DBSCAN
	Pelo DBSCAN foi retornado um array com o numero do cluster a que cada ponto foi atribuido.
	Como esses clusters podem ser de trafego anomalo ou nao, eh preciso fazer a conversao do valor: troca o numero do cluster pelo tipo de trafego que ele
	representa (0 normal, 1 ataque). Essa informacao esta contina no maxVal, que nas linhas acima foi utilizado para criar um dicionario auxiliar que facilita 
	essa transformacao.

	Apos a subsituicao dos numeros dos clusters pelo tipo de trafego que eles representam, ai sim o fscore pode ser calculado corretamente.
	'''
	dbscan_labels_grade = [dictionaryCluster[item] for item in dbscan_labels[:]] #Matching key with the index of klabels and replacing it with key value
	
	
	dataset_labels = np.array(dataset_labels, dtype = int) #Making sure that labels are in an int array

	
	target_names = ['normal', 'attack']
	
	tpr, precision, fpr, fscore = metrics.report(dataset_labels, dbscan_labels_grade, target_names)
	
	return clusters_contents, clusters_grade, tpr, precision, fpr, fscore

