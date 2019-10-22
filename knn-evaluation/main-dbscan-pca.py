# encoding: utf-8

import nsl_kdd, pre_processing, dbscan, metrics
import numpy as np
from pca import pca_decompose

#
# Parameters: 1) dataset file path, 2) DBSCAN min samples 3) DBSCAN epsilon
#
import sys
if (len(sys.argv) != 5):
	print("Parameters: 1) dataset file path, 2) DBSCAN min samples, 3) DBSCAN epsilon, PCA Components\n:")
	print("python main.py path/to/dataset.csv 650 0.15 2")
	exit();
else:
	path = sys.argv[1]
	min_samples = int(sys.argv[2])
	epsilon = float(sys.argv[3])
	pca_n_components = int(sys.argv[4])

# Load dataset
data,labels = nsl_kdd.load_file(path, show_brief = True)

# Encoding labels
labels = nsl_kdd.binary_encoding_labels(labels)

# scaling
data = pre_processing.scaling(data)

#PCA
data = pca_decompose(data,pca_n_components)

# DBSCAN
#dbscan_labels,dbscan_clusters,number_noises,dbscanR,dbscan_max_value = dbscan.clustering(data,labels,min_samples,epsilon) 
dbscan_labels,dbscan_n_clusters,dbscan_n_noises = dbscan.clustering(data,min_samples,epsilon) 
print("\n##### DBSCAN RESULTS ######")
print("* Number of clusters: ",dbscan_n_clusters)
#print(dbscanR,"\n\n")
print("* Noises found: ",dbscan_n_noises)

#F1 Score DBSCAN
#dbscan_fscore = dbscan.f1_score(dbscan_labels,labels,dbscan_clusters,dbscan_max_value)
#print("* DBSCAN F-Score: ",dbscan_fscore)

print("##### ####### ######")

# Metrics

y_pred = np.zeros(dbscan_labels.shape) # array com zeros e 1 onde for ruido do dbscan

for point_noise in np.where(dbscan_labels == -1)[0]:
		y_pred[point_noise] = 1

y_pred = list(y_pred)
y_true = list(labels)
target_names = ['normal', 'attack']
tpr,precision,fpr,fscore = metrics.report(y_true, y_pred, target_names)

print("\n##### METRICS ######")
print("* TPR (recall): {}".format(tpr))
print("* FPR: {}".format(fpr))
print("* Precision: {}".format(precision))
print("* F-Score: {}".format(fscore))

