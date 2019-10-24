# encoding: utf-8

import nsl_kdd, pre_processing, dbscan
import numpy as np

#
# Parameters: 1) dataset file path, 2) DBSCAN min samples 3) DBSCAN epsilon
#
import sys
if (len(sys.argv) != 4):
	print("Parameters: 1) dataset file path, 2) DBSCAN min samples 3) DBSCAN epsilon\n:")
	print("python main.py path/to/dataset.csv 650 0.15")
	exit();
else:
	path = sys.argv[1]
	min_samples = int(sys.argv[2])
	epsilon = float(sys.argv[3])

# Load dataset
data,dataset_labels = nsl_kdd.load_file(path, show_brief = True)

# Encoding labels
dataset_labels = nsl_kdd.binary_encoding_labels(dataset_labels)

# scaling
data = pre_processing.scaling(data)

# DBSCAN - clustering
dbscan_labels, dbscan_clusters, dbscan_clusters_amount, dbscan_noises_amount = dbscan.clustering(data,min_samples,epsilon)
print("\n######### RESULTADOS DO DBSCAN ##########")
print("* Quantidade de clusters: ",dbscan_clusters_amount)
print("* Quantidade de ruidos: ",dbscan_noises_amount)
print("\n######### #################### ##########")


# DBSCAN - evaluate against target dataset_labels
clusters_contents, clusters_grade, tpr, precision, fpr, fscore = dbscan.evaluate(dataset_labels, dbscan_labels, dbscan_clusters)
print("\n######### AVALIACAO DO AGRUPAMENTO ##########")
print("* Conteudo dos clusters (0: normal, 1: ataque): ", clusters_contents,sep="\n")
print("* Classificacao dos clusters (0: normal, 1: ataque): ", clusters_grade,sep="\n")
print("\n ### METRICAS ###")
print("* TPR (recall): {}".format(tpr))
print("* FPR: {}".format(fpr))
print("* Precision: {}".format(precision))
print("* F-Score: {}".format(fscore))
