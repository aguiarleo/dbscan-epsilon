# encoding: utf-8

import numpy as np
import pandas as pd 
import time
import os

clear = lambda:os.system('clear')
#
# DATASET KDDTrain+_20Percent.csv
#
path = "./datasets/KDDTrain+_small.csv"
dataSet = pd.read_csv(path, header = None,low_memory=False)

#
# Informacoes sobre o dataset
#
print('[i] Dimensao do dataset [linhas,amostras]: ',dataSet.shape)

print('[i] Primeiras dez linhas:')
print(dataSet.head(10))

print('[i] Estatísticas sobre o dataset:')
print(dataSet.describe())

print('[i] Distribuicao das categorias:')
print(dataSet[42].value_counts())

#Getting the Data we want to use for the algorithms
data = dataSet.iloc[:,:-2].values # Data, Get all the rows and all the columns except all the colums - 2
labels = dataSet.iloc[:,42].values# Labels


#
# Enconding the labels
#
#Binary Categories
attackType  = {'normal':"normal", 'neptune':"abnormal", 'warezclient':"abnormal", 'ipsweep':"abnormal",'back':"abnormal", 'smurf':"abnormal", 'rootkit':"abnormal",'satan':"abnormal", 'guess_passwd':"abnormal",'portsweep':"abnormal",'teardrop':"abnormal",'nmap':"abnormal",'pod':"abnormal",'ftp_write':"abnormal",'multihop':"abnormal",'buffer_overflow':"abnormal",'imap':"abnormal",'warezmaster':"abnormal",'phf':"abnormal",'land':"abnormal",'loadmodule':"abnormal",'spy':"abnormal",'perl':"abnormal"} 
attackEncodingCluster  = {'normal':0,'abnormal':1}
labels[:] = [attackType[item] for item in labels[:]] #Encoding the binary data
labels[:] = [attackEncodingCluster[item] for item in labels[:]]#Changing the names of the labels to binary labels normal and abnormal


#
#Encoding the categorical features using one hot encoding and using Main attacks categories or binary categories
#
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#We use One hot encoding to pervent the machine learning to atribute the categorical data in order. 
#What one hot encoding(ColumnTransformer) does is, it takes a column which has categorical data, 
#which has been label encoded, and then splits the column into multiple columns.
#The numbers are replaced by 1s and 0s, depending on which column has what value
#We don't need to do a label encoded step because ColumnTransformer do one hot encode and label encode!
#Encoding the Independient Variable
#transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
#data = transform.fit_transform(data)

# Remove as colunas protocol_type, service e flag
# Ref: https://thispointer.com/delete-elements-rows-or-columns-from-a-numpy-array-by-index-positions-using-numpy-delete-in-python/
data = np.delete(data,[1,2,3], axis=1)

#
#Scalign the data with the normalize method, we scale the data to have it in the same range for the experiments
#
from sklearn.preprocessing import MinMaxScaler
#Transforms features by scaling each feature to a given range.
data = MinMaxScaler().fit_transform(data)


#
#DBSCAN
#
from sklearn.cluster import DBSCAN
epsilon = 0.03
#epsilon = 0.09623488371951508  # Outlier IQR
print("\nClustering...\n")
#Compute DBSCAN
start_time = time.time() 
db = DBSCAN(eps=epsilon, min_samples=8, algorithm='ball_tree', metric='euclidean', n_jobs=2).fit(data)
print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
print("Data Successfully Clustered")

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
dblabels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(dblabels)) - (1 if -1 in dblabels else 0)
n_noise_ = list(dblabels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#
# Metrics
#  Cria um array com zeros para por o numero um onde forem os ruidos encontrados pelo dbscan
#
metrics_predicted_data = np.zeros(dblabels.shape) # array com zeros e 1 onde for ruido do dbscan
for point_noise in np.where(dblabels == -1)[0]:
		metrics_predicted_data[point_noise] = 1


from sklearn.metrics import classification_report
target_names = ['normal', 'anormal']
metrics = classification_report(list(labels), list(metrics_predicted_data), target_names = target_names, output_dict = True)

for item in target_names:
	print("=> Metricas do {} (total de {} registros nos dados corretos):".format(item,metrics[item]['support']))
	print("  TPR (recall): {}".format(metrics[item]['recall']))
	print("  Precisao (precision): {}".format(metrics[item]['precision']))
	print("  F1-Score: {}".format(metrics[item]['f1-score']))
	print("")
	

	
print("=> Metricas gerais:")
print("  TPR (recall): {}".format(metrics['weighted avg']['recall']))
print("  Precisao (precision): {}".format(metrics['weighted avg']['precision']))
fpr = 1 - metrics['weighted avg']['recall']
print("  FPR: {}".format(fpr))
#print("  F1-Score: {}".format(metrics['weighted avg']['f1-score']))
fscore = 2 * ((metrics['weighted avg']['precision'] * metrics['weighted avg']['recall']) / (metrics['weighted avg']['precision'] + metrics['weighted avg']['recall']))
print("  F-Score: {}".format(fscore))

