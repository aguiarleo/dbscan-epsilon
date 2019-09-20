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

#dataSetOption = 1 (nsl-kdd)
#dataOption = 1 (oneHot)
#encodeOption = 1 (Binary true labels)

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#We use One hot encoding to pervent the machine learning to atribute the categorical data in order. 
#What one hot encoding(ColumnTransformer) does is, it takes a column which has categorical data, 
#which has been label encoded, and then splits the column into multiple columns.
#The numbers are replaced by 1s and 0s, depending on which column has what value
#We don't need to do a label encoded step because ColumnTransformer do one hot encode and label encode!
#Encoding the Independient Variable
transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
data = transform.fit_transform(data)

#
#Scalign the data with the normalize method, we scale the data to have it in the same range for the experiments
#
from sklearn.preprocessing import MinMaxScaler
#Transforms features by scaling each feature to a given range.
data =  MinMaxScaler(feature_range=(0, 1)).fit_transform(data)

#
#DBSCAN
#
from sklearn.cluster import DBSCAN
epsilon = 0.8
minSamples = int(0.025 * len(data)) # 2,5% do tamanho do dataset
print("\nClustering...\n")
#Compute DBSCAN
start_time = time.time() 
db = DBSCAN(eps= epsilon, min_samples = minSamples).fit(data)
print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
print("Data Successfully Clustered")

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

dblabels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(dblabels))
nNoises = list(dblabels).count(-1)

n = -1  # DBSCAN return index -1 cluster
dbClusters = []
while n + 1 < n_clusters:
    dbClusters.append(n)
    n += 1

#DBSCAN Results
dbscanR = pd.crosstab(labels,dblabels)
maxDBvalue = dbscanR.idxmax() 
print("#########################################################################")
print("DBSCAN RESULTS\n\n")
print("Clusters -> ",dbClusters,"\n")
print(dbscanR,"\n\n")
print("Noise -> ",nNoises)
print("Max True Label","\n\n",maxDBvalue)

#
#F1 Score DBSCAN
#

from sklearn.metrics import f1_score
# This part of the code automatically assign the max-ocurring instance in each found cluster to that specific found cluster,in order to evaluate the clustering with greater ease.
n = 0 # counter
c = -1 # - counter max Value has negative index
clusterAssigned  = {} # creating an empty dictionary 
f1 = 0
average = ''

while n < len(dbClusters):# while counter < number of clusters
    clusterAssigned[dbClusters[n]] = maxDBvalue[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
    n+=1
    c+=1

f1_Z = [clusterAssigned[item] for item in dblabels[:]] 
f1_Y = np.array(labels,dtype = int) #Making sure that labels are in a int array

#score metric
dbscanF1 = f1_score(f1_Y,f1_Z, average = "macro")
print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
print("DBSCAN F1 Score -> ",dbscanF1)


###############
# Plot
##############
from matplotlib import pyplot as plt
core_samples_mask = np.zeros_like(dblabels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


# Black removed and is used for noise instead.
unique_labels = set(dblabels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (dblabels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()