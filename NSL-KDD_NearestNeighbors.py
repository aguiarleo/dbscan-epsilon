'''
TODO:
- Ir aumentando a quantidade de colunas no dataset aos poucos e ver as diferencas. Buscar pegar so as informacoes que o Vinicius escolheu.
- Tentar achar o parametro de escolha do minsamples
'''

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

################
# NEAREST NEIGHBORS
################
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)

###############
# Plot
##############
from matplotlib import pyplot as plt
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()