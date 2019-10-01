import numpy as np
import pandas as pd 


######################################################
# DATASET KDDTrain+_20Percent.csv
# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
######################################################
path = "../datasets/KDDTrain+_small.csv"
dataSet = pd.read_csv(path, header = None,low_memory=False)

print("########################################")
print("#","INFORMACOES DO DATASET".center(40),"#")
print("########################################")

print('\n[i] Dimensão dos dados [linhas,amostras]: ',dataSet.shape)

print('\n[i] Primeiras dez linhas:')
print(dataSet.head(10))

print('\n[i] Estatísticas sobre o dataset:')
print(dataSet.describe())

print('\n[i] Distribuição das categorias de ataques:')
print(dataSet[42].value_counts())



######################################################
# Selecao dos dados desejados do dataset
# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
######################################################
# Dados - linhas: todas, colunas: ate a antepenultima
data =  dataSet.iloc[:,:-2].values

# Remove as colunas protocol_type, service e flag
# Ref: https://thispointer.com/delete-elements-rows-or-columns-from-a-numpy-array-by-index-positions-using-numpy-delete-in-python/
data = np.delete(data,[1,2,3], axis=1)

# Rotulos-  coluna que contem a classificacao do trafego (anomalo ou ataque)
labels = dataSet.iloc[:,42].values


######################################################
# PRÉ-PROC: 1) Codificacao da classificacao do trafego
######################################################

# Categorizacao dos tipos de ataque
attackType  = {'normal':"normal", 'neptune':"abnormal", 'warezclient':"abnormal", 'ipsweep':"abnormal",'back':"abnormal", 'smurf':"abnormal", 'rootkit':"abnormal",'satan':"abnormal", 'guess_passwd':"abnormal",'portsweep':"abnormal",'teardrop':"abnormal",'nmap':"abnormal",'pod':"abnormal",'ftp_write':"abnormal",'multihop':"abnormal",'buffer_overflow':"abnormal",'imap':"abnormal",'warezmaster':"abnormal",'phf':"abnormal",'land':"abnormal",'loadmodule':"abnormal",'spy':"abnormal",'perl':"abnormal"} 
labels[:] = [attackType[item] for item in labels[:]] 

# Categorizacao binaria (trafego normal = 0, anormal = 1)
attackEncodingCluster  = {'normal':0,'abnormal':1}
labels[:] = [attackEncodingCluster[item] for item in labels[:]]