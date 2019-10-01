# encoding: utf-8

'''
Tentativa de plotar os pontos do dataset NSK-KDD.
1) O dataset será ormalizado (minmaxscaler)
2) Será criada uma matrix (Nx2) em que cada par sera a media de cada linha da matrix do dataset normalizado
	2.1) A media deve ser calculado sobre o numero do colunas iniciais. Nao deve ser usado o pos normalizacao porque o onehot vai criar mais colunas ... alias essa bagaca nao fara nenhuma diferenca, porque em todas as conexoes a soma dessas clunas vai agregar um ponto. Posso fazer um teste aleatorio, dividindo por 42 mesmo, so pra ver o resultado.
	
	A matriz com as medias pode ser gerada ao percorrer um range com o numero de linhas, somar os valores da linha toda (usando slice) e dividindo por 42.
	
3) Os pares serão plotados 

4) analisar se os pontos mais estranhos sao os de trafego anomalo;

ADICIONAL: verificar qual é a distância entre os vizinhos mais próximos entre os pontos criados a partir da média dos dados.

Adicional 2: remover as colunas de texto porque elas nada contribuem para a detecção
'''

import numpy as np
from dataset import labels, data

######################################################
# CANCELADO: AS COLUNAS COM TEXTO FORAM REMOVIDAS POIS COM NADA CONTRIBUIAM
# PRÉ-PROC: 2) Codificacao OneHot dos descritores com valores textuais
#
# Refs:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer
######################################################
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
#data = transform.fit_transform(data)

######################################################
# PRÉ-PROC: 3) Escalonar os dados para que todas os descritores fiquem com valores entre zero e 1
######################################################
from sklearn.preprocessing import MinMaxScaler
data_max = MinMaxScaler().fit(data).data_max_
data = MinMaxScaler().fit_transform(data)

print("\n [i] Ponto com os maiores valores na normalizacao: \n",data_max)


######################################################
# NEAREST NEIGHBOTS
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
######################################################
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

neigh = NearestNeighbors(n_neighbors=5, algorithm='ball_tree',metric='euclidean', n_jobs=2)
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)

print("########################################")
print("#","NearestNeighbors".center(40),"#")
print("########################################")
print('\n[R] Indices (10 primeiros):')
print(indices[:10,:])
print('\n[R] Distancias (10 primeiras):')
print(distances[:10,:])

distances = np.sort(distances, axis=0)
distances = distances[:,1]
print("Menor: ",distances.min())
print("Maior: ",distances.max())
print("Media: ",distances.mean())


'''
Teste de ideias para encontrar o ponto proeminente
	Primeira diff maior que a distancia anteior: nao da certo
	
	Primeira distancia outlier: http://mathworld.wolfram.com/Outlier.html
	
	1o IQ: 25%
	3o IQ: 75%
	
	IQR = IQ3 - IQ1
	
	A convenient definition of an outlier is a point which falls more than 1.5 times the interquartile 
	range above the third quartile or below the first quartile.
	
	Outlier if:
		Valor >= (1.5 * IQR) + IQ3
		OR:
		Valor <= IQ1 - (1.5 * IQR)
	
	Ou seja, se o IQR = 3, o IQ1 = 3 e o IQ3 = 9, então oulier será <= (3 - 1.5*3

'''

iq1 = np.percentile(distances,25)
iq3 = np.percentile(distances,75)
iqr = iq3 - iq1
distance_outlier = (iqr*1.5)+iq3
print("\n######### INTERQUARTIL ########")
print("IQ1: {}\nIQ3: {},\nIQR: {}\nOutlier: {}".format(iq1,iq3,iqr,distance_outlier))

###########################
# Exibicao do grafico com as distancias
###########################
plt.plot(distances)
plt.show()