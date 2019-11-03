from matplotlib import pyplot as plt
import sys, nsl_kdd,  pre_processing, knn
import numpy as np

# Argumentos do script
if (len(sys.argv) != 3):
	print("Parametros: 1) Caminho do arquivo dataset, 2) Numero de vizinhos\n:")
	print("python main-knn.py caminho/para/dataset.csv 76")
	exit();
else:
	path = sys.argv[1]
	n_neighbors = int(sys.argv[2])


# Load dataset
data,labels = nsl_kdd.load_file(path)

# scaling
data = pre_processing.scaling(data)

# NearestNeighbors
print("########### NearestNeighbors ##########")

distances, indices = knn.nearest_neighbors(data,n_neighbors)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
print("* Distancias: ")
print("** Min: ",distances.min())
print("** Max: ",distances.max())
print("** Media: ",distances.mean())

# Interquartis - tentativa de encontrar o "ponto fora da curva" matematicamente, nao deu certo
#iq1 = np.percentile(distances,25)
#iq3 = np.percentile(distances,75)
#iqr = iq3 - iq1
#distance_outlier = (iqr*1.5)+iq3
#print("* Distancias Interquartis")
#print("** IQ1: {}\n** IQ3: {},\n** IQR: {}\n** Outlier: {}".format(iq1,iq3,iqr,distance_outlier))

# Graph
plt.plot(distances)
plt.show()