'''

Gerar matrizes com valores em escalas diferentes para estudar como a normalizacao vai ficar.
Cada coluna da matriz eh uma feature, cada linha uma amostra.

Referencias do NumPy:
https://docs.scipy.org/doc/numpy-1.15.0/genindex.html

'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler


size = 10000

# Feature 00 - Zero, um, dois, tres, quatro ou cinco
feature00 = np.random.choice([0.0,1.0,2.0,3.0,4.0,5.0],size)

# Feature 01 - De zero a 255
feature01 = np.random.random_sample(size) * 255

# Feature 02 - De zero a Cem mil
feature02 = np.random.random_sample(size) * 100000

# Feature 03 - De zero a mil
feature03 = np.random.random_sample(size) * 100

# Feature 04 - De zero a mil
feature04 = np.random.random_sample(size) * 100

# Feature 05 - De zero a 1
feature05 = np.random.random_sample(size)

# Feature 06 - De zero a 1
feature06 = np.random.random_sample(size)

# Feature 07 - Zero ou um
feature07 = np.random.choice([0.0,1.0],size)

# agrupamento das matrizes individuais
matrix = np.stack((feature00, feature01, feature02, feature03, feature04, feature05, feature06, feature07), axis=-1)

#
# NORMALIZACAO
#
# TODO: estudar o compativo
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
#

# MinMaxScaler
matrix_max_data = MinMaxScaler().fit(matrix).data_max_
matrix_scaled = MinMaxScaler().fit_transform(matrix)

#
# Exibicao dos resultadsos
#
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions
#

np.set_printoptions(suppress=True, precision=1, linewidth=160)
print("########### MATRIX GERADA (10 primeiras linhas) #############")
print(matrix[:10,:]) # Mostra as dez primeiras linhas (:10) e todas as colunas (:)
print("")

np.set_printoptions(suppress=True, precision=5, linewidth=160)
print("########### MATRIX NORMALIZADA  - (10 primeiras linhas) #############")
print("\n - MinMaxScaler:")
print("Max data: ", matrix_max_data)
print("\n",matrix_scaled[:10,:]) # Mostra dez primeiras linhas (:10) e todas as colunas (:)

