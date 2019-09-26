'''

Gerar matrizes com valores em escalas diferentes para estudar como a normalizacao vai ficar.
Cada coluna da matriz eh uma feature, cada linha uma amostra.

Serao geradas 5 colunas com 25 linhas
Duas das cinco colunas terao escalas bem destoante das demais

Referencias:
https://docs.scipy.org/doc/numpy-1.15.0/genindex.html

'''
import numpy as np
from sklearn import preprocessing

size = 5

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

# NORMALIZACAO
# https://scikit-learn.org/stable/modules/preprocessing.html#normalization
matrix_normalized = preprocessing.normalize(matrix, norm='l2')

#
# Exibicao dos resultadsos
#

# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions
np.set_printoptions(suppress=True, precision=1, linewidth=160)

print("########### MATRIX GERADA #############")
print(matrix.view())
print("")
np.set_printoptions(suppress=True, precision=5, linewidth=160)
print("########### MATRIX NORMALIZADA #############")
print(matrix_normalized.view())

'''
https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

5.3.1.3. Scaling data with outliers
If your data contains many outliers, scaling using the mean and variance of the data is likely to not work very well. In these cases, you can use robust_scale and RobustScaler as drop-in replacements instead. They use more robust estimates for the center and range of your data.
'''


