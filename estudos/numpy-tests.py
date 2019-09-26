'''
Source: https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
'''

import numpy as np

np.random.seed(0)
x1 = np.random.randint(10,size=10)

#Array Slicing: Accessing Subarrays
# x[start:stop:step]
# If any of these are unspecified, they default to the values start=0, stop=size of dimension, step=1


#Multi-dimensional subarrays
x2 = np.random.randint(10,size=(10,10))
x2.view()

# => Multi-dimensional slices work in the same way, with multiple slices separated by commas.
#toda a matriz
x2[:,:]

#Metade da matriz - De 0,0 a 4,4
x2[:5,:5]

#So a primeira coluna - vai mostrar a posiaco 0,0; 1,0 ... 9,0
x2[:,:1]


# => Accessing array rows and columns
# This can be done by combining indexing and slicing, using an empty slice marked by a single colon (:)

# PRIMEIRA COLUNA - Mostra em forma de linha:
x2[:,0]

#PRIMEIRA LINHA 
x2[0,:]
#ou
x2[0]


'''
Subarrays as no-copy views
One important and extremely useful thing to know about array slices is that they return views rather than copies of the array data.
This is one area in which NumPy array slicing differs from Python list slicing: in lists, slices will be copies. 

If we modify this subarray, we'll see that the original array is changed!
'''
x2_sub = x2[:2, :2]
print(x2_sub)
x2_sub[0, 0] = 99
print(x2_sub)
print(x2)


'''
Creating copies of arrays
'''
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)

'''
Reshape array 
'''

# by reshape (should be same size)
grid = np.arange(1, 10)
grid.view()
print(grid.size)
print(grid.reshape((3, 3)))

#by newaxis
# row vector via newaxis
x = np.array([1, 2, 3])
x[np.newaxis, :]
x[:, np.newaxis]


# criar uma variavel ja com o array no formato desejado
grid = np.arange(16).reshape((4, 4))


'''
Concatenation
'''
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])