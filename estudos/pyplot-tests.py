import numpy as np
from matplotlib import pyplot as plt


#numpy.arange([start, ]stop, [step, ]dtype=None)  Return evenly spaced values within a given interval.
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
#x = np.arange(1,11) 


# Understanding Data Types in Python
# https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html

# The Basics of NumPy Arrays
# https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
#

xy = np.random.randint(100,size=(15,2))
print(xy.view())

# The axis() command takes a list of [xmin, xmax, ymin, ymax] and specifies the viewport of the axes
plt.axis([0,100,0,100])

#Plot y versus x as lines and/or markers.
plt.plot(xy[:,0],xy[:,1],'ob')
plt.show()

'''
https://matplotlib.org/3.1.1/tutorials/introductory/pyplot.html
https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm

https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
'''