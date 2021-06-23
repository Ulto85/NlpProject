import numpy as np

array1 = np.arange(0,25)
array2 = np.arange(25,50)

array3 = np.array([array1,array2])

print(array3.reshape(-1,1))