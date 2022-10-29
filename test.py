import numpy as np

X = np.array([1,0,1,0,1,0,1,0,1,0])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# check what this output does
print(np.where(X*y==0))