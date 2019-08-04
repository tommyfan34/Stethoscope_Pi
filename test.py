import numpy as np
x=np.array([[1,2,3]]).T
y=np.array([[4,6],[2,2]])
print(x.shape)
print(y.shape)
z=x.flatten()
print(z)
print(z.shape)
a=[0,2,4]
b=np.arange(0,10)
c=b[a]
print(c)

