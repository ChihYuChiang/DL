import numpy as np
import time

'''
--------------------------------------------------
Vectorization exp
--------------------------------------------------
'''
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print("Vectorized: " + str(1000 * (toc - tic)) + " ms")

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print("Un-vectorized: " + str(1000 * (toc - tic)) + " ms")




'''
--------------------------------------------------
Broadcasting
--------------------------------------------------
'''
A = np.array([
    [56.0, 0.0, 4.4, 68.0],
    [1.2, 104.0, 52.0, 8.0],
    [1.8, 135.0, 99.0, 0.9]
])

cal = A.sum(axis=0)
print(cal)

percentage = 100 * A / cal.reshape(1, 4)
print(percentage)


#--Avoid potential confusion: do not use 1-dimension array
a = np.random.randn(5)
a.shape
a = np.random.randn(5, 1)
a.shape