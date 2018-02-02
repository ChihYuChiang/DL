import numpy as np
import time

'''
--------------------------------------------------
Vectorization exp
--------------------------------------------------
'''
if __name__ == "__main__":
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




'''
--------------------------------------------------
Basic functions
--------------------------------------------------
'''
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))
    return s

x = np.array([1, 2, 3])
sigmoid(x)


#--Reshape array
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v

#This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print ("image2vector(image) = " + str(image2vector(image)))


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x by norm 2 of x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    #Apply exp() element-wise to x.
    x_exp = np.exp(x)

    #Create a vector x_sum that sums each row of x_exp.
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    
    #Compute softmax(x) by dividing x_exp by x_sum.
    s = x_exp / x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))


#--L1 and L2 loss function
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    loss = np.sum(abs(yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    loss = np.dot(yhat - y, yhat - y)
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))