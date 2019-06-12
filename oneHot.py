import numpy as np

def oneHot(arr):
    '''
    One-hot encode the input np array
    '''
    arr = arr.astype(int)   # in case it's float
    rows = arr.shape[0]
    cols = int(np.max(arr)+1)
    z = np.zeros((rows, cols))
    z[np.arange(rows), arr] = 1
    return z