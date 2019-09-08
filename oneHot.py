import numpy as np

def oneHot(arr):
    '''
    One-hot encode the input np array
    For very imbalanced data like Fraud Detection, sometimes the "max" is zero...if a batch has all negatives. So make sure at minimum you have 2 columns
    '''
    arr = arr.astype(int)   # in case it's float
    rows = arr.shape[0]
    cols = int(np.max(arr)+1)
    if cols ==1:
        cols =2
    z = np.zeros((rows, cols))
    z[np.arange(rows), arr] = 1
    return z