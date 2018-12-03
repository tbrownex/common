import numpy as np

def calcRMSE(actual, predictions):
    return np.sqrt(np.mean((actual-predictions)**2))