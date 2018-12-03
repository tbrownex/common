import numpy as np

def calcMAPE(actual, predictions):
    return np.mean(np.abs((actual - predictions) / actual))