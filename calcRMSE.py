import numpy as np

def calcRMSE(actuals, predictions):
    shape = np.ndim(actuals)
    if shape > 1:
        actuals = np.reshape(actuals, newshape=[-1,])
    shape = np.ndim(predictions)
    if shape > 1:
        predictions = np.reshape(predictions, newshape=[-1,])
    rmse = np.sqrt(np.mean((actuals-predictions)**2))
    return round(rmse, 2)