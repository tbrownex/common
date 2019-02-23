import numpy as np

def calcMAPE(actuals, predictions):
    shape = np.ndim(actuals)
    if shape > 1:
        actuals = np.reshape(actuals, newshape=[-1,])
    shape = np.ndim(predictions)
    if shape > 1:
        predictions = np.reshape(predictions, newshape=[-1,])
    mape = np.mean(np.abs((actuals - predictions) / actuals))
    return round(mape,2)