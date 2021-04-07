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

def calcF1(precision, recall):
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 2)

def calcG(precision, recall):
    # This is an alternative score for unbalanced data
    g = np.sqrt(precision * recall)
    return round(g, 2)

def calcMAPE(actuals, predictions):
    shape = np.ndim(actuals)
    if shape > 1:
        actuals = np.reshape(actuals, newshape=[-1,])
    shape = np.ndim(predictions)
    if shape > 1:
        predictions = np.reshape(predictions, newshape=[-1,])
    mape = np.mean(np.abs((actuals - predictions) / actuals))
    return round(mape,2)