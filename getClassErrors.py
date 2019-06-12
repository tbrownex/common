import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score

from oneHot import oneHot

def calcAccuracy(actuals, predictions):
    idx = np.argmax(predictions, axis=1)
    predOneHot = oneHot(idx)
    return accuracy_score(actuals, predOneHot)

def calcF1(actuals, predictions):
    idxP = np.argmax(predictions, axis=1)
    idxA = np.argmax(actuals, axis=1)
    return f1_score(idxA, idxP)

def calcRecall(actuals, predictions):
    '''
    - "posIdx" holds the indeces for each actual Positive
    - "posCount" has the number of indeces: same as the count of actual Positives
    - "preds" converts probabilities into [0,1] format
    - "TP": indexing the preds by the actual Positives, then sum to get the count of True Positives
    '''
    posIdx = np.where(actuals[:,1]==1)[0]
    posCount = posIdx.shape[0]
    preds = np.argmax(predictions, axis=1)
    TP = preds[posIdx].sum()     # how many True Positives there are
    return TP/posCount

def getClassErrors(actuals, predictions):
    '''
    For Classification only
    Inputs are numpy arrays that are one-hot encoded
    Any number of Classes is ok
    '''
    assert (actuals.shape == predictions.shape), "Shape mismatch"
    d = {}
    d["ll"]  = log_loss(actuals, predictions)
    d["auc"] = roc_auc_score(actuals, predictions)
    d["acc"] = calcAccuracy(actuals, predictions)
    d["f1"]  = calcF1(actuals, predictions)
    d["recall"] = calcRecall(actuals, predictions)
    return d