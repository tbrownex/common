from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def predict(parms, dataDict, typ):
    assert typ in ["C", "R"], "specify Classifier or Regressor with 'C' or 'R'"
    
    if typ == "R":
        m = RandomForestRegressor(**parms)
    else:
        m = RandomForestClassifier(**parms)
    model = m.fit(dataDict["trainX"], dataDict["trainY"])
    preds = model.predict(dataDict["testX"])
    return model, preds

