''' Create an XGBoost model on training data and return predictions from test data

"parms" is a dictionary of the model parameters
"X" is a Dataframe of features
"Y" can be a Series or numpy array. Y values may come in as one-hot encoded. If so need to just get the class value because Classifier doesn't like 2 dimensional label  '''

import xgboost as xgb
import numpy as np

def fitXGB(parms, X, Y, typ):
    assert typ in ["C", "R"], "specify Classifier or Regressor with 'C' or 'R'"
    
    if typ == "R":
        m = xgb.XGBRegressor(**parms)
    else:
        m = xgb.XGBClassifier(**parms)
        if Y.ndim == 2:
            labels = np.argmax(Y, axis=1)
        else:
            labels = Y
    return m.fit(X, labels)