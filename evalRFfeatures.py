''' For a Random Forest, show the most important features

"cols" is a list of all the df columns
"model" is the RandomForestRegressor model'''

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def evalFeatures(dataDict):
    cols = dataDict["trainX"].columns
    
    regr = RandomForestRegressor(n_estimators=100)
    regr.fit(dataDict["trainX"], dataDict["trainY"])
    tmp = zip(cols, regr.feature_importances_)
    features = sorted(tmp, key=lambda tup: tup[1], reverse=True)

    print("{:<20}{}".format("Feature", "importance"))
    for x in features:
        print("{:<20}{:.2f}".format(x[0], x[1]))
    return features