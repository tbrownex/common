import numpy as np

def splitLabel(df, config):
    ''' Input is a dictionary of dataframes
    Separate the Label from the data and put into distinct data dictionary keys.
    If the label is categorical then one-hot it
    '''
    d = {}
    
    # Create the Feature sets first
    d["trainX"] = df["train"].copy()
    del d["trainX"][config["labelColumn"]]
    if df["val"] is not None:
        d["valX"] = df["val"].copy()
        del d["valX"][config["labelColumn"]]
    if df["test"] is not None:
        d["testX"] = df["test"].copy()
        del d["testX"][config["labelColumn"]]
    
    # Now the Labels
    trainY = df["train"][config["labelColumn"]]
    if df["val"] is not None:
        valY  = df["val"][config["labelColumn"]]
    if df["test"] is not None:
        testY  = df["test"][config["labelColumn"]]
    
    if config["labelType"] == "continuous":
        d["trainY"] = trainY
        if df["val"] is not None:
            d["valY"]  = valY
        if df["test"] is not None:
            d["testY"]  = testY
    else:
        numClasses = config["numClasses"]
        d['trainY'] = np.eye(numClasses)[trainY].astype(np.float32)
        if df["val"] is not None:
            d['valY']  = np.eye(numClasses)[valY].astype(np.float32)
        if df["test"] is not None:
            d['testY']  = np.eye(numClasses)[testY].astype(np.float32)
    
    return d