import numpy as np

def splitLabel(df, config):
    ''' Input is a dictionary of dataframes
    Separate the Label from the data and put into distinct data dictionary keys.
    See if the Label should be one-hot encoded
    '''
    d = {}
    
    # Create the Feature sets first
    d["trainX"] = df["train"].copy()
    del d["trainX"][config["labelColumn"]]
    if "val" in df.keys():
        d["valX"] = df["val"].copy()
        del d["valX"][config["labelColumn"]]
    if "test" in df.keys():
        d["testX"] = df["test"].copy()
        del d["testX"][config["labelColumn"]]
    
    # Now the Labels
    trainY = df["train"][config["labelColumn"]]
    if "val" in df.keys():
        valY  = df["val"][config["labelColumn"]]
    if "test" in df.keys():
        testY  = df["test"][config["labelColumn"]]
    
    if config["labelType"] == "continuous":
        d["trainY"] = trainY
        if "val" in df.keys():
            d["valY"]  = valY
        if "test" in df.keys():
            d["testY"]  = testY
            
    if config["oneHot"]:
        numClasses = config["numClasses"]
        d['trainY'] = np.eye(numClasses)[trainY].astype(np.float32)
        if "val" in df.keys():
            d['valY']  = np.eye(numClasses)[valY].astype(np.float32)
        if "test" in df.keys():
            d['testY']  = np.eye(numClasses)[testY].astype(np.float32)
    else:
        d['trainY'] = trainY
        if "val" in df.keys():
            d['valY'] = valY
        if "test" in df.keys():
            d['testY'] = testY
    
    return d