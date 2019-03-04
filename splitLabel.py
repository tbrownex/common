import numpy as np

def splitLabel(data, config):
    '''Inputs is a dictionary of dataframes
    Separate the Label from the data and put into distinct data dictionary keys.
    If the label is categorical (not conitnuous) then one-hot it
    '''
    LABELS = np.array([0,1])
    d = {}
    
    # Create the Feature sets first
    d["trainX"] = data["train"].copy()
    del d["trainX"][config["labelColumn"]]
    if data["test"] is not None:
        d["testX"] = data["test"].copy()
        del d["testX"][config["labelColumn"]]
    
    # Now the Labels
    trainY = data["train"][config["labelColumn"]]
    if data["test"] is not None:
        testY  = data["test"][config["labelColumn"]]
    
    if config["labelType"] == "continuous":
        d["trainY"] = trainY
        if data["test"] is not None:
            d["testY"]  = testY
    else:
        d['trainY'] = (LABELS == trainY).astype(np.float32)     # This will one-hot the label
        if data["test"] is not None:
            d['testY']  = (LABELS == testY).astype(np.float32)
    
    return d