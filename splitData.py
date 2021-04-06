from sklearn.model_selection import train_test_split
import pandas as pd

def splitData(data, config, stratify=None, seed=None):
    '''
    Incoming dataframe will be split into Train/Val/Test and returned as a dictionary
    If "stratify' is set, then the splits will have balanced classes (for classification)
    '''
    d = {}
    svCols = data.columns
    
    if config["testPct"] > 0:
        if stratify:
            train, test = train_test_split(data, test_size=config["testPct"], stratify=data[stratify], random_state =seed)
        else:
            train, test = train_test_split(data, test_size=config["testPct"], random_state =seed)
        train.columns = svCols
        test.columns  = svCols
        d["train"] = train
        d["test"]  = test
    else:
        d["train"] = data 
        
    if config["valPct"] > 0:
        if stratify:
            train, val = train_test_split(d["train"], test_size=config["valPct"], stratify=d["train"][stratify], random_state =seed)
        else:
            train, val = train_test_split(d["train"], test_size=config["valPct"], random_state =seed)
        train.columns = svCols
        val.columns  = svCols
        d["train"] = train
        d["val"]   = val
    return d