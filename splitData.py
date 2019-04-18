from sklearn.model_selection import train_test_split
import pandas as pd

def splitData(data, config):
    '''
    Incoming dataframe will be split into Train/Val/Test and returned as a dictionary
    '''
    d = {}
    svCols = data.columns
    
    if config["testPct"] > 0:
        train, test = train_test_split(data, test_size=config["testPct"])
        train.columns = svCols
        test.columns  = svCols
        d["train"] = train
        d["test"]  = test
    else:
        d["train"] = data 
        
    if config["valPct"] > 0:
        train, val = train_test_split(d["train"], test_size=config["valPct"])
        train.columns = svCols
        val.columns  = svCols
        d["train"] = train
        d["val"]   = val
    return d