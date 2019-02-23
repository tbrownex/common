from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np

def splitData(data, config):
    '''
    Incoming dataframe will be split into Train/Val/Test and returned as a dictionary
    '''
    d = {}
    d["train"] = None
    d["val"]   = None
    d["test"]  = None
    
    svCols = data.columns
    
    if config["testPct"] > 0:
        train, test = train_test_split(data, test_size=config["testPct"])
        test.columns = svCols
        d["test"] = test
    if config["valPct"] > 0:
        train, val = train_test_split(train, test_size=config["valPct"])
        train.columns = svCols
        val.columns   = svCols
        d["train"]    = train
        d["val"]      = val
    else:
        d["train"] = data
        return d