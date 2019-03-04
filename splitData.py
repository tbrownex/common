from sklearn.model_selection import train_test_split
import pandas as pd
import numpy  as np

def splitData(data, config):
    '''
    Incoming dataframe will be split into Train and Test and returned as a dictionary
    '''
    d = {}
    svCols = data.columns
    
    train, test = train_test_split(data, test_size=config["testPct"])
    train.columns = svCols
    test.columns  = svCols
    d["train"] = train
    d["test"]  = test
    return d