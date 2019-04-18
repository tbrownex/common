import pandas as pd

__author__ = "Tom Browne"

def getData(config, sep=",", rows=None, colTypes=None):
    ''' Read the input file into a dataframe'''
    df = pd.read_csv(config["dataLoc"]+config["fileName"], sep=sep, nrows=rows, dtype=colTypes)
    return df