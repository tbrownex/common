from dateutil import parser
import pandas as pd
import logging

__author__ = "Tom Browne"

def getData(config, sep=","):
    ''' Read the input file into a dataframe'''
    df = pd.read_csv(config["dataLoc"]+config["fileName"], sep=sep)
    return df