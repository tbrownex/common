from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def normalize(dataDict, method):
    '''
    dataDict is a dictionary of dataframes for Train/Val/Test
    Normalize Train using "fit" which will set the Mean and StdDev. Then "transform" all the data using
    those values from "fit"
    '''
    assert method in ['MinMax', 'Std'],  "Invalid scaling method"
    
    cols = dataDict["trainX"].columns
    if method =='MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    scaler.fit(dataDict["trainX"])
    arr = scaler.transform(dataDict["trainX"])
    dataDict["trainX"] = pd.DataFrame(arr, columns=cols)
    if "valX" in dataDict.keys():
        arr = scaler.transform(dataDict["valX"])
        dataDict["valX"] = pd.DataFrame(arr, columns=cols)
    if "testX" in dataDict.keys():
        arr = scaler.transform(dataDict["testX"])
        dataDict["testX"] = pd.DataFrame(arr, columns=cols)
    return dataDict