import pandas as pd
import operator
'''
Passed in a DF and a threshold, return a list of column combination that are correlated above the threshold
    - "highCorr" is a tuple of a column name and its covariance with "col"
    - ignore the diagonal of covariance matrix ("col != idx")
    - only return unique column combinations
    - sort the list top-down: highest correlated first
'''
def checkCombo(d, col, idx):
    try:
        if d[(idx, col)]:         # Did we get the correlation for these two columns already?
            return True
    except:
        return False

def getHighCorr(df, threshold):
    corrDF = df.corr().abs()
    d = {}
    for col in corrDF.columns:
        highCorr = corrDF[col] > threshold
        for idx, cov in highCorr.items():
            if cov and col != idx:
                exists = checkCombo(d, col, idx)
                if exists:
                    pass
                else:
                    d[(col, idx)] = corrDF[col][idx]
    return sorted(d.items(), key=operator.itemgetter(1), reverse=True)