import pandas as pd

def analyzeCols(df):
    '''
    Incoming dataframe has all the columns from the Training set.
    Remove any columns that are static, i.e. have only a single value
    '''
    cols = df.columns
    remove = []
    for col in cols:
        tmp = df[col].unique()
        if tmp.shape[0] == 1:
            remove.append(col)
    if len(remove) > 0:
        print("\nRemoving columns:")
        for col in remove:
            print(" -", col)
    else:
        print("\n - No columns to remove")
    return remove