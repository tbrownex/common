import pickle

def loadPickle(file):
    tmp = open(file,'rb')
    return pickle.load(tmp)