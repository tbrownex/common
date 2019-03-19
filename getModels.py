import glob

def getModels(typ, config):
    '''
    Models have been trained and saved. Get all the models for the "typ" passed
    '''
    assert (typ in ["RF", "XGB", "NN", "AE"]), "invalid Model Type"
    
    if typ == "RF":
        m = config["modelDir"]+"RFmodel*"
    elif typ == "NN":
        m = config["modelDir"]+"NN*"
    elif typ == "XGB":
        m = config["modelDir"]+"XGB*"
    elif typ == "AE":         # AutoEncoder for outlier detection
        m = config["modelDir"]+"AE*"
    
    return glob.glob(m)