import random

def generateHyperParms(d):
    hyperParms = {}
    # Get an L1 value
    start = d["L1Size"][0]
    end = d["L1Size"][1]
    hyperParms["l1Size"] = random.randint(start, end)
    # random choice of activation function
    hyperParms["activation"] = random.sample(d["activation"], 1)[0]
    # Batch size is powers of 2
    hyperParms["batchSize"] = 2 ** random.sample(d["batchSize"], 1)[0]
    # LR is 
    hyperParms["learningRate"] = 10 ** -random.sample(d["learningRate"], 1)[0]
    # Limits of a uniform dist
    start = d["dropout"][0]
    end = d["dropout"][1]
    hyperParms["dropout"] = random.uniform(start, end)
    return hyperParms