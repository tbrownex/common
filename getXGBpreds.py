''' A model has been built. Get the predictions for the features passed in.
At this time I could be passing either XGB or RF model '''

def getpreds(model, X):
    return model.predict(X)