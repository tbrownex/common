''' This is for neural network: when you are happy with the initial weights of a model, save them. Then load them for all subsequent models during training. This makes model comparison easier since they're all working on the same initial conditions'''
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)