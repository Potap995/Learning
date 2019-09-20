import numpy as np



def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def logloss(x, y):
    return - np.dot(y, np.log(x).T)

def logloss_grad(x, y):
    return -(y / x)

def softmax_grad(x):
    return np.diagflat(x) - np.dot(x.T, x)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

