import numpy as np


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def logloss(x, y):
    return - np.dot(y, np.log(x))

def logloss_grad(x, y):
    return -(y / x)

def softmax_grad(x):
    # Let x is col
    x = x.reshape(3, 1)
    return np.diagflat(x) - np.dot(x, x.T)

def forward_prop(X, y, W, A):
    A[0] = np.dot(X, W[0])
    A[1] = softmax(A[0])
    L = logloss(A[1], y)

def back_prop(X, y, W, A, giper_params):
    dLdP = logloss_grad(A[1], y).reshape(1, 3) #1x3
    dPda = softmax_grad(A[1]) # 3x3
    dLda = np.dot(dLdP, dPda).T #3x1
    print(dLda, X)
    print("--------------------------------------------------------------")
    dLdW = np.dot(dLda, X)
    W[0] -= giper_params["learning rate"] * dLdW

