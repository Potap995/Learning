import numpy as np


class MinMaxNormalizer():  # to [0, 1]
    def __init(self):
        self.max = 0
        self.min = 0

    def fit(self, x):
        self.max = np.max(x)
        self.min = np.min(x)

    def normalize(self, x):
        x -= self.min
        return x / self.max

    def fit_and_normalize(self, x):
        self.fit(x)
        return self.normalize(x)


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def logloss(x, y):
    return - np.dot(y, np.log(x).T)

def logloss_grad(x, y):
    return -(y / x)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax_grad(x):
    return np.diagflat(x) - np.dot(x.T, x)


def check_shape_1(X, y):
    if X.shape !=  (1, 4):
        raise ValueError("X : Expect shape (1, 4), given {0}".format(X.shape))
    if y.shape !=  (1, 3):
        raise ValueError("X : Expect shape (1, 3), given {0}".format(y.shape))


def forward_prop_1(X, y, params):
    params["L0"] = X

    params["L1"] = np.dot(params["L0"], params["W0"]) + params["b0"]
    params["P1"] = sigmoid(params["L1"])

    # params["P1"] = X

    params["L2"] = np.dot(params["P1"], params["W1"]) + params["b1"]
    params["P2"] = softmax(params["L2"])

    params["E"] = logloss(params["P2"], y)


def back_prop_1(X, y, params, giper_params):
    check_shape_1(X, y)

    params["dE/dP2"] = - y / params["P2"]  # 1x3
    params["dP2/dL2"] = softmax_grad(params["P2"])  # 3x3
    params["dE/dL2"] = np.dot(params["dE/dP2"], params["dP2/dL2"])  # 1x3
    params["dE/dW1"] = np.dot(params["P1"].T, params["dE/dL2"])  # 4x3
    params["dE/db1"] = params["dE/dL2"]

    params["dE/dP1"] = np.dot(params["dE/dL2"], params["W1"].T)  # 1x4
    params["dP1/dL1"] = params["P1"] * (1 - params["P1"])  # 1x4
    params["dE/dL1"] = params["dE/dP1"] * params["dP1/dL1"]  # 1x4
    # params["dE/dL1"] = np.dot(params["dE/dL2"], params["W1"].T)
    params["dE/dW0"] = np.dot(params["L0"].T, params["dE/dL1"])  # 4x4
    params["dE/db0"] = params["dE/dL1"]

    params["W1"] -= giper_params["learning rate"] * params["dE/dW1"]
    params["b1"] -= giper_params["learning rate"] * params["dE/db1"]

    params["W0"] -= giper_params["learning rate"] * params["dE/dW0"]
    params["b0"] -= giper_params["learning rate"] * params["dE/db0"]


def random_init(size, left=-0.1, right=0.1):
    return np.random.random(size=size) * (right + left) + left

def net_init_1(params):
    params["L0"] = np.zeros((1, 4))

    params["W0"] = random_init((4, 4))
    params['b0'] = random_init((1, 4))

    params["L1"] = np.zeros((1, 4))
    params["P1"] = np.zeros((1, 4))

    params["W1"] = random_init((4, 3))
    params['b1'] = random_init((1, 3))

    params["L2"] = np.zeros((1, 3))
    params["P2"] = np.zeros((1, 3))

    params["E"] = np.zeros((1, 1))

