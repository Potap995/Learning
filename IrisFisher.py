from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from seaborn import pairplot
from pandas import DataFrame
from SuppFunc import MinMaxNormalizer, test_train_split


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


def count_accuracy(X, y, params):
    error_counter = 0
    for iter, item in enumerate(y):
        X_cur = X[iter]
        y_cur = y[iter]
        forward_prop_1(X_cur, y_cur, params)
        if np.argmax(y_cur) != np.argmax(params["P2"]):
            error_counter += 1
    return (len(y) - error_counter) / len(y)


#Doesn't work
def softmax_grad_2(x):
    # I don't know how to make dot product along first axes. Like [2, 3] and [3, 2] to [2, 3, 3]
    # Can be implemented in loop, but i don't want
    Id = np.eye(x.shape[-1])
    return x[..., np.newaxis] * Id


def random_init(size, left=-0.1, right=0.1):
    return np.random.random(size=size) * (right - left) + left

def net_init_1(params):
    np.random.seed(1)
    params["L0"] = np.zeros((1, 4))

    params["W0"] = random_init((4, 4))
    params['b0'] = np.zeros((1, 4))

    params["L1"] = np.zeros((1, 4))
    params["P1"] = np.zeros((1, 4))

    params["W1"] = random_init((4, 3))
    params['b1'] = np.zeros((1, 3))

    params["L2"] = np.zeros((1, 3))
    params["P2"] = np.zeros((1, 3))

    params["E"] = np.zeros((1, 1))



def forward_prop_1(X, y, params):
    params["L0"] = X

    params["L1"] = np.dot(params["L0"], params["W0"]) + params["b0"]
    params["P1"] = sigmoid(params["L1"])

    # params["P1"] = X

    params["L2"] = np.dot(params["P1"], params["W1"]) + params["b1"]
    params["P2"] = softmax(params["L2"])

    params["E"] = logloss(params["P2"], y)


def back_prop_2(X, y, params, giper_params):
    #Can simplify it to
    # params["dE/dP2"] = - y / params["P2"]  # 1x3
    # params["dP2/dL2"] = softmax_grad_2(params["P2"])  # 3x3
    # params["dE/dL2"] = np.dot(params["dE/dP2"], params["dP2/dL2"])  # 1x3
    # to
    params["dE/dL2"] = params["P2"] - y

    params["dE/dW1"] = np.dot(params["P1"].T, params["dE/dL2"])  # 4x3
    params["dE/db1"] = np.sum(params["dE/dL2"], axis=0)

    params["dE/dP1"] = np.dot(params["dE/dL2"], params["W1"].T)  # 1x4
    params["dP1/dL1"] = params["P1"] * (1 - params["P1"])  # 1x4
    params["dE/dL1"] = params["dE/dP1"] * params["dP1/dL1"]  # 1x4
    params["dE/dW0"] = np.dot(params["L0"].T, params["dE/dL1"])  # 4x4
    params["dE/db0"] = np.sum(params["dE/dL1"], axis=0)

    params["W1"] -= giper_params["learning rate"] * (params["dE/dW1"] / giper_params["batch size"])
    params["b1"] -= giper_params["learning rate"] * (params["dE/db1"] / giper_params["batch size"])

    params["W0"] -= giper_params["learning rate"] * (params["dE/dW0"] / giper_params["batch size"])
    params["b0"] -= giper_params["learning rate"] * (params["dE/db0"] / giper_params["batch size"])


data = load_iris()

X = data["data"]
y_lables = data["target"]

All_len = len(X)

y = np.zeros((len(y_lables), 3))
for i, a in enumerate(y_lables):
    y[i][a] = 1

df = DataFrame(X)
df["labels"] = y_lables

X = X.reshape(-1, 4)
y = y.reshape(-1, 3)

TEST_SIZE = 30

test_len = TEST_SIZE
train_len = All_len - test_len

# np.random.seed(1)
X_train, y_train, X_test, y_test = test_train_split(X, y, shuffle=True, test_size=(test_len / All_len))


normalizer = MinMaxNormalizer()
X_train = normalizer.fit_and_normalize(X_train)
X_test = normalizer.normalize(X_test)

params = dict()

giper_params = dict()
giper_params["learning rate"] = 0.01
giper_params["batch size"] = 1

lenX = len(y)

net_init_1(params)

test_accuracy = []
train_accuracy = []
time = []

iteration = 400
batch_size = giper_params["batch size"]

for iter in range(iteration):
    for batch_iter in range(0, train_len, batch_size):

        if batch_iter + batch_size < train_len:
            X_cur = X_train[batch_iter: batch_iter + batch_size]
            y_cur = y_train[batch_iter: batch_iter + batch_size]
        else:
            X_cur = X_train[batch_iter: train_len]
            y_cur = y_train[batch_iter: train_len]

        forward_prop_1(X_cur, y_cur, params)
        back_prop_2(X_cur, y_cur, params, giper_params)
    print("---------------")
    print("W0 = ", params["W0"])
    print("P2 = ", params["P2"])
    print("E = ", params["E"])
    print("dE = ",  params["dE/db1"])
    time.append(iter)
    test_accuracy.append(count_accuracy(X_test, y_test, params))
    train_accuracy.append(count_accuracy(X_train, y_train, params))

# for i in range(train_len * 400):
#     X_cur = X_train[i % train_len].reshape(1, -1)
#     y_cur = y_train[i % train_len].reshape(1, -1)
#     forward_prop_1(X_cur, y_cur, params)
#     back_prop_2(X_cur, y_cur, params, giper_params)
#     if (i % train_len) == train_len - 1:
#         time.append(i // train_len)
#
#         test_accuracy.append(count_accuracy(X_test, y_test, params))
#         train_accuracy.append(count_accuracy(X_train, y_train, params))

plt.plot(time, train_accuracy, "r")
plt.plot(time, test_accuracy, "b")
plt.show()
