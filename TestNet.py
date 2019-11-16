import numpy as np
from NNandLayers import *
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from SuppFunc import MinMaxNormalizer, test_train_split

net = NET()

net.addLayer(FullyConnectedLayer(4, 4))
net.addLayer(SigmoidLayer())

net.addLayer(FullyConnectedLayer(4, 3))
net.addLayer((SoftMaxLayer({"loss":"logloss"})))

net.changeGiperParams(lr=0.01)

data = load_iris()

X = data["data"]
y_lables = data["target"]

All_len = len(X)

y = np.zeros((len(y_lables), 3))
for i, a in enumerate(y_lables):
    y[i][a] = 1


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

train_a, test_a = net.train(X_train, y_train, X_test, y_test, epoch=400, batch_size=1)
time = np.arange(len(train_a))

plt.plot(time, train_a, "r")
plt.plot(time, test_a, "b")
plt.show()