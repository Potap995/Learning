import numpy as np

def random_init(size, left=-0.1, right=0.1):
    return np.random.random(size=size) * (right - left) + left

class FullyConnectedLayer:

    def __init__(self, in_shape, out_shape, **params):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lr = params.pop("learning_rate", 0.01)
        self.w = random_init((in_shape, out_shape))
        self.b = np.zeros((1, out_shape))

    def forward(self, data):
        self.in_data = data
        self.out_data = np.dot(data, self.w) + self.b
        return self.out_data

    def backward(self, deriv):
        pass


class SigmoidLayer:

    def __init__(self):
        pass

    def forward(self, data):
        self.in_data = data
        self.out_data =  1. / (1. + np.exp(-data))
        return self.out_data

    def backward(self, deriv):
        pass


class SoftMaxLayer:

    def __init__(self):
        pass

    def forward(self, data):
        pass

    def backward(self, deriv):
        pass


class NET:

    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

