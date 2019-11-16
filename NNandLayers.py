import numpy as np

def random_init(size, left=-0.1, right=0.1):
    return np.random.random(size=size) * (right - left) + left


class FullyConnectedLayer:

    def __init__(self, in_shape, out_shape, **params):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lr = params.pop("learning_rate", 0.01)
        self.W = random_init((in_shape, out_shape))
        self.b = np.zeros((1, out_shape))

    def changeGiperParams(self, **params):
        self.lr = params.pop("learning_rate", 0.01)

    def forward(self, data):
        self.in_data = data
        self.out_data = np.dot(data, self.W) + self.b
        return self.out_data

    def backward(self, loss):
        batch_size = loss.shape[0]

        grad_b = np.sum(loss, axis=0) / batch_size
        grad_W = np.dot(self.in_data.T, loss)

        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

        return np.dot(loss, self.W.T)


class SigmoidLayer:

    def __init__(self):
        pass

    def changeGiperParams(self, **params):
        pass

    def forward(self, data):
        self.in_data = data
        self.out_data = 1. / (1. + np.exp(-data))
        return self.out_data

    def backward(self, loss):
        deriv = self.out_data * (1 - self.out_data)
        return loss * deriv


class SoftMaxLayer:

    def __init__(self, params):
        self.lossfunc = params.pop("loss", "logloss")

    def changeGiperParams(self, **params):
        pass

    def forward(self, data):
        exps = np.exp(data)
        self.out_data = exps / np.sum(exps)
        return self.out_data

    def backward(self, deriv):
        if self.lossfunc == "logloss":
            return self.out_data - deriv
        else:
            # I don't know how to make dot product along first axes. Like [2, 3] and [3, 2] to [2, 3, 3]
            # Can be implemented in loop, but i don't want
            Id = np.eye(deriv.shape[-1])
            return deriv[..., np.newaxis] * Id


class NET:

    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def trainBatch(self, data, labels):
        for i in range(len(self.layers)):
            data = self.layers[i].forward(data)

        deriv = labels
        for i in range(len(self.layers) - 1, -1, -1):
            deriv = self.layers[i].backward(deriv)

    def changeGiperParams(self, **params):
        for layer in self.layers:
            layer.changeGiperParams(**params)

    def train(self, train_data, train_labels, valid_data, valid_labels, **params):
        epoch = params.pop("epoch", 1)
        batch_size = params.pop("batch_size", 1)
        assert train_labels.shape[0] == train_data.shape[0]
        assert valid_labels.shape[0] == valid_data.shape[0]
        train_len = train_labels.shape[0]

        test_accuracy = []
        train_accuracy = []

        for i in range(epoch):
            for batch_iter in range(0, train_len, batch_size):

                if batch_iter + batch_size < train_len:
                    X_cur = train_data[batch_iter: batch_iter + batch_size]
                    y_cur = train_labels[batch_iter: batch_iter + batch_size]
                else:
                    X_cur = train_data[batch_iter: train_len]
                    y_cur = train_labels[batch_iter: train_len]

                self.trainBatch(X_cur, y_cur)

            train_accuracy.append(self.eval(train_data, train_labels))
            test_accuracy.append(self.eval(valid_data, valid_labels))

        return train_accuracy, test_accuracy



    def eval(self, data, labels):
        for layer in self.layers:
            data = layer.forward(data)
        out_data = data
        out_id = np.argmax(out_data, axis=1)
        label_id = np.argmax(labels, axis=1)
        return np.sum(out_id == label_id) / float(labels.shape[0])
