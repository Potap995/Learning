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

def UnisonShuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def test_train_split(X, y, **options):
    lenX = len(X)

    shuffle = options.pop('shuffle', False)
    test_size = options.pop('test_size', 0.1)

    if shuffle:
        X, y = UnisonShuffle(X, y)

    test_elems = np.random.choice(lenX, int(lenX * test_size), replace=False)
    test_mask = np.zeros(lenX, dtype=bool)
    test_mask[test_elems] = True

    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]