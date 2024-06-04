import numpy as np


class MaxScaler:
    def __init__(self):
        self.max = -np.inf

    def __str__(self):
        return 'MaxScaler()'

    def fit_transform(self, data):
        self.max = np.max(data, axis=0)
        data /= self.max
        return data

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max

