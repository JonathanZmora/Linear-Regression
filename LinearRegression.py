import numpy as np


class LinearRegression:

    def __init__(self):
        self._weights = None

    def fit(self, X, y):
        Xb = np.c_[X, np.ones(X.shape[0])]

        try:
            self._weights = np.linalg.inv(Xb.T @ Xb + np.eye(Xb.shape[1])) @ Xb.T @ y

        except np.linalg.LinAlgError:
            print("cannot invert a singular matrix")

    def predict(self, X):
        Xb = np.c_[X, np.ones(X.shape[0])]
        return np.dot(Xb, self._weights)

    def score(self, X, y):
        y_predicted = self.predict(X)
        u = ((y - y_predicted) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)

    @property
    def weights(self):
        return self._weights
