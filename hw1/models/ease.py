import numpy as np
from scipy import linalg


class EASE:
    def __init__(self, reg=300):
        self.reg = reg

    def fit(self, train_matrix):
        self.train_matrix = train_matrix.tocsr()
        gram = (self.train_matrix.T @ self.train_matrix).toarray().astype(np.float64)
        np.fill_diagonal(gram, np.diag(gram) + self.reg)
        factor = linalg.cho_factor(gram, lower=True)
        precision = linalg.cho_solve(factor, np.eye(gram.shape[0]))
        self.coef = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.coef, 0)
        return self

    def score(self, users):
        return np.asarray(self.train_matrix[users] @ self.coef)
