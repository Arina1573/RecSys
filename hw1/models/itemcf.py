import numpy as np
from scipy import sparse


class ItemCF:
    def __init__(self, neighbors=100, shrinkage=10, min_common=2):
        self.neighbors = neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common

    def fit(self, train_matrix):
        self.train_matrix = train_matrix.tocsr().astype(np.float32)
        common = (self.train_matrix.T @ self.train_matrix).tocoo()
        norms = np.sqrt(np.asarray(self.train_matrix.sum(axis=0)).ravel())
        denom = norms[common.row] * norms[common.col]
        cosine = common.data / np.maximum(denom, 1e-12)
        values = cosine * common.data / (common.data + self.shrinkage)
        values[common.data < self.min_common] = 0
        similarity = sparse.csr_matrix(
            (values, (common.row, common.col)), shape=(self.train_matrix.shape[1],) * 2
        )
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        rows, cols, data = [], [], []
        for item in range(similarity.shape[0]):
            start, end = similarity.indptr[item:item + 2]
            chosen = np.argsort(similarity.data[start:end])[-self.neighbors:]
            rows.extend([item] * len(chosen))
            cols.extend(similarity.indices[start:end][chosen])
            data.extend(similarity.data[start:end][chosen])
        self.similarity = sparse.csr_matrix((data, (rows, cols)), shape=similarity.shape)
        return self

    def score(self, users):
        return (self.train_matrix[users] @ self.similarity).toarray()
