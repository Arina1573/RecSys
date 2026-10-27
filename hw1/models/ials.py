import numpy as np


class IALS:
    def __init__(self, factors=32, reg=0.1, alpha=12, iterations=10, seed=42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, train_matrix, targets=None):
        X = train_matrix.tocsr()
        T = X if targets is None else targets.tocsr()
        if X.shape != T.shape:
            raise ValueError('Target matrix must match the training matrix')
        n_users, n_items = X.shape
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.05, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.05, (n_items, self.factors))
        eye = np.eye(self.factors)
        X_t = X.T.tocsr()
        T_t = T.T.tocsr()

        for _ in range(self.iterations):
            item_gram = self.item_factors.T @ self.item_factors
            for user in range(n_users):
                start, end = X.indptr[user:user + 2]
                items = X.indices[start:end]
                if len(items) == 0:
                    self.user_factors[user] = 0
                    continue
                factors = self.item_factors[items]
                target_values = T[user, items].toarray().ravel()
                left = item_gram + self.alpha * (factors.T @ factors) + self.reg * eye
                right = (1 + self.alpha) * (factors.T @ target_values)
                self.user_factors[user] = np.linalg.solve(left, right)

            user_gram = self.user_factors.T @ self.user_factors
            for item in range(n_items):
                start, end = X_t.indptr[item:item + 2]
                users = X_t.indices[start:end]
                if len(users) == 0:
                    self.item_factors[item] = 0
                    continue
                factors = self.user_factors[users]
                target_values = T_t[item, users].toarray().ravel()
                left = user_gram + self.alpha * (factors.T @ factors) + self.reg * eye
                right = (1 + self.alpha) * (factors.T @ target_values)
                self.item_factors[item] = np.linalg.solve(left, right)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
