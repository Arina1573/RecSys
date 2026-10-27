import numpy as np


class FM:
    def __init__(self, factors=16, lr=0.02, reg=0.002, epochs=4, seed=42):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

    def _features(self, user, item):
        start, end = self.features.indptr[item:item + 2]
        feature_ids = self.features.indices[start:end] + self.n_users + self.n_items
        values = self.features.data[start:end]
        ids = np.concatenate(([user, self.n_users + item], feature_ids))
        x = np.concatenate(([1.0, 1.0], values))
        return ids, x

    def _predict(self, ids, x):
        latent = self.V[ids]
        total = np.sum(latent * x[:, None], axis=0)
        interaction = 0.5 * np.sum(total ** 2 - np.sum(latent ** 2 * x[:, None] ** 2, axis=0))
        return float(np.dot(self.w[ids], x) + interaction)

    def _gradient(self, ids, x):
        latent = self.V[ids]
        total = np.sum(latent * x[:, None], axis=0)
        return x[:, None] * (total[None, :] - latent * x[:, None])

    def fit(self, train_matrix, item_features, seen_matrix=None):
        X = train_matrix.tocsr()
        self.features = item_features.tocsr().astype(float)
        self.n_users, self.n_items = X.shape
        if self.features.shape[0] != self.n_items:
            raise ValueError('Metadata rows must match item count')
        rng = np.random.default_rng(self.seed)
        n_features = self.n_users + self.n_items + self.features.shape[1]
        self.w = np.zeros(n_features)
        self.V = rng.normal(0, 0.01, (n_features, self.factors))
        known = X if seen_matrix is None else seen_matrix.tocsr()
        seen = [set(known.indices[known.indptr[u]:known.indptr[u + 1]]) for u in range(self.n_users)]
        rows, cols = X.nonzero()
        order = np.arange(len(rows))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for index in order:
                user, positive = int(rows[index]), int(cols[index])
                if len(seen[user]) == self.n_items:
                    continue
                negative = int(rng.integers(self.n_items))
                while negative in seen[user]:
                    negative = int(rng.integers(self.n_items))
                pos_ids, pos_x = self._features(user, positive)
                neg_ids, neg_x = self._features(user, negative)
                difference = self._predict(pos_ids, pos_x) - self._predict(neg_ids, neg_x)
                weight = 1.0 / (1.0 + np.exp(np.clip(difference, -35, 35)))
                pos_grad = self._gradient(pos_ids, pos_x)
                neg_grad = self._gradient(neg_ids, neg_x)
                active = np.union1d(pos_ids, neg_ids)
                pos_at = np.searchsorted(active, pos_ids)
                neg_at = np.searchsorted(active, neg_ids)
                grad_w = np.zeros(len(active))
                grad_v = np.zeros((len(active), self.factors))
                np.add.at(grad_w, pos_at, -weight * pos_x)
                np.add.at(grad_w, neg_at, weight * neg_x)
                np.add.at(grad_v, pos_at, -weight * pos_grad)
                np.add.at(grad_v, neg_at, weight * neg_grad)
                self.w[active] -= self.lr * (grad_w + self.reg * self.w[active])
                self.V[active] -= self.lr * (grad_v + self.reg * self.V[active])
        return self

    def score(self, users):
        scores = np.empty((len(users), self.n_items), dtype=np.float32)
        for row, user in enumerate(users):
            for item in range(self.n_items):
                ids, x = self._features(int(user), item)
                scores[row, item] = self._predict(ids, x)
        return scores
