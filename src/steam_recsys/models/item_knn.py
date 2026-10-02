from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags

from ..data import DataSplit


@dataclass
class ItemKNNRecommender:
    neighbors: int = 80
    name: str = "Item-kNN"
    user_item: csr_matrix | None = None
    similarity: csr_matrix | None = None

    def fit(self, split: DataSplit) -> "ItemKNNRecommender":
        binary = split.train_matrix.copy()
        binary.data = np.ones_like(binary.data)
        # Với ma trận nhị phân, chuẩn cột = căn số người chơi game đó.
        item_norms = np.sqrt(np.asarray(binary.sum(axis=0)).ravel())
        normalized = binary @ diags(1.0 / np.maximum(item_norms, 1))
        similarities = (normalized.T @ normalized).tocsr()
        similarities.setdiag(0)
        similarities.eliminate_zeros()
        # Keep only the strongest neighbors in each sparse row.
        for row in range(split.n_items):
            start, end = similarities.indptr[row:row + 2]
            values = similarities.data[start:end]
            if len(values) > self.neighbors:
                weakest = np.argpartition(values, -self.neighbors)[:-self.neighbors]
                values[weakest] = 0
        similarities.eliminate_zeros()
        self.similarity = similarities
        self.user_item = split.train_matrix
        return self

    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        if self.user_item is None or self.similarity is None:
            raise RuntimeError("Model must be fitted before scoring")
        scores = self.user_item.getrow(user_idx) @ self.similarity[:, item_indices]
        return np.asarray(scores.toarray()).ravel().astype(np.float32)
