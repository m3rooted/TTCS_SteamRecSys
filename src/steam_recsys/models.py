from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, diags

from .data import DataSplit


class Recommender(ABC):
    name: str

    @abstractmethod
    def fit(self, split: DataSplit) -> "Recommender":
        raise NotImplementedError

    @abstractmethod
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PopularityRecommender(Recommender):
    name: str = "Popularity"
    item_scores: np.ndarray | None = None

    def fit(self, split: DataSplit) -> "PopularityRecommender":
        counts = np.bincount(split.train_items, minlength=split.n_items).astype(np.float32)
        self.item_scores = np.log1p(counts)
        return self

    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        if self.item_scores is None:
            raise RuntimeError("Model must be fitted before scoring")
        return self.item_scores[item_indices]


@dataclass
class ItemKNNRecommender(Recommender):
    neighbors: int = 80
    name: str = "Item-kNN"
    user_item: csr_matrix | None = None
    similarity: csr_matrix | None = None

    def fit(self, split: DataSplit) -> "ItemKNNRecommender":
        binary = split.train_matrix.copy()
        binary.data = np.ones_like(binary.data)
        item_norms = np.sqrt(np.asarray(binary.power(2).sum(axis=0)).ravel())
        inverse_norms = np.divide(
            1.0,
            item_norms,
            out=np.zeros_like(item_norms, dtype=np.float32),
            where=item_norms > 0,
        )
        normalized = binary @ diags(inverse_norms, dtype=np.float32)
        similarities = (normalized.T @ normalized).tocsr()
        similarities.setdiag(0)
        similarities.eliminate_zeros()
        self.similarity = _keep_top_k(similarities, self.neighbors)
        self.user_item = split.train_matrix
        return self

    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        if self.user_item is None or self.similarity is None:
            raise RuntimeError("Model must be fitted before scoring")
        scores = self.user_item.getrow(user_idx) @ self.similarity[:, item_indices]
        return np.asarray(scores.toarray()).ravel().astype(np.float32)


def _keep_top_k(matrix: csr_matrix, k: int) -> csr_matrix:
    values: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_values = matrix.data[start:end]
        row_columns = matrix.indices[start:end]
        if len(row_values) > k:
            selected = np.argpartition(row_values, -k)[-k:]
            order = selected[np.argsort(row_values[selected])[::-1]]
            row_values = row_values[order]
            row_columns = row_columns[order]
        values.append(row_values.astype(np.float32, copy=False))
        columns.append(row_columns.astype(np.int32, copy=False))
        indptr.append(indptr[-1] + len(row_values))
    return csr_matrix(
        (
            np.concatenate(values) if values else np.array([], dtype=np.float32),
            np.concatenate(columns) if columns else np.array([], dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=matrix.shape,
    )


@dataclass
class BPRRecommender(Recommender):
    factors: int = 48
    epochs: int = 15
    batch_size: int = 2048
    learning_rate: float = 0.04
    regularization: float = 0.002
    seed: int = 42
    name: str = "BPR-MF"
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    losses: list[float] = field(default_factory=list)

    def fit(self, split: DataSplit) -> "BPRRecommender":
        rng = np.random.default_rng(self.seed)
        scale = 0.1 / np.sqrt(self.factors)
        self.user_factors = rng.normal(
            0, scale, size=(split.n_users, self.factors)
        ).astype(np.float32)
        self.item_factors = rng.normal(
            0, scale, size=(split.n_items, self.factors)
        ).astype(np.float32)
        self.item_bias = np.zeros(split.n_items, dtype=np.float32)
        positives = tuple(frozenset(items.tolist()) for items in split.user_train_items)
        learning_rate = self.learning_rate

        for epoch in range(self.epochs):
            order = rng.permutation(len(split.train_users))
            epoch_loss = 0.0
            observations = 0
            for start in range(0, len(order), self.batch_size):
                selected = order[start : start + self.batch_size]
                users = split.train_users[selected]
                positive_items = split.train_items[selected]
                negative_items = rng.integers(
                    0, split.n_items, size=len(selected), dtype=np.int32
                )
                for row, user_idx in enumerate(users):
                    while int(negative_items[row]) in positives[int(user_idx)]:
                        negative_items[row] = rng.integers(0, split.n_items)

                user_vectors = self.user_factors[users].copy()
                positive_vectors = self.item_factors[positive_items].copy()
                negative_vectors = self.item_factors[negative_items].copy()
                positive_bias = self.item_bias[positive_items].copy()
                negative_bias = self.item_bias[negative_items].copy()
                margins = positive_bias - negative_bias + np.einsum(
                    "ij,ij->i", user_vectors, positive_vectors - negative_vectors
                )
                gradients = 1.0 / (1.0 + np.exp(np.clip(margins, -20, 20)))
                epoch_loss += float(np.logaddexp(0, -margins).sum())
                observations += len(selected)

                np.add.at(
                    self.user_factors,
                    users,
                    learning_rate
                    * (
                        gradients[:, None] * (positive_vectors - negative_vectors)
                        - self.regularization * user_vectors
                    ),
                )
                np.add.at(
                    self.item_factors,
                    positive_items,
                    learning_rate
                    * (
                        gradients[:, None] * user_vectors
                        - self.regularization * positive_vectors
                    ),
                )
                np.add.at(
                    self.item_factors,
                    negative_items,
                    learning_rate
                    * (
                        -gradients[:, None] * user_vectors
                        - self.regularization * negative_vectors
                    ),
                )
                np.add.at(
                    self.item_bias,
                    positive_items,
                    learning_rate
                    * (gradients - self.regularization * positive_bias),
                )
                np.add.at(
                    self.item_bias,
                    negative_items,
                    learning_rate
                    * (-gradients - self.regularization * negative_bias),
                )
            self.losses.append(epoch_loss / max(observations, 1))
            learning_rate *= 0.96
        return self

    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.item_bias is None
        ):
            raise RuntimeError("Model must be fitted before scoring")
        return (
            self.item_factors[item_indices] @ self.user_factors[user_idx]
            + self.item_bias[item_indices]
        )

    def save(self, path: str) -> None:
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.item_bias is None
        ):
            raise RuntimeError("Model must be fitted before saving")
        np.savez_compressed(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            item_bias=self.item_bias,
            losses=np.asarray(self.losses),
        )
