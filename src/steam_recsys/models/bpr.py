from dataclasses import dataclass, field

import numpy as np

from ..data import DataSplit


@dataclass
class BPRRecommender:
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
        self.losses.clear()
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

        for _ in range(self.epochs):
            order = rng.permutation(len(split.train_users))
            epoch_loss = 0.0
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

                user_vectors = self.user_factors[users]
                positive_vectors = self.item_factors[positive_items]
                negative_vectors = self.item_factors[negative_items]
                positive_bias = self.item_bias[positive_items]
                negative_bias = self.item_bias[negative_items]
                # Điểm game đã chơi trừ điểm game chưa chơi.
                margins = positive_bias - negative_bias + np.sum(
                    user_vectors * (positive_vectors - negative_vectors), axis=1
                )
                gradients = 1.0 / (1.0 + np.exp(np.clip(margins, -20, 20)))
                epoch_loss += float(np.logaddexp(0, -margins).sum())

                # Gradient của vector người dùng, hai game và bias.
                weight = gradients[:, None]
                reg = self.regularization
                user_gradient = weight * (positive_vectors - negative_vectors) - reg * user_vectors
                positive_gradient = weight * user_vectors - reg * positive_vectors
                negative_gradient = -weight * user_vectors - reg * negative_vectors

                # add.at cộng đủ cập nhật khi một ID xuất hiện nhiều lần trong batch.
                np.add.at(self.user_factors, users, learning_rate * user_gradient)
                np.add.at(self.item_factors, positive_items, learning_rate * positive_gradient)
                np.add.at(self.item_factors, negative_items, learning_rate * negative_gradient)
                np.add.at(self.item_bias, positive_items, learning_rate * (gradients - reg * positive_bias))
                np.add.at(self.item_bias, negative_items, learning_rate * (-gradients - reg * negative_bias))
            self.losses.append(epoch_loss / max(len(order), 1))
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
