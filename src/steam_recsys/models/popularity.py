from dataclasses import dataclass

import numpy as np

from ..data import DataSplit


@dataclass
class PopularityRecommender:
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


