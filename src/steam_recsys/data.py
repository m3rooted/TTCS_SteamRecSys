from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class InteractionData:
    frame: pd.DataFrame
    user_ids: np.ndarray
    item_titles: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_titles)


@dataclass(frozen=True)
class DataSplit:
    n_users: int
    n_items: int
    train_users: np.ndarray
    train_items: np.ndarray
    train_strengths: np.ndarray
    train_matrix: csr_matrix
    user_train_items: tuple[np.ndarray, ...]
    user_all_items: tuple[frozenset[int], ...]
    validation_item: np.ndarray
    test_item: np.ndarray

    def target_items(self, split: str) -> np.ndarray:
        if split == "validation":
            return self.validation_item
        if split == "test":
            return self.test_item
        raise ValueError("split must be 'validation' or 'test'")


def _iterative_k_core(
    frame: pd.DataFrame, min_user_interactions: int, min_item_users: int
) -> pd.DataFrame:
    filtered = frame
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user_id", sort=False)["game"].transform("size")
        filtered = filtered[user_counts >= min_user_interactions]
        item_counts = filtered.groupby("game", sort=False)["user_id"].transform("size")
        filtered = filtered[item_counts >= min_item_users]
        if len(filtered) == before:
            return filtered.reset_index(drop=True)


def load_steam_interactions(
    path: str | Path,
    min_user_interactions: int = 5,
    min_item_users: int = 3,
) -> InteractionData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Run `python main.py download` first."
        )

    raw = pd.read_csv(
        path,
        header=None,
        names=["user_id", "game", "behavior", "value", "unused"],
        usecols=[0, 1, 2, 3],
    )
    raw["behavior"] = raw["behavior"].astype(str).str.lower()
    valid_behaviors = {"purchase", "play"}
    unexpected = set(raw["behavior"].unique()) - valid_behaviors
    if unexpected:
        raise ValueError(f"Unexpected behavior values: {sorted(unexpected)}")

    raw["playtime"] = np.where(raw["behavior"].eq("play"), raw["value"], 0.0)
    raw["purchased"] = raw["behavior"].eq("purchase").astype(np.int8)
    interactions = (
        raw.groupby(["user_id", "game"], as_index=False, sort=False)
        .agg(playtime=("playtime", "sum"), purchased=("purchased", "max"))
    )
    interactions = interactions[
        interactions["purchased"].eq(1) | interactions["playtime"].gt(0)
    ]
    interactions = _iterative_k_core(
        interactions, min_user_interactions, min_item_users
    )
    if interactions.empty:
        raise ValueError("No interactions remain after k-core filtering")

    user_ids = np.sort(interactions["user_id"].unique())
    item_titles = np.sort(interactions["game"].astype(str).unique())
    user_lookup = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    item_lookup = pd.Series(
        np.arange(len(item_titles), dtype=np.int32), index=item_titles
    )
    interactions["user_idx"] = interactions["user_id"].map(user_lookup).astype(np.int32)
    interactions["item_idx"] = interactions["game"].map(item_lookup).astype(np.int32)
    interactions["strength"] = (1.0 + np.log1p(interactions["playtime"])).astype(
        np.float32
    )
    return InteractionData(interactions, user_ids, item_titles)


def leave_two_out_split(data: InteractionData, seed: int = 42) -> DataSplit:
    rng = np.random.default_rng(seed)
    train_rows: list[int] = []
    validation_item = np.full(data.n_users, -1, dtype=np.int32)
    test_item = np.full(data.n_users, -1, dtype=np.int32)

    grouped = data.frame.groupby("user_idx", sort=True).indices
    for user_idx in range(data.n_users):
        rows = np.asarray(grouped[user_idx], dtype=np.int64)
        shuffled = rng.permutation(rows)
        test_row, validation_row = shuffled[:2]
        test_item[user_idx] = int(data.frame.iloc[test_row]["item_idx"])
        validation_item[user_idx] = int(data.frame.iloc[validation_row]["item_idx"])
        train_rows.extend(shuffled[2:].tolist())

    train = data.frame.iloc[np.asarray(train_rows, dtype=np.int64)].copy()
    train_users = train["user_idx"].to_numpy(np.int32)
    train_items = train["item_idx"].to_numpy(np.int32)
    train_strengths = train["strength"].to_numpy(np.float32)
    train_matrix = csr_matrix(
        (train_strengths, (train_users, train_items)),
        shape=(data.n_users, data.n_items),
        dtype=np.float32,
    )

    item_support = np.bincount(train_items, minlength=data.n_items)
    validation_item[item_support[validation_item] == 0] = -1
    test_item[item_support[test_item] == 0] = -1

    user_train_items = tuple(
        train_matrix.indices[train_matrix.indptr[u] : train_matrix.indptr[u + 1]].copy()
        for u in range(data.n_users)
    )
    all_items: list[set[int]] = [set(items.tolist()) for items in user_train_items]
    for user_idx, item_idx in enumerate(validation_item):
        if item_idx >= 0:
            all_items[user_idx].add(int(item_idx))
    for user_idx, item_idx in enumerate(test_item):
        if item_idx >= 0:
            all_items[user_idx].add(int(item_idx))

    return DataSplit(
        n_users=data.n_users,
        n_items=data.n_items,
        train_users=train_users,
        train_items=train_items,
        train_strengths=train_strengths,
        train_matrix=train_matrix,
        user_train_items=user_train_items,
        user_all_items=tuple(frozenset(items) for items in all_items),
        validation_item=validation_item,
        test_item=test_item,
    )
