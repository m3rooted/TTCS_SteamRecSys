"""Recommendation of unseen games for a specific user."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Support direct script execution without needing PYTHONPATH set
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from .data import DataSplit, InteractionData
    from .experiment import load_bpr
except ImportError:
    from steam_recsys.data import DataSplit, InteractionData
    from steam_recsys.experiment import load_bpr


def recommend_unseen(
    model: Any,
    user_idx: int,
    split: DataSplit,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Recommend top-K unseen game indices with scores for a user given DataSplit."""
    if not (0 <= user_idx < split.n_users):
        raise ValueError(f"user_idx {user_idx} out of range [0, {split.n_users})")

    seen = set(split.user_train_items[user_idx])
    unseen_items = np.array(
        [i for i in range(split.n_items) if i not in seen], dtype=np.int32
    )
    if len(unseen_items) == 0:
        return []

    scores = model.score(user_idx, unseen_items)
    k = min(top_k, len(unseen_items))
    top_order = np.argsort(-scores)[:k]
    return [(int(unseen_items[idx]), float(scores[idx])) for idx in top_order]


def recommend_for_user(
    model: Any,
    user: Any,
    data: InteractionData,
    split: DataSplit | None = None,
    top_k: int = 10,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Recommend top-K unseen games for a user specified by raw ID or integer index.

    Returns a DataFrame with columns: ['rank', 'item_idx', 'game', 'score'].
    """
    # Resolve user index and user ID
    user_str = str(user).strip()
    user_int = int(user_str) if user_str.lstrip("-").isdigit() else None

    if user_int is not None and user_int in data.user_ids:
        user_idx = int(np.where(data.user_ids == user_int)[0][0])
        user_id = user_int
    elif user in data.user_ids:
        user_idx = int(np.where(data.user_ids == user)[0][0])
        user_id = user
    elif user_int is not None and 0 <= user_int < data.n_users:
        user_idx = user_int
        user_id = data.user_ids[user_idx]
    else:
        raise ValueError(
            f"User '{user}' not found in user IDs or index out of range [0, {data.n_users})"
        )

    all_items = np.arange(data.n_items, dtype=np.int32)
    if exclude_seen:
        if split is not None:
            seen_items = set(split.user_train_items[user_idx])
        else:
            seen_items = set(data.frame[data.frame["user_idx"] == user_idx]["item_idx"])
        candidate_items = np.array(
            [i for i in all_items if i not in seen_items], dtype=np.int32
        )
    else:
        candidate_items = all_items

    if len(candidate_items) == 0:
        return pd.DataFrame(columns=["rank", "item_idx", "game", "score"])

    scores = model.score(user_idx, candidate_items)
    k = min(top_k, len(candidate_items))
    top_order = np.argsort(-scores)[:k]
    recommended_items = candidate_items[top_order]
    recommended_scores = scores[top_order]

    return pd.DataFrame(
        {
            "rank": np.arange(1, k + 1),
            "item_idx": recommended_items,
            "game": data.item_titles[recommended_items],
            "score": recommended_scores,
        }
    )


def recommend_from_saved_bpr(
    model_path: str | Path,
    user: Any,
    top_k: int = 10,
    interactions_path: str | Path | None = "data/raw/steam-200k.csv",
) -> pd.DataFrame:
    """Load a saved BPR model and recommend unseen games for a user."""
    model, user_ids, item_titles = load_bpr(model_path)

    # Normalize user identifier (accept numeric string, int ID, or integer index)
    user_str = str(user).strip()
    user_int = int(user_str) if user_str.lstrip("-").isdigit() else None

    if user_int is not None and user_int in user_ids:
        user_idx = int(np.where(user_ids == user_int)[0][0])
    elif user in user_ids:
        user_idx = int(np.where(user_ids == user)[0][0])
    elif user_int is not None and 0 <= user_int < len(user_ids):
        user_idx = user_int
    else:
        sample_users = ", ".join(str(u) for u in user_ids[:5])
        raise ValueError(
            f"User '{user}' not found in saved model ({len(user_ids)} users). Sample user IDs: {sample_users}"
        )

    # Filter out seen games if interaction history is available
    all_items = np.arange(len(item_titles), dtype=np.int32)
    seen_indices = set()
    if interactions_path:
        csv_file = Path(interactions_path)
        if not csv_file.is_absolute() and not csv_file.exists():
            csv_file = Path(__file__).resolve().parent.parent.parent / interactions_path
        if csv_file.exists():
            try:
                raw_user_id = user_ids[user_idx]
                raw_df = pd.read_csv(csv_file, header=None, usecols=[0, 1], names=["user_id", "game"])
                user_games = set(raw_df[raw_df["user_id"] == raw_user_id]["game"].astype(str))
                title_to_idx = {str(title): i for i, title in enumerate(item_titles)}
                seen_indices = {title_to_idx[g] for g in user_games if g in title_to_idx}
            except Exception:
                pass

    if seen_indices:
        candidate_items = np.array([i for i in all_items if i not in seen_indices], dtype=np.int32)
    else:
        candidate_items = all_items

    if len(candidate_items) == 0:
        return pd.DataFrame(columns=["rank", "item_idx", "game", "score"])

    scores = model.score(user_idx, candidate_items)
    k = min(top_k, len(candidate_items))
    top_order = np.argsort(-scores)[:k]
    recommended_items = candidate_items[top_order]
    recommended_scores = scores[top_order]

    return pd.DataFrame(
        {
            "rank": np.arange(1, k + 1),
            "item_idx": recommended_items,
            "game": item_titles[recommended_items],
            "score": recommended_scores,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend unseen games for a user")
    parser.add_argument("--user", required=True, help="User ID or user index")
    parser.add_argument(
        "--model-path",
        default="artifacts/bpr_model.npz",
        help="Path to saved BPR model",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations"
    )
    args = parser.parse_args()

    user_val: Any = int(args.user) if args.user.isdigit() else args.user
    df = recommend_from_saved_bpr(args.model_path, user_val, top_k=args.top_k)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
