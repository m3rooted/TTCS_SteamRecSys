"""Candidate generation and recommendation ranking evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
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
    from .data import DataSplit
except ImportError:
    from steam_recsys.data import DataSplit



def build_candidates(
    split: DataSplit,
    target_items: np.ndarray,
    n_negatives: int = 100,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Build candidate items for each user consisting of 1 positive target item and n_negatives sampled negatives.

    The positive item is always placed at index 0 of the returned array.
    Users with target_items[u] < 0 are skipped.
    """
    rng = np.random.default_rng(seed)
    candidates: dict[int, np.ndarray] = {}
    all_items = np.arange(split.n_items, dtype=np.int32)

    for user_idx in range(split.n_users):
        target_item = int(target_items[user_idx])
        if target_item < 0:
            continue

        # Seen items include training items plus validation/test items
        seen = set(split.user_train_items[user_idx])
        val_item = int(split.validation_item[user_idx])
        test_item = int(split.test_item[user_idx])
        if val_item >= 0:
            seen.add(val_item)
        if test_item >= 0:
            seen.add(test_item)

        # Candidate negatives must not include positive target or seen items
        unseen_mask = np.ones(split.n_items, dtype=bool)
        if seen:
            unseen_mask[list(seen)] = False

        unseen_items = all_items[unseen_mask]
        if len(unseen_items) == 0:
            continue

        if len(unseen_items) > n_negatives:
            sampled_negatives = rng.choice(
                unseen_items, size=n_negatives, replace=False
            )
        else:
            sampled_negatives = unseen_items

        user_candidates = np.empty(len(sampled_negatives) + 1, dtype=np.int32)
        user_candidates[0] = target_item
        user_candidates[1:] = sampled_negatives
        candidates[user_idx] = user_candidates

    return candidates


def compute_positive_ranks(
    model: Any,
    candidates: dict[int, np.ndarray],
) -> dict[int, int]:
    """Compute 1-based rank of the positive candidate (index 0) for each user."""
    ranks: dict[int, int] = {}
    for user_idx, item_indices in candidates.items():
        scores = model.score(user_idx, item_indices)
        # Sort descending; index 0 was positive
        order = np.argsort(-scores, kind="stable")
        pos_rank = int(np.where(order == 0)[0][0]) + 1
        ranks[user_idx] = pos_rank
    return ranks


def recall_at_k(ranks: Sequence[int], k: int = 10) -> float:
    """Recall@K when exactly 1 target item exists per user."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))


def precision_at_k(ranks: Sequence[int], k: int = 10) -> float:
    """Precision@K when exactly 1 target item exists per user (Recall@K / K)."""
    if not ranks or k <= 0:
        return 0.0
    return float(recall_at_k(ranks, k) / k)


def hit_rate_at_k(ranks: Sequence[int], k: int = 10) -> float:
    """HitRate@K (equivalent to Recall@K with single ground-truth item)."""
    return recall_at_k(ranks, k)


def ndcg_at_k(ranks: Sequence[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K for single positive target item."""
    if not ranks:
        return 0.0
    # IDCG for 1 item at rank 1 is 1.0 / log2(1 + 1) = 1.0
    gains = [1.0 / np.log2(r + 1) if r <= k else 0.0 for r in ranks]
    return float(np.mean(gains))


def coverage_at_k(
    recommendations: Sequence[Sequence[int] | np.ndarray],
    n_items: int,
) -> float:
    """Fraction of catalog items recommended at least once in top-K lists."""
    if n_items <= 0 or not recommendations:
        return 0.0
    unique_items = set()
    for rec_list in recommendations:
        unique_items.update(int(item) for item in rec_list)
    return float(len(unique_items) / n_items)


def novelty_at_k(
    recommendations: Sequence[Sequence[int] | np.ndarray],
    train_item_counts: np.ndarray,
    n_items: int,
) -> float:
    """Mean self-information -log2(p_i) of recommended items across users.

    p_i = (count_i + 1) / (sum(counts) + n_items).
    """
    if not recommendations or n_items <= 0:
        return 0.0
    total_tokens = float(np.sum(train_item_counts) + n_items)
    p_i = (train_item_counts + 1.0) / total_tokens
    self_info = -np.log2(p_i)

    item_infos = []
    for rec_list in recommendations:
        for item in rec_list:
            item_infos.append(self_info[int(item)])

    if not item_infos:
        return 0.0
    return float(np.mean(item_infos))


def generate_catalog_top_k(
    model: Any,
    split: DataSplit,
    user_indices: Sequence[int],
    k: int = 10,
) -> list[np.ndarray]:
    """Generate top-K recommendations across the whole catalog for users, masking seen train items."""
    all_items = np.arange(split.n_items, dtype=np.int32)
    recommendations: list[np.ndarray] = []
    for user_idx in user_indices:
        scores = model.score(user_idx, all_items).copy()
        train_seen = split.user_train_items[user_idx]
        if len(train_seen) > 0:
            scores[train_seen] = -np.inf
        # Top-k items with highest scores
        if k < len(scores):
            top_k_indices = np.argpartition(-scores, k)[:k]
            # Sort within top k
            top_k_sorted = top_k_indices[np.argsort(-scores[top_k_indices])]
        else:
            top_k_sorted = np.argsort(-scores)
        recommendations.append(top_k_sorted)
    return recommendations


def evaluate_model(
    model: Any,
    split: DataSplit,
    target: str = "test",
    k: int = 10,
    n_negatives: int = 100,
    candidates: dict[int, np.ndarray] | None = None,
    seed: int = 42,
    compute_coverage: bool = True,
) -> dict[str, float]:
    """Evaluate a recommender model using candidate ranking and catalog diversity metrics."""
    if target == "validation":
        target_items = split.validation_item
    elif target == "test":
        target_items = split.test_item
    else:
        raise ValueError(
            f"Unknown evaluation target: {target}. Must be 'validation' or 'test'."
        )

    if candidates is None:
        candidates = build_candidates(
            split, target_items, n_negatives=n_negatives, seed=seed
        )

    ranks_dict = compute_positive_ranks(model, candidates)
    evaluated_users = list(ranks_dict.keys())
    ranks = list(ranks_dict.values())
    n_evaluated = len(ranks)

    prec = precision_at_k(ranks, k)
    rec = recall_at_k(ranks, k)
    ndcg = ndcg_at_k(ranks, k)
    hr = hit_rate_at_k(ranks, k)

    results: dict[str, float] = {
        f"Precision@{k}": prec,
        f"Recall@{k}": rec,
        f"NDCG@{k}": ndcg,
        f"HitRate@{k}": hr,
        "EvaluatedUsers": float(n_evaluated),
    }

    if compute_coverage:
        train_counts = np.bincount(split.train_items, minlength=split.n_items).astype(
            np.float64
        )
        catalog_recs = generate_catalog_top_k(model, split, evaluated_users, k=k)
        cov = coverage_at_k(catalog_recs, split.n_items)
        nov = novelty_at_k(catalog_recs, train_counts, split.n_items)
        results["Coverage"] = cov
        results["Novelty"] = nov

    return results


def evaluate_models(
    models: Sequence[Any],
    split: DataSplit,
    target: str = "test",
    k: int = 10,
    n_negatives: int = 100,
    seed: int = 42,
    compute_coverage: bool = True,
) -> pd.DataFrame:
    """Evaluate multiple models on the exact same candidate sets and report comparison table."""
    if target == "validation":
        target_items = split.validation_item
    elif target == "test":
        target_items = split.test_item
    else:
        raise ValueError(f"Unknown evaluation target: {target}")

    candidates = build_candidates(
        split, target_items, n_negatives=n_negatives, seed=seed
    )
    rows: list[dict[str, Any]] = []

    for model in models:
        metrics = evaluate_model(
            model=model,
            split=split,
            target=target,
            k=k,
            candidates=candidates,
            compute_coverage=compute_coverage,
        )
        row = {"model": getattr(model, "name", model.__class__.__name__), **metrics}
        rows.append(row)

    return pd.DataFrame(rows)
