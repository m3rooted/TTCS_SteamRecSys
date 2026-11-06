from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .content_based import ContentBasedRecommender, load_game_catalog
from .data import InteractionData
from .models.bpr import BPRRecommender


def save_bpr(model: BPRRecommender, data: InteractionData, path: str | Path) -> None:
    if (
        model.user_factors is None
        or model.item_factors is None
        or model.item_bias is None
    ):
        raise RuntimeError("Model must be fitted before saving")
    np.savez_compressed(
        path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        item_bias=model.item_bias,
        losses=np.asarray(model.losses),
        user_ids=data.user_ids,
        item_titles=data.item_titles.astype(str),
        factors=model.factors,
        epochs=model.epochs,
        batch_size=model.batch_size,
        learning_rate=model.learning_rate,
        regularization=model.regularization,
        seed=model.seed,
    )


def load_bpr(path: str | Path) -> tuple[BPRRecommender, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        model = BPRRecommender(
            factors=int(saved["factors"]),
            epochs=int(saved["epochs"]),
            batch_size=int(saved["batch_size"]),
            learning_rate=float(saved["learning_rate"]),
            regularization=float(saved["regularization"]),
            seed=int(saved["seed"]),
        )
        model.user_factors = saved["user_factors"]
        model.item_factors = saved["item_factors"]
        model.item_bias = saved["item_bias"]
        model.losses = list(saved["losses"])
        user_ids = saved["user_ids"]
        item_titles = saved["item_titles"]
    return model, user_ids, item_titles


def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = config_path.parent.parent / data_path
    config["data_path"] = data_path
    return config


def build_recommender(
    config_path: str | Path = "config/experiment.json",
) -> ContentBasedRecommender:
    config = _load_config(config_path)
    catalog = load_game_catalog(str(config["data_path"]))
    return ContentBasedRecommender.fit(catalog)


def run(config_path: str | Path = "config/experiment.json") -> pd.DataFrame:
    """Build the content feature index and report catalog coverage."""
    config = _load_config(config_path)
    catalog = load_game_catalog(str(config["data_path"]))
    recommender = ContentBasedRecommender.fit(catalog)
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    stats = {
        "games": catalog.n_games,
        "feature_count": recommender.features.shape[1],
        "feature_fields": [
            field for field in catalog.frame.columns if field != "appid"
        ],
    }
    Path("reports/dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep a portable sparse feature index; column/name order is stored alongside it.
    from scipy.sparse import save_npz

    save_npz("artifacts/game_features.npz", recommender.features, compressed=True)
    catalog.frame[["appid", "name"]].to_csv("artifacts/game_index.csv", index=False)
    return pd.DataFrame([{"model": "Content-based", **stats}])


def recommend(
    game: str, top_k: int = 10, config_path: str | Path = "config/experiment.json"
) -> pd.DataFrame:
    return build_recommender(config_path).recommend(game, top_k)
