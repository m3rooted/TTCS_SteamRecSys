from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .content_based import ContentBasedRecommender, load_game_catalog


def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = config_path.parent.parent / data_path
    config["data_path"] = data_path
    return config


def build_recommender(config_path: str | Path = "config/experiment.json") -> ContentBasedRecommender:
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
        "feature_fields": [field for field in catalog.frame.columns if field != "appid"],
    }
    Path("reports/dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep a portable sparse feature index; column/name order is stored alongside it.
    from scipy.sparse import save_npz
    save_npz("artifacts/game_features.npz", recommender.features, compressed=True)
    catalog.frame[["appid", "name"]].to_csv("artifacts/game_index.csv", index=False)
    return pd.DataFrame([{"model": "Content-based", **stats}])


def recommend(game: str, top_k: int = 10, config_path: str | Path = "config/experiment.json") -> pd.DataFrame:
    return build_recommender(config_path).recommend(game, top_k)
