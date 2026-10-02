from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import InteractionData, load_steam_interactions, leave_two_out_split
from .models import BPRRecommender, ItemKNNRecommender, PopularityRecommender


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


def run(config_path: str | Path = "config/experiment.json") -> pd.DataFrame:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_steam_interactions(
        config["data_path"],
        config["min_user_interactions"],
        config["min_item_users"],
    )
    split = leave_two_out_split(data, config["seed"])
    models = [
        PopularityRecommender(),
        ItemKNNRecommender(**config["item_knn"]),
        BPRRecommender(seed=config["seed"], **config["bpr"]),
    ]

    rows: list[dict[str, float | str]] = []
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    for model in models:
        print(f"Training {model.name}...")
        started = time.perf_counter()
        model.fit(split)
        fit_seconds = time.perf_counter() - started
        rows.append({"model": model.name, "fit_seconds": fit_seconds})
        print(f"  Fit time: {fit_seconds:.2f}s")
        if isinstance(model, BPRRecommender):
            save_bpr(model, data, "artifacts/bpr_model.npz")
            pd.DataFrame(
                {"epoch": np.arange(1, len(model.losses) + 1), "bpr_loss": model.losses}
            ).to_csv("reports/bpr_training_loss.csv", index=False)

    results = pd.DataFrame(rows)
    stats = {
        "users": data.n_users,
        "items": data.n_items,
        "interactions": len(data.frame),
        "train_interactions": len(split.train_items),
        "density": len(data.frame) / (data.n_users * data.n_items),
    }
    Path("reports/dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
