from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("artifacts") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import leave_two_out_split, load_steam_interactions
from .models import BPRRecommender, ItemKNNRecommender, PopularityRecommender


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_metrics(results: pd.DataFrame, output: Path, k: int) -> None:
    """Shared reporting helper; connect after evaluation is implemented."""
    chart = results.set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), width_ratios=(1.7, 1))
    chart[[f"Recall@{k}", f"NDCG@{k}"]].plot(
        kind="bar", ax=axes[0], rot=0, width=0.72
    )
    chart[["Coverage"]].plot(
        kind="bar", ax=axes[1], rot=0, width=0.55, color="#d62728", legend=False
    )
    axes[0].set_title(f"Chất lượng xếp hạng top-{k}")
    axes[1].set_title("Độ phủ danh mục")
    for axis in axes:
        axis.set_xlabel("")
        axis.set_ylabel("Giá trị")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    figure.suptitle("So sánh các mô hình khuyến nghị Steam", fontsize=15)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def run(config_path: str | Path = "config/experiment.json") -> pd.DataFrame:
    config_path = Path(config_path)
    config = _load_config(config_path)
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
            model.save("artifacts/bpr_model.npz")
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
