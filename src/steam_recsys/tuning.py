"""Hyperparameter tuning and configuration comparison for BPR-MF on validation data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# Support direct script execution without needing PYTHONPATH set
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from .data import DataSplit, leave_two_out_split, load_steam_interactions
    from .evaluation import build_candidates, evaluate_model
    from .models.bpr import BPRRecommender
except ImportError:
    from steam_recsys.data import DataSplit, leave_two_out_split, load_steam_interactions
    from steam_recsys.evaluation import build_candidates, evaluate_model
    from steam_recsys.models.bpr import BPRRecommender

DEFAULT_BPR_CONFIGS: list[dict[str, Any]] = [
    {"factors": 32, "epochs": 15, "learning_rate": 0.04, "regularization": 0.002},
    {"factors": 48, "epochs": 25, "learning_rate": 0.04, "regularization": 0.002},
    {"factors": 64, "epochs": 30, "learning_rate": 0.03, "regularization": 0.002},
    {"factors": 64, "epochs": 40, "learning_rate": 0.02, "regularization": 0.001},
]


def compare_bpr_configurations(
    split: DataSplit,
    configs: Sequence[dict[str, Any]] | None = None,
    k: int = 10,
    n_negatives: int = 100,
    seed: int = 42,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare multiple BPR parameter configurations on validation data.

    Evaluation uses sampled negatives for ranking (Precision@K, Recall@K, NDCG@K, HitRate@K).
    """
    if configs is None:
        configs = DEFAULT_BPR_CONFIGS

    # Pre-generate validation candidates so every configuration is evaluated on identical sets
    candidates = build_candidates(split, split.validation_item, n_negatives=n_negatives, seed=seed)

    rows: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        if verbose:
            print(f"[{idx}/{len(configs)}] Evaluating BPR config: {cfg}")
        start_time = time.perf_counter()
        model_kwargs = dict(cfg)
        model = BPRRecommender(seed=seed, **model_kwargs)
        model.fit(split)
        fit_time = time.perf_counter() - start_time

        metrics = evaluate_model(
            model=model,
            split=split,
            target="validation",
            k=k,
            candidates=candidates,
            compute_coverage=False,
        )
        final_loss = float(model.losses[-1]) if model.losses else 0.0

        row = {
            "factors": model.factors,
            "epochs": model.epochs,
            "learning_rate": model.learning_rate,
            "regularization": model.regularization,
            f"Precision@{k}": metrics.get(f"Precision@{k}", 0.0),
            f"Recall@{k}": metrics.get(f"Recall@{k}", 0.0),
            f"NDCG@{k}": metrics.get(f"NDCG@{k}", 0.0),
            f"HitRate@{k}": metrics.get(f"HitRate@{k}", 0.0),
            "EvaluatedUsers": metrics.get("EvaluatedUsers", 0.0),
            "final_loss": final_loss,
            "fit_seconds": fit_time,
        }
        rows.append(row)
        if verbose:
            print(f"  -> NDCG@{k}: {row[f'NDCG@{k}']:.4f}, Recall@{k}: {row[f'Recall@{k}']:.4f}, Loss: {final_loss:.4f} ({fit_time:.2f}s)")

    results_df = pd.DataFrame(rows)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out, index=False)
        if verbose:
            print(f"Saved validation search results to: {out}")

    return results_df


def find_best_configuration(
    results: pd.DataFrame,
    metric: str = "NDCG@10",
) -> dict[str, Any]:
    """Find the best parameter configuration based on validation metric."""
    if metric not in results.columns:
        raise ValueError(f"Metric '{metric}' not in results columns: {list(results.columns)}")
    best_row = results.sort_values(by=metric, ascending=False).iloc[0]
    return {
        "factors": int(best_row["factors"]),
        "epochs": int(best_row["epochs"]),
        "learning_rate": float(best_row["learning_rate"]),
        "regularization": float(best_row["regularization"]),
        "best_metric": metric,
        "best_value": float(best_row[metric]),
    }


def tune_from_config(
    config_path: str | Path = "config/experiment.json",
    output_path: str | Path = "reports/bpr_validation_search.csv",
) -> pd.DataFrame:
    """Load dataset configuration, run BPR hyperparameter search on validation, and save results."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    raw_path = config.get("interactions_path", config.get("data_path", "data/raw/steam-200k.csv"))
    data_path = Path(raw_path)
    if not data_path.is_absolute():
        if not data_path.exists():
            data_path = config_path.parent.parent / data_path

    # Fallback if config data_path points to games metadata CSV instead of interaction history
    if "games.csv" in str(data_path) and not Path(data_path).name.startswith("steam-200k"):
        fallback = Path("data/raw/steam-200k.csv")
        if fallback.exists():
            data_path = fallback
        elif (config_path.parent.parent / "data/raw/steam-200k.csv").exists():
            data_path = config_path.parent.parent / "data/raw/steam-200k.csv"

    data = load_steam_interactions(
        str(data_path),
        min_user_interactions=config.get("min_user_interactions", 5),
        min_item_users=config.get("min_item_users", 3),
    )
    split = leave_two_out_split(data, seed=config.get("seed", 42))

    eval_cfg = config.get("evaluation", {})
    k = eval_cfg.get("k", 10)
    n_negatives = eval_cfg.get("negative_samples", 100)

    return compare_bpr_configurations(
        split=split,
        k=k,
        n_negatives=n_negatives,
        seed=config.get("seed", 42),
        output_path=output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BPR hyperparameters on validation split")
    parser.add_argument("--config", default="config/experiment.json", help="Path to config file")
    parser.add_argument("--output", default="reports/bpr_validation_search.csv", help="Output CSV path")
    args = parser.parse_args()

    results = tune_from_config(args.config, args.output)
    print("\nValidation Search Results:")
    print(results.to_string(index=False))
    best = find_best_configuration(results)
    print("\nBest Configuration:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
