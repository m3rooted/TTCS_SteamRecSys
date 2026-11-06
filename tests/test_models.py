"""Unit tests for recommender models, evaluation metrics, recommendation, and tuning pipelines."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steam_recsys.data import leave_two_out_split, load_steam_interactions
from steam_recsys.evaluation import (
    build_candidates,
    coverage_at_k,
    evaluate_model,
    evaluate_models,
    hit_rate_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
)
from steam_recsys.experiment import save_bpr
from steam_recsys.models import (
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
)
from steam_recsys.recommend import (
    recommend_for_user,
    recommend_from_saved_bpr,
    recommend_unseen,
)
from steam_recsys.tuning import (
    compare_bpr_configurations,
    find_best_configuration,
)


class BaseModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        lines = []
        # 6 users all interact with G0..G4 (so G0..G4 have 6 users each)
        for u in [10, 20, 30, 40, 50, 60]:
            for g in ["G0", "G1", "G2", "G3", "G4"]:
                lines.append(f"{u},{g},purchase,1,0\n")
        # Users 10, 20, 30 also interact with G5, G6 (leaving G7, G8 unseen)
        for u in [10, 20, 30]:
            for g in ["G5", "G6"]:
                lines.append(f"{u},{g},purchase,1,0\n")
        # Users 40, 50, 60 also interact with G7, G8 (leaving G5, G6 unseen)
        for u in [40, 50, 60]:
            for g in ["G7", "G8"]:
                lines.append(f"{u},{g},purchase,1,0\n")

        path = Path(self.temp_dir.name) / "test_models.csv"
        path.write_text("".join(lines), encoding="utf-8")
        self.data = load_steam_interactions(
            path, min_user_interactions=3, min_item_users=1
        )
        self.split = leave_two_out_split(self.data, seed=42)


class TestRecommenderModels(BaseModelsTestCase):
    def test_popularity_recommender(self):
        pop = PopularityRecommender()
        with self.assertRaisesRegex(RuntimeError, "Model must be fitted"):
            pop.score(0, np.array([0, 1]))

        pop.fit(self.split)
        expected = np.log1p(
            np.bincount(self.split.train_items, minlength=self.split.n_items)
        )
        scores = pop.score(0, np.arange(self.split.n_items))
        np.testing.assert_allclose(scores, expected)

    def test_item_knn_recommender(self):
        knn = ItemKNNRecommender(neighbors=3)
        with self.assertRaisesRegex(RuntimeError, "Model must be fitted"):
            knn.score(0, np.array([0, 1]))

        knn.fit(self.split)
        self.assertEqual(knn.similarity.shape, (self.split.n_items, self.split.n_items))
        # Self similarity must be 0
        np.testing.assert_array_equal(
            knn.similarity.diagonal(), np.zeros(self.split.n_items)
        )

        scores = knn.score(0, np.arange(self.split.n_items))
        self.assertEqual(len(scores), self.split.n_items)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0))

    def test_bpr_recommender(self):
        bpr = BPRRecommender(factors=4, epochs=10, batch_size=4, seed=42)
        with self.assertRaisesRegex(RuntimeError, "Model must be fitted"):
            bpr.score(0, np.array([0, 1]))

        bpr.fit(self.split)
        self.assertEqual(len(bpr.losses), 10)
        # Loss should generally decrease or remain stable
        self.assertLess(bpr.losses[-1], bpr.losses[0])

        scores = bpr.score(0, np.arange(self.split.n_items))
        self.assertEqual(len(scores), self.split.n_items)
        expected = self.split.n_items
        self.assertEqual(len(scores), expected)


class TestEvaluationMetrics(unittest.TestCase):
    def test_ranking_metrics_exact_values(self):
        # 4 users: ranks 1, 2, 10, 11 for k=10
        ranks = [1, 2, 10, 11]
        k = 10

        # Ranks 1, 2, 10 are <= 10 (3 out of 4)
        expected_rec = 3.0 / 4.0
        expected_prec = (3.0 / 4.0) / 10.0
        expected_hr = 3.0 / 4.0
        expected_ndcg = (
            (1.0 / np.log2(1 + 1))
            + (1.0 / np.log2(2 + 1))
            + (1.0 / np.log2(10 + 1))
            + 0.0
        ) / 4.0

        self.assertAlmostEqual(recall_at_k(ranks, k), expected_rec)
        self.assertAlmostEqual(precision_at_k(ranks, k), expected_prec)
        self.assertAlmostEqual(hit_rate_at_k(ranks, k), expected_hr)
        self.assertAlmostEqual(ndcg_at_k(ranks, k), expected_ndcg)

    def test_empty_ranks_metric_behavior(self):
        self.assertEqual(recall_at_k([], 10), 0.0)
        self.assertEqual(precision_at_k([], 10), 0.0)
        self.assertEqual(hit_rate_at_k([], 10), 0.0)
        self.assertEqual(ndcg_at_k([], 10), 0.0)

    def test_coverage_metric(self):
        recs = [[0, 1, 2], [1, 2, 3], [3, 4]]
        # unique items: 0, 1, 2, 3, 4 (5 unique items out of 10 items)
        cov = coverage_at_k(recs, n_items=10)
        self.assertAlmostEqual(cov, 0.5)

    def test_novelty_metric(self):
        train_counts = np.array([10, 5, 1], dtype=np.float64)
        n_items = 3
        # Total tokens = 10 + 5 + 1 + 3 = 19
        # p_0 = 11 / 19, p_1 = 6 / 19, p_2 = 2 / 19
        # recs: user 0 gets [0, 2], user 1 gets [1, 2]
        recs = [[0, 2], [1, 2]]
        p = (train_counts + 1.0) / (np.sum(train_counts) + n_items)
        self_info = -np.log2(p)
        expected_novelty = float(
            np.mean([self_info[0], self_info[2], self_info[1], self_info[2]])
        )

        nov = novelty_at_k(recs, train_counts, n_items)
        self.assertAlmostEqual(nov, expected_novelty)


class TestEvaluationAndPipeline(BaseModelsTestCase):
    def test_candidate_generation(self):
        candidates = build_candidates(
            self.split, self.split.test_item, n_negatives=5, seed=42
        )
        self.assertEqual(len(candidates), self.split.n_users)
        for u in range(self.split.n_users):
            cand = candidates[u]
            # Index 0 must be positive test item
            self.assertEqual(cand[0], self.split.test_item[u])
            # Negative candidates must not be seen in train
            train_seen = set(self.split.user_train_items[u])
            for neg in cand[1:]:
                self.assertNotIn(neg, train_seen)

    def test_evaluate_model_and_compare(self):
        pop = PopularityRecommender().fit(self.split)
        metrics = evaluate_model(pop, self.split, target="test", k=5, n_negatives=5)
        for key in [
            "Precision@5",
            "Recall@5",
            "NDCG@5",
            "HitRate@5",
            "Coverage",
            "Novelty",
            "EvaluatedUsers",
        ]:
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

        df = evaluate_models([pop], self.split, target="test", k=5, n_negatives=5)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["model"], "Popularity")

    def test_recommend_unseen_and_for_user(self):
        pop = PopularityRecommender().fit(self.split)

        # recommend_unseen
        recs = recommend_unseen(pop, user_idx=0, split=self.split, top_k=3)
        self.assertLessEqual(len(recs), 3)
        seen_0 = set(self.split.user_train_items[0])
        for item_idx, score in recs:
            self.assertNotIn(item_idx, seen_0)

        # recommend_for_user with raw user ID (10)
        df_rec = recommend_for_user(
            pop, user=10, data=self.data, split=self.split, top_k=3
        )
        self.assertIsInstance(df_rec, pd.DataFrame)
        self.assertEqual(len(df_rec), min(3, self.split.n_items - len(seen_0)))
        self.assertListEqual(
            list(df_rec.columns), ["rank", "item_idx", "game", "score"]
        )

        # Invalid user raises ValueError
        with self.assertRaises(ValueError):
            recommend_for_user(pop, user=999999, data=self.data, split=self.split)

    def test_recommend_from_saved_bpr(self):
        bpr = BPRRecommender(factors=4, epochs=2, batch_size=4, seed=42).fit(self.split)
        model_path = Path(self.temp_dir.name) / "test_bpr.npz"
        save_bpr(bpr, self.data, model_path)

        df_rec = recommend_from_saved_bpr(model_path, user=10, top_k=3)
        self.assertEqual(len(df_rec), 3)
        self.assertListEqual(
            list(df_rec.columns), ["rank", "item_idx", "game", "score"]
        )

    def test_tuning_compare_bpr_configurations(self):
        test_configs = [
            {"factors": 2, "epochs": 1, "learning_rate": 0.05, "regularization": 0.001},
            {"factors": 4, "epochs": 1, "learning_rate": 0.02, "regularization": 0.002},
        ]
        out_csv = Path(self.temp_dir.name) / "tuning_results.csv"
        results_df = compare_bpr_configurations(
            split=self.split,
            configs=test_configs,
            k=3,
            n_negatives=3,
            output_path=out_csv,
            verbose=False,
        )
        self.assertEqual(len(results_df), 2)
        self.assertTrue(out_csv.exists())
        self.assertIn("NDCG@3", results_df.columns)
        self.assertIn("final_loss", results_df.columns)

        best_cfg = find_best_configuration(results_df, metric="NDCG@3")
        self.assertIn("factors", best_cfg)
        self.assertIn("best_value", best_cfg)


if __name__ == "__main__":
    unittest.main()
