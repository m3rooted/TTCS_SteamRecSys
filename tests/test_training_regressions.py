"""Regression checks for the implemented training code."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steam_recsys.data import load_steam_interactions, leave_two_out_split
from steam_recsys.models import BPRRecommender, ItemKNNRecommender, PopularityRecommender


class TrainingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "games.csv"
        self.path.write_text("".join(
            f"{user},{game},purchase,1,0\n"
            for user in (10, 20, 30) for game in ("A", "B", "C", "D")
        ), encoding="utf-8")
        self.data = load_steam_interactions(self.path, 3, 1)
        self.split = leave_two_out_split(self.data)

    def test_refit_resets_loss_history(self):
        model = BPRRecommender(epochs=2, factors=4)
        model.fit(self.split)
        first_scores = model.score(0, np.arange(4)).copy()
        model.fit(self.split)
        self.assertEqual(len(model.losses), 2)
        np.testing.assert_array_equal(first_scores, model.score(0, np.arange(4)))

    def test_saved_model_contains_ids_titles_and_training_config(self):
        model = BPRRecommender(epochs=1, factors=4).fit(self.split)
        output = Path(self.temp.name) / "model.npz"
        model.save(output, self.data)
        with np.load(output, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["user_ids"], [10, 20, 30])
            np.testing.assert_array_equal(saved["item_titles"], ["A", "B", "C", "D"])
            self.assertEqual(int(saved["factors"]), 4)
            self.assertEqual(int(saved["epochs"]), 1)
            scores = saved["item_factors"] @ saved["user_factors"][0] + saved["item_bias"]
            np.testing.assert_allclose(scores, model.score(0, np.arange(4)))

    def test_loader_rejects_threshold_with_no_training_history(self):
        for minimum in (1, 2):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, "at least 3"):
                load_steam_interactions(self.path, minimum, 1)

    def test_split_rejects_users_with_only_two_games(self):
        self.data.frame.drop(self.data.frame[self.data.frame.game.isin(["C", "D"])].index, inplace=True)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            leave_two_out_split(self.data)

    def test_split_and_basic_models(self):
        for user in range(3):
            self.assertEqual(len(self.split.user_train_items[user]), 2)
            self.assertNotIn(self.split.validation_item[user], self.split.user_train_items[user])
            self.assertNotIn(self.split.test_item[user], self.split.user_train_items[user])
        model = PopularityRecommender().fit(self.split)
        expected = np.log1p(np.bincount(self.split.train_items, minlength=4))
        np.testing.assert_allclose(model.score(0, np.arange(4)), expected)
        knn = ItemKNNRecommender(neighbors=1).fit(self.split)
        self.assertTrue(np.all(knn.similarity.diagonal() == 0))
        self.assertTrue(np.all(np.diff(knn.similarity.indptr) <= 1))
        self.assertTrue(np.isfinite(knn.score(0, np.arange(4))).all())


if __name__ == "__main__":
    unittest.main()
