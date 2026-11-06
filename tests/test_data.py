"""Unit tests for data loading, preprocessing, and leave-two-out splitting."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steam_recsys.data import (
    DataSplit,
    InteractionData,
    leave_two_out_split,
    load_steam_interactions,
)


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_path = Path(self.temp_dir.name) / "steam_sample.csv"

    def _write_csv(self, content: str) -> Path:
        self.data_path.write_text(content, encoding="utf-8")
        return self.data_path

    def test_load_steam_interactions_basic(self):
        # 3 users, 3 games, each user interacts with all 3 games
        csv_text = "\n".join(
            [
                "1001,Dota 2,purchase,1,0",
                "1001,Dota 2,play,15.5,0",
                "1001,Portal,purchase,1,0",
                "1001,Terraria,purchase,1,0",
                "1002,Dota 2,purchase,1,0",
                "1002,Portal,purchase,1,0",
                "1002,Terraria,play,4.0,0",
                "1003,Dota 2,purchase,1,0",
                "1003,Portal,purchase,1,0",
                "1003,Terraria,purchase,1,0",
            ]
        )
        path = self._write_csv(csv_text)
        data = load_steam_interactions(path, min_user_interactions=3, min_item_users=1)

        self.assertIsInstance(data, InteractionData)
        self.assertEqual(data.n_users, 3)
        self.assertEqual(data.n_items, 3)
        self.assertEqual(len(data.user_ids), 3)
        self.assertEqual(len(data.item_titles), 3)

        # Check factorized sorted order
        np.testing.assert_array_equal(data.user_ids, [1001, 1002, 1003])
        np.testing.assert_array_equal(
            data.item_titles, ["Dota 2", "Portal", "Terraria"]
        )

        # Check aggregated purchase + play row
        dota_user_1001 = data.frame[
            (data.frame["user_id"] == 1001) & (data.frame["game"] == "Dota 2")
        ]
        self.assertEqual(len(dota_user_1001), 1)
        self.assertEqual(dota_user_1001.iloc[0]["purchased"], 1)
        self.assertAlmostEqual(dota_user_1001.iloc[0]["playtime"], 15.5)
        # Strength: 1 + log1p(15.5)
        expected_strength = 1.0 + np.log1p(15.5)
        self.assertAlmostEqual(
            dota_user_1001.iloc[0]["strength"], expected_strength, places=5
        )

    def test_k_core_filtering_prunes_sparse_users_and_items(self):
        # User 1 has 4 games, User 2 has 4 games, User 3 has 1 game (should be dropped)
        # Game "ObscureGame" only played by User 1 (should be dropped if min_item_users=2)
        lines = [
            "1,A,purchase,1,0",
            "1,B,purchase,1,0",
            "1,C,purchase,1,0",
            "1,D,purchase,1,0",
            "1,ObscureGame,purchase,1,0",
            "2,A,purchase,1,0",
            "2,B,purchase,1,0",
            "2,C,purchase,1,0",
            "2,D,purchase,1,0",
            "3,ObscureGame,purchase,1,0",
        ]
        path = self._write_csv("\n".join(lines))
        # With min_user=4 and min_item=2, ObscureGame has user 1 and user 3. But user 3 has 1 game (<4),
        # so user 3 dropped -> ObscureGame has 1 user (<2) -> ObscureGame dropped.
        # User 1 and User 2 still have games A, B, C, D (each >=4 games, each game >= 2 users).
        data = load_steam_interactions(path, min_user_interactions=4, min_item_users=2)
        self.assertEqual(data.n_users, 2)
        self.assertNotIn("ObscureGame", data.item_titles)
        self.assertListEqual(list(data.item_titles), ["A", "B", "C", "D"])

    def test_invalid_parameters_raise_value_errors(self):
        csv_text = "1,A,purchase,1,0\n1,B,purchase,1,0\n1,C,purchase,1,0\n"
        path = self._write_csv(csv_text)

        # min_user_interactions < 3
        with self.assertRaisesRegex(
            ValueError, "min_user_interactions must be at least 3"
        ):
            load_steam_interactions(path, min_user_interactions=2, min_item_users=1)

        # min_item_users < 1
        with self.assertRaisesRegex(ValueError, "min_item_users must be at least 1"):
            load_steam_interactions(path, min_user_interactions=3, min_item_users=0)

    def test_invalid_behavior_raises_value_error(self):
        csv_text = "1,A,wishlist,1,0\n1,B,purchase,1,0\n1,C,play,2,0\n"
        path = self._write_csv(csv_text)
        with self.assertRaisesRegex(ValueError, "Unexpected behavior values"):
            load_steam_interactions(path, min_user_interactions=3, min_item_users=1)

    def test_empty_dataset_after_filtering_raises_value_error(self):
        csv_text = "1,A,purchase,1,0\n2,B,purchase,1,0\n"
        path = self._write_csv(csv_text)
        with self.assertRaisesRegex(
            ValueError, "No interactions remain after k-core filtering"
        ):
            load_steam_interactions(path, min_user_interactions=3, min_item_users=1)


class TestDataSplitting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        # Create a valid dataset with 3 users and 5 games
        lines = []
        for u in [10, 20, 30]:
            for g in ["G1", "G2", "G3", "G4", "G5"]:
                lines.append(f"{u},{g},purchase,1,0")
        path = Path(self.temp_dir.name) / "test_split.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        self.data = load_steam_interactions(
            path, min_user_interactions=3, min_item_users=1
        )

    def test_leave_two_out_split_structure(self):
        split = leave_two_out_split(self.data, seed=42)
        self.assertIsInstance(split, DataSplit)
        self.assertEqual(split.n_users, 3)
        self.assertEqual(split.n_items, 5)

        # Each user must have exactly 1 validation item and 1 test item
        self.assertEqual(len(split.validation_item), 3)
        self.assertEqual(len(split.test_item), 3)

        for u in range(split.n_users):
            val_item = split.validation_item[u]
            test_item = split.test_item[u]
            train_items = set(split.user_train_items[u])

            self.assertGreaterEqual(val_item, 0)
            self.assertGreaterEqual(test_item, 0)
            self.assertNotEqual(val_item, test_item)

            # Neither validation nor test item can be in train_items
            self.assertNotIn(val_item, train_items)
            self.assertNotIn(test_item, train_items)

            # Total items for each user were 5; train must have 3 items
            self.assertEqual(len(train_items), 3)

        # Check train matrix properties
        self.assertEqual(split.train_matrix.shape, (3, 5))
        self.assertEqual(split.train_matrix.nnz, 3 * 3)

    def test_leave_two_out_seed_reproducibility(self):
        split1 = leave_two_out_split(self.data, seed=123)
        split2 = leave_two_out_split(self.data, seed=123)
        split3 = leave_two_out_split(self.data, seed=999)

        np.testing.assert_array_equal(split1.validation_item, split2.validation_item)
        np.testing.assert_array_equal(split1.test_item, split2.test_item)
        np.testing.assert_array_equal(split1.train_items, split2.train_items)

        # Different seed should give different shuffle
        self.assertFalse(
            np.array_equal(split1.validation_item, split3.validation_item)
            and np.array_equal(split1.test_item, split3.test_item)
        )

    def test_leave_two_out_user_with_fewer_than_3_interactions_fails(self):
        # Artificially alter data.frame so a user has fewer than 3 interactions
        modified_frame = self.data.frame.copy()
        # Drop interactions for user 0 down to 2
        user0_indices = modified_frame[modified_frame["user_idx"] == 0].index
        modified_frame.drop(user0_indices[:3], inplace=True)
        bad_data = InteractionData(
            modified_frame, self.data.user_ids, self.data.item_titles
        )

        with self.assertRaisesRegex(
            ValueError, "Each user must have at least 3 interactions"
        ):
            leave_two_out_split(bad_data)


if __name__ == "__main__":
    unittest.main()
