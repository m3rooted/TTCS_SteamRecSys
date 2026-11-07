from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Đóng gói dữ liệu tương tác đã làm sạch cùng danh sách ID gốc để tra cứu.
@dataclass(frozen=True)
class InteractionData:
    frame: pd.DataFrame
    user_ids: np.ndarray
    item_titles: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_titles)


# Kết quả chia dữ liệu: ma trận train và một game validation/test cho mỗi người dùng.
@dataclass(frozen=True)
class DataSplit:
    n_users: int
    n_items: int
    train_users: np.ndarray
    train_items: np.ndarray
    train_matrix: csr_matrix
    user_train_items: tuple[np.ndarray, ...]
    validation_item: np.ndarray
    test_item: np.ndarray


def load_steam_interactions(path, min_user_interactions=5, min_item_users=3):
    """Đọc Steam-200k, gộp hành vi mua/chơi và lọc dữ liệu theo k-core."""
    if min_user_interactions < 3:
        raise ValueError("min_user_interactions must be at least 3 for leave-two-out")
    if min_item_users < 1:
        raise ValueError("min_item_users must be at least 1")
    # Steam-200k không có dòng tiêu đề; chỉ cần bốn cột đầu của dữ liệu.
    raw = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3],
                      names=["user_id", "game", "behavior", "value", "unused"])
    raw["behavior"] = raw["behavior"].astype(str).str.lower()
    unexpected = set(raw["behavior"].unique()) - {"purchase", "play"}
    if unexpected:
        raise ValueError(f"Unexpected behavior values: {sorted(unexpected)}")

    raw["playtime"] = np.where(raw["behavior"].eq("play"), raw["value"], 0.0)
    raw["purchased"] = raw["behavior"].eq("purchase").astype(np.int8)
    # Gộp các dòng trùng user-game: cộng thời lượng chơi và giữ cờ đã mua.
    interactions = raw.groupby(["user_id", "game"], as_index=False, sort=False).agg(
        playtime=("playtime", "sum"), purchased=("purchased", "max")
    )
    interactions = interactions[
        interactions["purchased"].eq(1) | interactions["playtime"].gt(0)
    ]
    # Lặp lọc vì xóa người dùng có thể làm game còn quá ít người chơi.
    while True:
        before = len(interactions)
        user_counts = interactions.groupby("user_id")["game"].transform("size")
        interactions = interactions[user_counts >= min_user_interactions]
        game_counts = interactions.groupby("game")["user_id"].transform("size")
        interactions = interactions[game_counts >= min_item_users]
        if len(interactions) == before:
            break
    interactions = interactions.reset_index(drop=True)
    if interactions.empty:
        raise ValueError("No interactions remain after k-core filtering")

    # Các mã đã được sắp xếp cung cấp chỉ số người dùng/trò chơi ổn định cho tất cả các mẫu.
    interactions["user_idx"], user_ids = pd.factorize(interactions["user_id"], sort=True)
    interactions["item_idx"], item_titles = pd.factorize(interactions["game"], sort=True)
    # Chuyển thời lượng chơi thành độ mạnh tương tác; log giảm ảnh hưởng ngoại lệ lớn.
    interactions["strength"] = (1 + np.log1p(interactions["playtime"])).astype(np.float32)
    return InteractionData(interactions, user_ids.to_numpy(), item_titles.to_numpy())


def leave_two_out_split(data, seed=42):
    """Giữ lại ngẫu nhiên hai game mỗi user cho validation và test; phần còn lại dùng train."""
    counts = data.frame.groupby("user_idx").size()
    if len(counts) != data.n_users or (counts < 3).any():
        raise ValueError("Each user must have at least 3 interactions for leave-two-out")
    rng = np.random.default_rng(seed)
    train_rows = []
    validation_item = np.full(data.n_users, -1, dtype=np.int32)
    test_item = np.full(data.n_users, -1, dtype=np.int32)

    items = data.frame["item_idx"].to_numpy()
    # Dùng seed cố định để kết quả chia tập có thể tái lập.
    for user_idx, rows in data.frame.groupby("user_idx", sort=True).indices.items():
        shuffled = rng.permutation(rows)
        test_item[user_idx], validation_item[user_idx] = items[shuffled[:2]]
        train_rows.extend(shuffled[2:])

    train = data.frame.iloc[train_rows]
    train_users = train["user_idx"].to_numpy(np.int32)
    train_items = train["item_idx"].to_numpy(np.int32)
    train_matrix = csr_matrix(
        (train["strength"].to_numpy(np.float32), (train_users, train_items)),
        shape=(data.n_users, data.n_items),
        dtype=np.float32,
    )

    # Game không xuất hiện trong train không thể được mô hình học, nên bỏ khỏi tập đánh giá.
    item_support = np.bincount(train_items, minlength=data.n_items)
    validation_item[item_support[validation_item] == 0] = -1
    test_item[item_support[test_item] == 0] = -1

    # Lưu danh sách game train từng user để mô hình lấy mẫu âm và loại game đã biết.
    user_train_items = tuple(train_matrix.getrow(u).indices for u in range(data.n_users))

    return DataSplit(
        n_users=data.n_users,
        n_items=data.n_items,
        train_users=train_users,
        train_items=train_items,
        train_matrix=train_matrix,
        user_train_items=user_train_items,
        validation_item=validation_item,
        test_item=test_item,
    )
