"""Khuyến nghị game Steam cho một người dùng cụ thể.

Hàm chính:
    recommend_for_user -- trả về top-K game chưa được người dùng tương tác,
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataSplit, InteractionData
from .models import Recommender


def recommend_for_user(
    model: Recommender,
    user_idx: int,
    data: InteractionData,
    split: DataSplit,
    k: int = 10,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Sinh top-K khuyến nghị game chưa xem cho một người dùng.

    Parameters
    ----------
    model:
        Mô hình đã huấn luyện (PopularityRecommender, ItemKNNRecommender
        hoặc BPRRecommender).
    user_idx:
        Chỉ số nội bộ của người dùng (0-based, tương ứng với
        ``data.user_ids[user_idx]``).
    data:
        Đối tượng InteractionData chứa thông tin ánh xạ user/item.
    split:
        Đối tượng DataSplit chứa danh sách item đã tương tác.
    k:
        Số lượng game được khuyến nghị.
    exclude_seen:
        Nếu ``True`` (mặc định), loại bỏ toàn bộ game đã xuất hiện
        trong tập train, validation và test trước khi xếp hạng.

    Returns
    -------
    pd.DataFrame
        Bảng ``k`` dòng với các cột:
        - ``rank``  : thứ hạng (1 = tốt nhất).
        - ``game``  : tên game.
        - ``score`` : điểm số do mô hình tính.
    """
    if not (0 <= user_idx < split.n_users):
        raise IndexError(f"user_idx={user_idx} nằm ngoài phạm vi [0, {split.n_users})")

    # Xây dựng tập item ứng viên (toàn bộ item trừ những item đã thấy)
    all_item_indices = np.arange(split.n_items, dtype=np.int32)
    if exclude_seen:
        seen: frozenset[int] = split.user_all_items[user_idx]
        candidate_mask = np.ones(split.n_items, dtype=bool)
        candidate_mask[np.fromiter(seen, dtype=np.int32, count=len(seen))] = False
        candidate_indices = all_item_indices[candidate_mask]
    else:
        candidate_indices = all_item_indices

    if len(candidate_indices) == 0:
        return pd.DataFrame(columns=["rank", "game", "score"])

    # Tính điểm cho các ứng viên
    scores = model.score(user_idx, candidate_indices)

    # Lấy top-K theo điểm giảm dần
    actual_k = min(k, len(candidate_indices))
    top_positions = np.argpartition(scores, -actual_k)[-actual_k:]
    top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]

    top_item_indices = candidate_indices[top_positions]
    top_scores = scores[top_positions]

    return pd.DataFrame(
        {
            "rank": np.arange(1, actual_k + 1, dtype=np.int32),
            "game": data.item_titles[top_item_indices],
            "score": top_scores.astype(np.float64),
        }
    )


def recommend_by_steam_id(
    model: Recommender,
    steam_id: int,
    data: InteractionData,
    split: DataSplit,
    k: int = 10,
) -> pd.DataFrame:
    """Wrapper:
    nhận Steam user ID thực tế thay vì chỉ số nội bộ.

    Parameters
    ----------
    steam_id:
        Steam user ID (giá trị trong cột ``user_id`` của dataset gốc).

    Raises
    ------
    KeyError
        Nếu ``steam_id`` không có trong tập dữ liệu.

    Returns
    -------
    pd.DataFrame
        Giống ``recommend_for_user``.
    """
    matches = np.where(data.user_ids == steam_id)[0]
    if len(matches) == 0:
        raise KeyError(
            f"steam_id={steam_id} không tồn tại trong tập dữ liệu. "
            f"Kiểm tra data.user_ids để xem danh sách user hợp lệ."
        )
    user_idx = int(matches[0])
    return recommend_for_user(model, user_idx, data, split, k=k)
