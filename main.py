"""Điểm khởi chạy duy nhất của chương trình gợi ý game Steam."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    # Khai báo giao diện dòng lệnh: train tạo chỉ mục, recommend tìm game tương tự.
    parser = argparse.ArgumentParser(description="Steam game recommendation")
    parser.add_argument("command", choices=("train", "recommend"))
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--game", help="Steam AppID or game name to use as the recommendation seed")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # Import sau khi thêm thư mục src vào đường dẫn để chạy được trực tiếp từ repo.
    from steam_recsys.experiment import recommend, run

    if args.command == "recommend":
        # Lệnh recommend bắt buộc có game mẫu; game có thể là AppID hoặc tên.
        if not args.game:
            parser.error("recommend requires --game with an AppID or game name")
        print(recommend(args.game, args.top_k, args.config).to_string(index=False))
    else:
        # Mặc định chạy pipeline tạo đặc trưng và thống kê catalog.
        print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
