"""Single entry point for the Steam recommendation project."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def download_data(output: str) -> None:
    output = Path(output)
    if output.exists():
        print(f"Dataset already exists: {output}")
        return
    url = (
        "https://raw.githubusercontent.com/Michalos88/"
        "Game-Recommendation-System/master/steam-200k.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    try:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded file is unexpectedly small")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output} ({output.stat().st_size / 1_000_000:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam game recommendation")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, description in (
        ("download", "Download Steam-200k if missing"),
        ("train", "Train the three models and save BPR parameters"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", default="config/experiment.json")

    recommend_cmd = commands.add_parser(
        "recommend", help="Sinh top-K game chua xem cho mot nguoi dung"
    )
    recommend_cmd.add_argument("--config", default="config/experiment.json")
    recommend_cmd.add_argument(
        "--user",
        type=int,
        required=True,
        help="Steam user ID (gia tri thuc trong dataset)",
    )
    recommend_cmd.add_argument(
        "--model",
        choices=["bpr", "knn", "pop"],
        default="bpr",
        help="Mo hinh dung de xep hang (mac dinh: bpr)",
    )
    recommend_cmd.add_argument(
        "--k", type=int, default=10, help="So luong game khuyen nghi (mac dinh: 10)"
    )

    args = parser.parse_args()

    if args.command == "download":
        with Path(args.config).open(encoding="utf-8") as handle:
            download_data(json.load(handle)["data_path"])

    elif args.command == "train":
        from steam_recsys.experiment import run

        print(run(args.config).to_string(index=False))

    elif args.command == "recommend":
        import numpy as np

        from steam_recsys.data import leave_two_out_split, load_steam_interactions
        from steam_recsys.models import (
            BPRRecommender,
            ItemKNNRecommender,
            PopularityRecommender,
        )
        from steam_recsys.recommend import recommend_by_steam_id

        with Path(args.config).open(encoding="utf-8") as handle:
            config = json.load(handle)

        print("Dang tai du lieu...")
        data = load_steam_interactions(
            config["data_path"],
            config["min_user_interactions"],
            config["min_item_users"],
        )
        split = leave_two_out_split(data, config["seed"])

        if args.model == "bpr":
            artifact = Path("artifacts/bpr_model.npz")
            if not artifact.exists():
                print("Chua co artifacts/bpr_model.npz. Chay `train` truoc.")
                sys.exit(1)
            npz = np.load(artifact)
            model = BPRRecommender(seed=config["seed"], **config["bpr"])
            model.user_factors = npz["user_factors"]
            model.item_factors = npz["item_factors"]
            model.item_bias = npz["item_bias"]
        elif args.model == "knn":
            model = ItemKNNRecommender(**config["item_knn"])
            print("Dang huan luyen Item-kNN...")
            model.fit(split)
        else:
            model = PopularityRecommender()
            print("Dang huan luyen Popularity...")
            model.fit(split)

        result = recommend_by_steam_id(model, args.user, data, split, k=args.k)
        print(f"\nTop-{args.k} game khuyen nghi cho user {args.user} [{model.name}]:\n")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
