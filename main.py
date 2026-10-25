"""Single entry point for the Steam recommendation project."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam game recommendation")
    parser.add_argument("command", choices=("train", "recommend"))
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--game", help="Steam AppID or game name to use as the recommendation seed")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    from steam_recsys.experiment import recommend, run

    if args.command == "recommend":
        if not args.game:
            parser.error("recommend requires --game with an AppID or game name")
        print(recommend(args.game, args.top_k, args.config).to_string(index=False))
    else:
        print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
