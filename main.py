"""Single entry point for the Steam recommendation project."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam game recommendation")
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--config", default="config/experiment.json")
    args = parser.parse_args()

    from steam_recsys.experiment import run

    print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
