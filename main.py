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
    parser.add_argument("command", choices=("download", "train"))
    parser.add_argument("--config", default="config/experiment.json")
    args = parser.parse_args()

    if args.command == "download":
        with Path(args.config).open(encoding="utf-8") as handle:
            download_data(json.load(handle)["data_path"])
    elif args.command == "train":
        from steam_recsys.experiment import run

        print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
