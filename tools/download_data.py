"""Download a pinned snapshot of prepared data into the showcase."""
import argparse
import re
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "revision",
        nargs="?",
        default="5bf35b8304f2c43f34946ca070ac3f642c8bf80d",
        help="Full Hugging Face dataset commit SHA (defaults to the capstone release)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full 40-character commit SHA")

    snapshot_download(
        repo_id="glavvrach79/nyc-mobility-digital-twin",
        repo_type="dataset",
        revision=args.revision,
        local_dir=Path(__file__).resolve().parents[1],
        allow_patterns=["data/**", "models/**"],
        token=False,
    )


if __name__ == "__main__":
    main()
