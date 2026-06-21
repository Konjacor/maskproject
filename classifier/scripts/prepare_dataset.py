from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.config import load_config
from classifier.dataset import ensure_label_dirs, split_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split raw dataset into train/val directories")
    parser.add_argument("--config", default="classifier/config/default.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    labels = config.get("labels", [])
    dataset_config = config.get("dataset", {})
    ensure_label_dirs(dataset_config["raw_dir"], labels)
    split_dataset(
        raw_dir=dataset_config["raw_dir"],
        train_dir=dataset_config["processed_train_dir"],
        val_dir=dataset_config["processed_val_dir"],
        val_split=float(dataset_config.get("val_split", 0.2)),
        seed=int(dataset_config.get("seed", 7)),
    )
    print("dataset split complete")


if __name__ == "__main__":
    main()
