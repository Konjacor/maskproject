from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable


def ensure_label_dirs(root: str | Path, labels: Iterable[str]) -> None:
    root_path = Path(root)
    for label in labels:
        (root_path / label).mkdir(parents=True, exist_ok=True)


def split_dataset(raw_dir: str | Path, train_dir: str | Path, val_dir: str | Path, val_split: float, seed: int) -> None:
    raw_path = Path(raw_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    if train_path.exists():
        shutil.rmtree(train_path)
    if val_path.exists():
        shutil.rmtree(val_path)
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    randomizer = random.Random(seed)
    for label_dir in sorted(path for path in raw_path.iterdir() if path.is_dir()):
        files = sorted(label_dir.glob("*.png")) + sorted(label_dir.glob("*.jpg"))
        randomizer.shuffle(files)
        val_count = int(len(files) * val_split)
        val_files = set(files[:val_count])
        for destination_root in (train_path, val_path):
            (destination_root / label_dir.name).mkdir(parents=True, exist_ok=True)
        for file_path in files:
            destination = val_path if file_path in val_files else train_path
            shutil.copy2(file_path, destination / label_dir.name / file_path.name)
