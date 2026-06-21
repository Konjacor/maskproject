from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.config import load_config
from classifier.export import export_labels, export_saved_model, export_tflite, export_tflite_int8
from classifier.preprocess import normalize_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained classifier to TFLite")
    parser.add_argument("--config", default="classifier/config/default.json")
    parser.add_argument("--model", default="classifier/models/checkpoints/best.keras")
    return parser


def load_representative_samples(processed_train_dir: str, labels: list[str], input_size: int, grayscale: bool) -> np.ndarray:
    samples: list[np.ndarray] = []
    for label in labels:
        label_dir = Path(processed_train_dir) / label
        for path in sorted(list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg")))[:25]:
            image = tf.io.decode_png(tf.io.read_file(str(path)), channels=1 if grayscale else 3)
            image = tf.image.resize(image, [input_size, input_size]).numpy()
            if grayscale:
                image = image.squeeze(-1)
            samples.append(normalize_image(image, grayscale=grayscale))
    if not samples:
        raise RuntimeError("Representative dataset is empty")
    return np.asarray(samples, dtype=np.float32)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    labels = config.get("labels", [])
    input_size = int(config.get("input_size", 96))
    grayscale = bool(config.get("grayscale", True))
    export_config = config.get("export", {})
    dataset_config = config.get("dataset", {})

    model = tf.keras.models.load_model(args.model)
    representative = load_representative_samples(dataset_config["processed_train_dir"], labels, input_size, grayscale)
    export_saved_model(model, export_config["saved_model_dir"])
    export_tflite(model, export_config["tflite_path"])
    export_tflite_int8(model, export_config["tflite_int8_path"], representative)
    export_labels(labels, export_config["labels_path"])
    print("export complete")


if __name__ == "__main__":
    main()
