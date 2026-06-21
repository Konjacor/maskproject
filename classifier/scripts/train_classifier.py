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
from classifier.model import build_augmentation, build_mobilenetv3_classifier
from classifier.preprocess import normalize_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Small upper-face classifier")
    parser.add_argument("--config", default="classifier/config/default.json")
    return parser


def load_split(directory: str, labels: list[str], input_size: int, grayscale: bool) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    targets: list[int] = []
    for index, label in enumerate(labels):
        label_dir = Path(directory) / label
        for path in sorted(list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))):
            color_flag = tf.io.decode_png(tf.io.read_file(str(path)), channels=1 if grayscale else 3)
            image = tf.image.resize(color_flag, [input_size, input_size]).numpy()
            if grayscale:
                image = image.squeeze(-1)
            images.append(normalize_image(image, grayscale=grayscale))
            targets.append(index)
    x = np.asarray(images, dtype=np.float32)
    y = tf.keras.utils.to_categorical(targets, num_classes=len(labels))
    return x, y


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    labels = config.get("labels", [])
    input_size = int(config.get("input_size", 96))
    grayscale = bool(config.get("grayscale", True))
    dataset_config = config.get("dataset", {})
    training_config = config.get("training", {})

    x_train, y_train = load_split(dataset_config["processed_train_dir"], labels, input_size, grayscale)
    x_val, y_val = load_split(dataset_config["processed_val_dir"], labels, input_size, grayscale)

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError("Training and validation sets must not be empty")

    model = build_mobilenetv3_classifier(input_size=input_size, class_count=len(labels), grayscale=grayscale)
    augmentation = build_augmentation(config)

    inputs = tf.keras.Input(shape=model.input_shape[1:])
    outputs = model(augmentation(inputs), training=True)
    train_model = tf.keras.Model(inputs, outputs)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(training_config.get("learning_rate", 5e-4))),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_dir = Path("classifier/models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    history = train_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=int(training_config.get("batch_size", 16)),
        epochs=int(training_config.get("epochs", 20)),
        callbacks=callbacks,
        verbose=1,
    )

    train_model.save("classifier/models/checkpoints/final.keras")
    print({"best_val_accuracy": max(history.history.get("val_accuracy", [0.0]))})


if __name__ == "__main__":
    main()
