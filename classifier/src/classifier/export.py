from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf


def export_saved_model(model: tf.keras.Model, saved_model_dir: str | Path) -> None:
    path = Path(saved_model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(path))


def export_tflite(model: tf.keras.Model, output_path: str | Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite_model)


def export_tflite_int8(model: tf.keras.Model, output_path: str | Path, representative_data: np.ndarray) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for sample in representative_data[: min(len(representative_data), 100)]:
            yield [sample[None, ...].astype("float32")]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tflite_model)


def export_labels(labels: Sequence[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(labels), ensure_ascii=False, indent=2), encoding="utf-8")
