from __future__ import annotations

from typing import Any

import tensorflow as tf


def build_mobilenetv3_classifier(input_size: int, class_count: int, grayscale: bool) -> tf.keras.Model:
    channels = 1 if grayscale else 3
    inputs = tf.keras.Input(shape=(input_size, input_size, channels))
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    if grayscale:
        x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    else:
        x = inputs
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(x)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_augmentation(config: dict[str, Any]) -> tf.keras.Sequential:
    augmentation_config = config.get("training", {}).get("augmentation", {})
    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.RandomRotation(augmentation_config.get("rotation", 8) / 180.0),
        tf.keras.layers.RandomTranslation(
            height_factor=augmentation_config.get("translation", 0.08),
            width_factor=augmentation_config.get("translation", 0.08),
        ),
        tf.keras.layers.RandomZoom(augmentation_config.get("zoom", 0.08)),
        tf.keras.layers.RandomBrightness(augmentation_config.get("brightness", 0.12)),
        tf.keras.layers.RandomContrast(augmentation_config.get("contrast", 0.12)),
    ]
    return tf.keras.Sequential(layers, name="augmentation")
