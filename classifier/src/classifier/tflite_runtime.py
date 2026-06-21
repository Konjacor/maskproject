from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover
    from tensorflow.lite import Interpreter  # type: ignore


class TFLiteClassifier:
    def __init__(self, model_path: str | Path) -> None:
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def predict(self, image: np.ndarray) -> np.ndarray:
        tensor = image[None, ...].astype(np.float32)
        if self.input_details["dtype"] == np.int8:
            scale, zero_point = self.input_details["quantization"]
            tensor = np.round(tensor / scale + zero_point).astype(np.int8)
        self.interpreter.set_tensor(self.input_details["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])[0]
        if self.output_details["dtype"] == np.int8:
            scale, zero_point = self.output_details["quantization"]
            output = scale * (output.astype(np.float32) - zero_point)
        output = output.astype(np.float32)
        output = np.exp(output - np.max(output))
        return output / np.sum(output)
