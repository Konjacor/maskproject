from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.config import load_config
from classifier.preprocess import normalize_image
from classifier.roi import build_tracker, extract_upper_face_crop, open_camera, read_packet, save_debug_frame
from classifier.tflite_runtime import TFLiteClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live TFLite classifier inference")
    parser.add_argument("--config", default="classifier/config/default.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    inference_config = config.get("inference", {})
    labels = json.loads(Path(inference_config["labels_path"]).read_text(encoding="utf-8"))

    interpreter = TFLiteClassifier(inference_config["model_path"])
    capture = open_camera(config)
    tracker = build_tracker(config)
    frame_id = 0
    print_every = int(inference_config.get("print_every", 6))
    snapshot_every = int(inference_config.get("snapshot_every", 12))
    snapshot_dir = Path(inference_config.get("snapshot_dir", "classifier/runs/inference_debug"))

    try:
        while True:
            packet = read_packet(capture, frame_id, config)
            tracking = tracker.track(packet)
            crop = extract_upper_face_crop(packet, tracking.visible_regions["face"], config)
            tensor = normalize_image(crop, grayscale=config.get("grayscale", True)).astype(np.float32)
            probabilities = interpreter.predict(tensor)
            best_index = int(np.argmax(probabilities))
            best_label = labels[best_index]
            confidence = float(probabilities[best_index])

            preview = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview, f"{best_label} {confidence:.2f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("classifier_inference", preview)

            if frame_id % print_every == 0:
                summary = {label: float(probabilities[index]) for index, label in enumerate(labels)}
                print(f"FRAME {frame_id:04d} -> {summary}")
            if snapshot_every > 0 and frame_id % snapshot_every == 0:
                save_debug_frame(snapshot_dir / f"frame_{frame_id:06d}.jpg", packet, tracking, best_label)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            frame_id += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
