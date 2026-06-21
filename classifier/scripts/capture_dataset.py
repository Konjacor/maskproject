from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.config import load_config
from classifier.dataset import ensure_label_dirs
from classifier.roi import build_tracker, extract_upper_face_crop, open_camera, read_packet, save_debug_frame


LABEL_KEYS = {
    ord("1"): "neutral",
    ord("2"): "sleepy",
    ord("3"): "surprised",
    ord("4"): "cheerful",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture labeled upper-face dataset samples")
    parser.add_argument("--config", default="classifier/config/default.json")
    parser.add_argument("--label", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    labels = config.get("labels", [])
    raw_dir = Path(config["dataset"]["raw_dir"])
    ensure_label_dirs(raw_dir, labels)

    capture = open_camera(config)
    tracker = build_tracker(config)
    frame_id = 0
    saved_count = 0
    active_label = args.label or config.get("capture", {}).get("label", labels[0])
    save_every = int(config.get("capture", {}).get("save_every", 2))
    snapshot_dir = Path(config.get("capture", {}).get("snapshot_dir", "classifier/runs/capture_debug"))

    try:
        while True:
            packet = read_packet(capture, frame_id, config)
            tracking = tracker.track(packet)
            face_roi = tracking.visible_regions["face"]
            crop = extract_upper_face_crop(packet, face_roi, config)

            preview = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview, f"label={active_label}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("classifier_crop", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key in LABEL_KEYS and LABEL_KEYS[key] in labels:
                active_label = LABEL_KEYS[key]
                print(f"active label -> {active_label}")
            elif key == ord("s"):
                output_path = raw_dir / active_label / f"sample_{frame_id:06d}.png"
                cv2.imwrite(str(output_path), crop)
                save_debug_frame(snapshot_dir / f"frame_{frame_id:06d}.jpg", packet, tracking, active_label)
                saved_count += 1
                print(f"saved {output_path}")
            elif key == ord("a"):
                for _ in range(save_every):
                    output_path = raw_dir / active_label / f"sample_{frame_id:06d}_{saved_count:03d}.png"
                    cv2.imwrite(str(output_path), crop)
                    saved_count += 1
                save_debug_frame(snapshot_dir / f"frame_{frame_id:06d}.jpg", packet, tracking, active_label)
                print(f"saved burst for {active_label}")
            frame_id += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
