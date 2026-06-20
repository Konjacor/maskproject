from __future__ import annotations

import argparse
from pathlib import Path

from face_mask.core.config import load_config
from face_mask.core.logging_setup import configure_logging
from face_mask.core.pipeline import RuntimePipeline


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FaceMask expressive runtime")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "default.yaml"),
        help="Path to YAML config file",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    logger = configure_logging(config)
    pipeline = RuntimePipeline(config=config, logger=logger)
    pipeline.run()


if __name__ == "__main__":
    main()
