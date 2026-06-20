from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Config parsing requires JSON-compatible YAML for now: {path} ({error})"
        ) from error
