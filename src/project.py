"""Small shared helpers for configuration and inspection previews."""

import json
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/default.json"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("data_root", "cache_dir"):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else PROJECT_ROOT / value
    return config


def save_depth_preview(depth_mm, path, minimum=None, maximum=None):
    """8-bit preview ONLY. Never use this image as a metric-depth array."""
    lower = float(np.min(depth_mm)) if minimum is None else minimum
    upper = float(np.max(depth_mm)) if maximum is None else maximum
    scaled = np.clip((depth_mm - lower) / max(upper - lower, 1e-12), 0, 1)
    Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).save(path)
    return {"preview_min_mm": lower, "preview_max_mm": upper}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
