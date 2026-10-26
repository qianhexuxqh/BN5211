"""Native-resolution C3VD pairs. Depth is camera Z, encoded over 0–100 mm.

Official definition: https://durrlab.github.io/C3VD/
Zero and 65535 are retained; endpoint values alone do not establish invalidity.
The default mask is a storage/range check, not a final evaluation ROI policy.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import numpy as np
from PIL import Image

DEPTH_MAX_MM = 100.0
ENCODED_MAX = 65535
MaskPolicy = Callable[[np.ndarray], np.ndarray]


def decode_depth_mm(raw: np.ndarray) -> np.ndarray:
    """Decode uint16 without reducing bit depth or normalizing per image."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"Expected 2D uint16 depth, got {raw.shape}, {raw.dtype}")
    return raw.astype(np.float32) * np.float32(DEPTH_MAX_MM / ENCODED_MAX)


def read_depth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        raw = np.array(image)  # No convert('L'): preserve the TIFF's 16 bits.
    return raw, decode_depth_mm(raw)


def make_valid_mask(depth_mm: np.ndarray, policy: MaskPolicy | None = None) -> np.ndarray:
    """Retain both encoded endpoints; optionally intersect an explicit ROI policy.

    Ratio metrics require strictly positive selected GT/prediction. Callers must
    deliberately resolve zero GT pixels before evaluation, e.g. policy=lambda d:
    d > 0. No final evaluation masking policy is chosen by this module.
    """
    depth_mm = np.asarray(depth_mm)
    if depth_mm.ndim != 2:
        raise ValueError("Depth must be two-dimensional")
    mask = np.isfinite(depth_mm) & (depth_mm >= 0) & (depth_mm <= DEPTH_MAX_MM)
    if policy is not None:
        extra = np.asarray(policy(depth_mm))
        if extra.dtype != np.bool_ or extra.shape != depth_mm.shape:
            raise ValueError("Mask policy must return a boolean array with the depth shape")
        mask &= extra
    return mask


@dataclass(frozen=True)
class FramePair:
    frame_id: str
    rgb_path: Path
    depth_path: Path


@dataclass
class DepthSample:
    pair: FramePair
    rgb: np.ndarray
    depth_raw: np.ndarray
    depth_mm: np.ndarray
    valid_mask: np.ndarray


def locate_sequence(root: str | Path, sequence: str) -> Path:
    """Bounded lookup: supports project root and the downloaded nested folders."""
    if sequence not in {"trans_t1_a", "trans_t2_b"}:
        raise ValueError(f"Unsupported course sequence: {sequence}")
    root = Path(root).resolve()
    candidates = [root / sequence / sequence, root / sequence]
    if root.name == sequence:
        candidates.extend([root / sequence, root])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*_color.png")):
            return candidate
    raise FileNotFoundError(f"Cannot locate {sequence} under {root}; set data_root in config")


class C3VDDataset:
    def __init__(self, root: str | Path, sequence: str, mask_policy: MaskPolicy | None = None):
        self.sequence = sequence
        self.directory = locate_sequence(root, sequence)
        self.mask_policy = mask_policy
        rgb, depth = {}, {}
        pattern = re.compile(r"^(\d+)_(color\.png|depth\.tiff?)$", re.IGNORECASE)
        for path in self.directory.iterdir():
            match = pattern.fullmatch(path.name)
            if not match or not path.is_file():
                continue
            frame_id, kind = match.groups()
            mapping = rgb if kind.lower() == "color.png" else depth
            if frame_id in mapping:
                raise ValueError(f"Duplicate {kind} for frame {frame_id}")
            mapping[frame_id] = path
        if not rgb or rgb.keys() != depth.keys():
            missing_depth = sorted(rgb.keys() - depth.keys())[:5]
            missing_rgb = sorted(depth.keys() - rgb.keys())[:5]
            raise ValueError(f"Unpaired/empty sequence; missing depth={missing_depth}, RGB={missing_rgb}")
        self.pairs = [FramePair(k, rgb[k], depth[k]) for k in sorted(rgb, key=lambda k: (int(k), k))]

    def __len__(self) -> int:
        return len(self.pairs)

    def load_depth(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw, mm = read_depth(self.pairs[index].depth_path)
        return raw, mm, make_valid_mask(mm, self.mask_policy)

    def __getitem__(self, index: int) -> DepthSample:
        pair = self.pairs[index]
        with Image.open(pair.rgb_path) as image:
            rgb = np.array(image.convert("RGB"))
        raw, mm, mask = self.load_depth(index)
        if rgb.shape[:2] != mm.shape:
            raise ValueError(f"RGB/depth shape mismatch for {pair.frame_id}: {rgb.shape}, {mm.shape}")
        return DepthSample(pair, rgb, raw, mm, mask)
