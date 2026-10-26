"""Display-only final panels and a targeted check of frame 0102's central mask."""

import argparse
from collections import deque
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import GlobalScaleCalibration
from src.contrast_calibration import LogAffineCalibration
from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256
from src.project import load_config, write_json


def central_gray_region(raw, seed_xy=(750, 400)):
    """Inspect the excluded connected component at the observed central patch.

    Connectivity uses the existing evaluation mask, not a presumption that the
    patch is zero or saturated. Coordinates are zero-based original pixels.
    """
    excluded = ~evaluation_mask(raw)
    x0, y0 = seed_xy
    if not excluded[y0, x0]:
        raise ValueError("Selected central-gray seed is not excluded by the fixed mask")
    visited = np.zeros_like(excluded)
    queue = deque([(y0, x0)])
    visited[y0, x0] = True
    ys, xs = [], []
    while queue:
        y, x = queue.popleft()
        ys.append(y)
        xs.append(x)
        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= ny < raw.shape[0] and 0 <= nx < raw.shape[1] and excluded[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    values, counts = np.unique(raw[np.array(ys), np.array(xs)], return_counts=True)
    return {"frame_id": "0102", "seed_xy_zero_based": list(seed_xy),
            "seed_raw_uint16": int(raw[y0, x0]),
            "component_connectivity": "four-connected excluded-mask component",
            "component_pixels": len(xs),
            "bbox_xy_inclusive": [min(xs), min(ys), max(xs), max(ys)],
            "raw_values_and_counts": {str(int(v)): int(n) for v, n in zip(values, counts)},
            "seed_decoded_mm": float(raw[y0, x0]) * 100 / 65535,
            "display_reason": "fixed evaluation mask excludes raw 0 and 65535; the display maps excluded pixels to gray"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/final_visualizations"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Final visualization directory already exists; select a new output to preserve it")
    root = Path("outputs")
    scale_path = root / "calibration" / "scale_fit.json"
    contrast_path = root / "contrast_calibration" / "log_affine_fit.json"
    protected = [scale_path, contrast_path, root / "results" / "summary.json",
                 root / "calibration" / "test_summary.json", root / "contrast_calibration" / "test_summary.json"]
    fingerprints = {str(path): file_sha256(path) for path in protected}
    scale_fit = json.loads(scale_path.read_text(encoding="utf-8"))
    contrast_fit = json.loads(contrast_path.read_text(encoding="utf-8"))
    scale = GlobalScaleCalibration(scale_fit["scale_factor"])
    contrast = LogAffineCalibration(contrast_fit["a"], contrast_fit["b"])
    dataset = C3VDDataset(load_config(args.config)["data_root"], "trans_t2_b")
    args.output.mkdir(parents=True)
    depth_cmap, error_cmap = plt.get_cmap("viridis").copy(), plt.get_cmap("magma").copy()
    gray = "#aaaaaa"
    depth_cmap.set_bad(gray)
    error_cmap.set_bad(gray)
    depth_norm = matplotlib.colors.Normalize(0, 100, clip=False)
    original_norm = matplotlib.colors.Normalize(0, 1300, clip=False)
    error_norm = matplotlib.colors.Normalize(0, 100, clip=False)
    reports = []
    gray_report = None
    for frame_id in ("0000", "0051", "0102"):
        index = next(i for i, pair in enumerate(dataset.pairs) if pair.frame_id == frame_id)
        sample = dataset[index]
        mask = evaluation_mask(sample.depth_raw)
        source = root / "results" / "predictions_mm" / f"{frame_id}_prediction_mm.npy"
        original = np.load(source, mmap_mode="r", allow_pickle=False)
        if original.shape != sample.depth_mm.shape or original.dtype != np.float32:
            raise ValueError("Cached prediction does not match the RGB/GT dimensions")
        global_depth, contrast_depth = scale.apply(original), contrast.apply(original)
        errors = [np.abs(prediction.astype(np.float64) - sample.depth_mm)
                  for prediction in (original, global_depth, contrast_depth)]
        fig, axes = plt.subplots(3, 3, figsize=(17.5, 14), constrained_layout=True)
        axes[0, 0].imshow(sample.rgb)
        axes[0, 0].set_title("RGB image", fontsize=13)
        arrays = [sample.depth_mm, original, global_depth, contrast_depth, *errors]
        counts = [int(np.count_nonzero(array[mask] > (1300 if i == 1 else 100)))
                  for i, array in enumerate(arrays)]
        panels = [
            (axes[0, 1], arrays[0], depth_cmap, depth_norm, "Ground truth | 0–100 mm", counts[0]),
            (axes[0, 2], arrays[1], depth_cmap, original_norm, "Original prediction | independent 0–1300 mm", counts[1]),
            (axes[1, 0], arrays[2], depth_cmap, depth_norm, "Global Scale | shared 0–100 mm", counts[2]),
            (axes[1, 1], arrays[3], depth_cmap, depth_norm, "Log-Affine | shared 0–100 mm", counts[3]),
            (axes[2, 0], arrays[4], error_cmap, error_norm, "Original absolute error | 0–100 mm", counts[4]),
            (axes[2, 1], arrays[5], error_cmap, error_norm, "Global Scale absolute error | 0–100 mm", counts[5]),
            (axes[2, 2], arrays[6], error_cmap, error_norm, "Log-Affine absolute error | 0–100 mm", counts[6]),
        ]
        for axis, array, cmap, norm, title, count in panels:
            artist = axis.imshow(np.ma.array(array, mask=~mask, copy=False), cmap=cmap,
                                 norm=norm, interpolation="nearest")
            subtitle = f"{count:,} valid pixels above display limit" if count else "Fixed physical scale; no per-image normalization"
            axis.set_title(f"{title}\n{subtitle}", fontsize=11)
            fig.colorbar(artist, ax=axis, fraction=.04, pad=.02, extend="max" if count else "neither")
        note = ("DISPLAY GUIDE\n\nGT + both calibrations: same 0–100 mm scale.\n"
                "Original prediction: independent 0–1300 mm scale.\n"
                "All three absolute error maps: same 0–100 mm scale.\n\n"
                "Gray: excluded by the same GT mask in every map.\n"
                "Mask: 0 < raw uint16 depth < 65535.\n\n"
                "Colorbar arrows show display saturation.\n"
                "Prediction arrays and evaluation values are not clipped.\n"
                "Original-error saturation reflects its large actual error.\n\n"
                "Cached predictions + frozen parameters only.\nNo inference, fitting or metric changes.")
        if frame_id == "0102":
            gray_report = central_gray_region(sample.depth_raw)
            note += "\n\n0102 central gray patch: raw = 65535 (100 mm).\nExcluded as the protocol's saturated boundary."
        axes[1, 2].text(.02, .98, note, transform=axes[1, 2].transAxes,
                        va="top", fontsize=10, linespacing=1.6)
        for axis in axes.flat:
            axis.set_axis_off()
        fig.suptitle(f"C3VD trans_t2_b / frame {frame_id} / final depth comparison\n"
                     "Shared GT/calibration ranges and masks; independent original-prediction scale is explicitly labeled",
                     fontsize=14)
        fig.savefig(args.output / f"{frame_id}_final_comparison.png", dpi=160)
        plt.close(fig)
        reports.append({"frame_id": frame_id, "rgb_file": sample.pair.rgb_path.name,
            "gt_file": sample.pair.depth_path.name, "raw_gt_dtype": str(sample.depth_raw.dtype),
            "depth_shape": list(sample.depth_mm.shape), "prediction_file": str(source),
            "source_array_read_only": not original.flags.writeable,
            "valid_pixels": int(mask.sum()),
            "above_display_limit_counts": dict(zip(("gt", "original", "global_scale", "log_affine",
                "original_error", "global_scale_error", "log_affine_error"), counts))})
        print(f"Saved {frame_id}_final_comparison.png", flush=True)
    for path in protected:
        if file_sha256(path) != fingerprints[str(path)]:
            raise RuntimeError("A protected fit/result file changed during plotting")
    write_json(args.output / "visualization.json", {"frames": reports,
        "shared_gt_calibrated_range_mm": [0, 100], "original_range_mm": [0, 1300],
        "shared_error_range_mm": [0, 100], "depth_colormap": "viridis", "error_colormap": "magma",
        "excluded_color": gray, "mask": "0 < depth_raw < 65535", "per_image_normalization": False,
        "raw_prediction_arrays_modified": False, "calibration_refitted": False,
        "model_inference_performed": False, "formal_results_modified": False,
        "previous_correct_calibration_visualizations_preserved": True,
        "0102_central_gray_check": gray_report})
    print(f"0102 central gray check: {gray_report}")


if __name__ == "__main__":
    main()
