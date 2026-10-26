"""Fixed frames and shared 0–100 mm ranges for both calibrations and errors."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contrast_calibration"))
    args = parser.parse_args()
    fit = json.loads((args.output / "log_affine_fit.json").read_text(encoding="utf-8"))
    summary = json.loads((args.output / "test_summary.json").read_text(encoding="utf-8"))
    if summary["log_affine_fit_sha256"] != file_sha256(args.output / "log_affine_fit.json"):
        raise ValueError("Frozen fit changed")
    scale_fit = json.loads((args.previous / "scale_fit.json").read_text(encoding="utf-8"))
    scale, contrast = GlobalScaleCalibration(scale_fit["scale_factor"]), LogAffineCalibration(fit["a"], fit["b"])
    dataset = C3VDDataset(load_config(args.config)["data_root"], "trans_t2_b")
    indices = [0, 51, 102]
    output = args.output / "visualizations"
    output.mkdir(exist_ok=True)
    cmap, error_cmap = plt.get_cmap("viridis").copy(), plt.get_cmap("magma").copy()
    cmap.set_bad("#aaaaaa")
    error_cmap.set_bad("#aaaaaa")
    records = []
    for index in indices:
        sample = dataset[index]
        mask = evaluation_mask(sample.depth_raw)
        original = np.load(args.reference / "predictions_mm" / f"{sample.pair.frame_id}_prediction_mm.npy", allow_pickle=False)
        global_depth, contrast_depth = scale.apply(original), contrast.apply(original)
        errors = [np.abs(depth.astype(np.float64) - sample.depth_mm) for depth in (global_depth, contrast_depth)]
        arrays = [sample.depth_mm, global_depth, contrast_depth, *errors]
        counts = [int(np.count_nonzero(array[mask] > 100)) for array in arrays]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes[0, 0].imshow(sample.rgb)
        axes[0, 0].set_title("RGB")
        panels = [(axes[0, 1], arrays[0], cmap, "Ground truth Z-depth (mm)", counts[0]),
                  (axes[0, 2], arrays[1], cmap, "Global Scale Calibration (mm)", counts[1]),
                  (axes[1, 0], arrays[2], cmap, "Log-Affine Calibration (mm)", counts[2]),
                  (axes[1, 1], arrays[3], error_cmap, "Global scale absolute error (mm)", counts[3]),
                  (axes[1, 2], arrays[4], error_cmap, "Log-affine absolute error (mm)", counts[4])]
        for axis, array, panel_cmap, title, count in panels:
            artist = axis.imshow(np.ma.array(array, mask=~mask), cmap=panel_cmap,
                                 vmin=0, vmax=100, interpolation="nearest")
            axis.set_title(f"{title}\n{count:,} valid pixels above 100 mm display limit", fontsize=11)
            fig.colorbar(artist, ax=axis, fraction=.04, pad=.02, extend="max" if count else "neither")
        for axis in axes.flat:
            axis.set_axis_off()
        fig.suptitle(f"C3VD trans_t2_b / frame {sample.pair.frame_id} / fixed global calibrations\n"
                     "All depth ranges: 0–100 mm; both error ranges: 0–100 mm. Gray = excluded GT. Display saturation only; no numerical clipping.",
                     fontsize=12)
        fig.savefig(output / f"{sample.pair.frame_id}_contrast_comparison.png", dpi=150)
        plt.close(fig)
        records.append({"frame_id": sample.pair.frame_id, "valid_pixels": int(mask.sum()),
                        "global_scale_depth_above_100_mm": counts[1], "log_affine_depth_above_100_mm": counts[2],
                        "global_scale_error_above_100_mm": counts[3], "log_affine_error_above_100_mm": counts[4]})
    write_json(output / "visualization.json", {"fixed_frames": ["0000", "0051", "0102"],
        "selection_independent_of_errors": True, "shared_depth_range_mm": [0, 100],
        "shared_error_range_mm": [0, 100], "numerical_prediction_clipping": False,
        "per_image_normalization": False, "display_saturation_counts": records, "inference_performed": False})
    print(f"Saved three fixed-frame shared-range figures to {output}")


if __name__ == "__main__":
    main()
