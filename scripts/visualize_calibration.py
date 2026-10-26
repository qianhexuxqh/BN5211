"""Fixed temporal test samples, with honest separate original/calibrated scales."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import GlobalScaleCalibration
from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args()
    fit = json.loads((args.output / "scale_fit.json").read_text(encoding="utf-8"))
    summary = json.loads((args.output / "test_summary.json").read_text(encoding="utf-8"))
    if summary["scale_fit_sha256"] != file_sha256(args.output / "scale_fit.json"):
        raise ValueError("Frozen scale changed")
    calibration = GlobalScaleCalibration(fit["scale_factor"])
    dataset = C3VDDataset(load_config(args.config)["data_root"], "trans_t2_b")
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    samples = [dataset[index] for index in indices]
    originals = [np.load(args.reference / "predictions_mm" / f"{s.pair.frame_id}_prediction_mm.npy",
                         allow_pickle=False) for s in samples]
    calibrated = [calibration.apply(original) for original in originals]
    original_vmax = float(np.ceil(max(float(p.max()) for p in originals) / 100) * 100)
    # Fixed physical range for GT and calibrated predictions; shared across frames.
    depth_vmax = 100.
    max_error = max(float(np.abs(s.depth_mm[evaluation_mask(s.depth_raw)] - c[evaluation_mask(s.depth_raw)]).max())
                    for s, c in zip(samples, calibrated))
    error_vmax = max(100., float(np.ceil(max_error / 25) * 25))
    output = args.output / "visualizations"
    output.mkdir(exist_ok=True)
    depth_cmap, error_cmap = plt.get_cmap("viridis").copy(), plt.get_cmap("magma").copy()
    depth_cmap.set_bad("#aaaaaa")
    error_cmap.set_bad("#aaaaaa")
    records = []
    for sample, original, corrected in zip(samples, originals, calibrated):
        mask = evaluation_mask(sample.depth_raw)
        error = np.abs(corrected.astype(np.float64) - sample.depth_mm)
        out_of_display = int(np.count_nonzero(corrected[mask] > depth_vmax))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), constrained_layout=True)
        axes[0, 0].imshow(sample.rgb)
        axes[0, 0].set_title("RGB")
        display = [
            (axes[0, 1], np.ma.array(sample.depth_mm, mask=~mask), depth_cmap, depth_vmax, "GT Z-depth (0–100 mm)"),
            (axes[0, 2], np.ma.array(original, mask=~mask), depth_cmap, original_vmax,
             f"Original prediction (0–{original_vmax:.0f} mm; separate scale)"),
            (axes[1, 0], np.ma.array(corrected, mask=~mask), depth_cmap, depth_vmax,
             f"Calibrated prediction (0–100 mm)\n{out_of_display} valid pixels above display limit"),
            (axes[1, 1], np.ma.array(error, mask=~mask), error_cmap, error_vmax,
             "Calibrated absolute error (mm)"),
        ]
        for axis, array, cmap, vmax, title in display:
            artist = axis.imshow(array, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            axis.set_title(title, fontsize=11)
            fig.colorbar(artist, ax=axis, fraction=.04, pad=.02,
                         extend="max" if axis is axes[1, 0] and out_of_display else "neither")
        axes[1, 2].text(.02, .95,
            f"One development-only factor\ns = {calibration.scale_factor:.12g}\n\n"
            "GT and calibrated depth: same 0–100 mm scale.\n"
            f"Original prediction: separate 0–{original_vmax:.0f} mm scale.\n"
            f"Error: shared 0–{error_vmax:.0f} mm scale across figures.\n\n"
            "Gray = excluded GT pixels.\nDisplay saturation does not change arrays or scores.\n"
            "Fixed first / middle / last frames; no score selection.",
            transform=axes[1, 2].transAxes, va="top", fontsize=10, linespacing=1.6)
        for axis in axes.flat:
            axis.set_axis_off()
        fig.suptitle(f"C3VD trans_t2_b / frame {sample.pair.frame_id} / global development scale calibration", fontsize=14)
        fig.savefig(output / f"{sample.pair.frame_id}_calibration.png", dpi=150)
        plt.close(fig)
        records.append({"frame_id": sample.pair.frame_id, "valid_pixels": int(mask.sum()),
                        "calibrated_valid_pixels_above_100_mm_display": out_of_display})
    write_json(output / "visualization.json", {"selection": "fixed first, middle, last test frames, independent of errors",
        "indices": indices, "frames": records, "gt_and_calibrated_range_mm": [0, depth_vmax],
        "original_range_mm": [0, original_vmax], "calibrated_error_range_mm": [0, error_vmax],
        "arrays_or_metrics_clipped": False, "inference_rerun": False})
    print(f"Saved three fixed-frame calibration figures to {output}")


if __name__ == "__main__":
    main()
