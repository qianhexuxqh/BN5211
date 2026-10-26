"""Three fixed temporal samples from saved predictions; no model inference."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, read_frozen_artifacts
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--results", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()
    protocol, _ = read_frozen_artifacts(args.results)
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Only visualize completed evaluation")
    config = load_config(args.config)
    dataset = C3VDDataset(config["data_root"], protocol["test_sequence"])
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    samples = [dataset[index] for index in indices]
    predictions = [np.load(args.results / "predictions_mm" / f"{s.pair.frame_id}_prediction_mm.npy",
                           allow_pickle=False) for s in samples]
    # One display range across GT, prediction and all three figures. This does
    # not change any saved prediction or metric; no images selected by score.
    maximum = max(100., max(float(pred.max()) for pred in predictions))
    maximum = float(np.ceil(maximum / 100) * 100)
    output = args.results / "visualizations"
    output.mkdir(exist_ok=True)
    depth_cmap, error_cmap = plt.get_cmap("viridis").copy(), plt.get_cmap("magma").copy()
    depth_cmap.set_bad("#a0a0a0")
    error_cmap.set_bad("#a0a0a0")
    for sample, prediction in zip(samples, predictions):
        mask = evaluation_mask(sample.depth_raw)
        error = np.abs(prediction.astype(np.float64) - sample.depth_mm)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
        axes[0, 0].imshow(sample.rgb)
        axes[0, 0].set_title("RGB")
        gt = axes[0, 1].imshow(np.ma.array(sample.depth_mm, mask=~mask), cmap=depth_cmap,
                                vmin=0, vmax=maximum, interpolation="nearest")
        axes[0, 1].set_title("Ground truth Z-depth (mm)")
        pred = axes[1, 0].imshow(prediction, cmap=depth_cmap, vmin=0, vmax=maximum,
                                  interpolation="nearest")
        axes[1, 0].set_title("Raw zero-shot prediction (mm)")
        err = axes[1, 1].imshow(np.ma.array(error, mask=~mask), cmap=error_cmap,
                                 vmin=0, vmax=maximum, interpolation="nearest")
        axes[1, 1].set_title("Absolute error on evaluation pixels (mm)")
        for axis in axes.flat:
            axis.set_axis_off()
        for artist, axis in ((gt, axes[0, 1]), (pred, axes[1, 0]), (err, axes[1, 1])):
            fig.colorbar(artist, ax=axis, fraction=0.04, pad=0.02)
        fig.suptitle(f"C3VD trans_t2_b / frame {sample.pair.frame_id}\n"
                     f"Fixed mask: 0 < raw depth < 65535; gray = excluded GT; shared range 0–{maximum:.0f} mm",
                     fontsize=13)
        fig.savefig(output / f"{sample.pair.frame_id}_comparison.png", dpi=150)
        plt.close(fig)
    write_json(output / "visualization.json", {
        "selection": "fixed first, middle and last chronological test frames; independent of scores",
        "indices": indices, "frame_ids": [s.pair.frame_id for s in samples],
        "shared_depth_display_range_mm": [0, maximum],
        "shared_error_display_range_mm": [0, maximum],
        "prediction_modified": False, "inference_rerun": False,
    })
    print(f"Saved three fixed-sample comparison figures to {output}")


if __name__ == "__main__":
    main()
