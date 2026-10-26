"""Read at most three development pairs; no full-dataset statistics."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from src.c3vd_dataset import C3VDDataset
from src.project import load_config, save_depth_preview, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--samples", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_inspection"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = C3VDDataset(config["data_root"], config["development_sequence"])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "limited data-loading check, not evaluation",
              "sequence": dataset.sequence, "directory": str(dataset.directory),
              "paired_filenames": len(dataset),
              "conversion": "depth_mm = uint16_code * 100 / 65535", "samples": []}
    for index in range(min(args.samples, len(dataset))):
        sample = dataset[index]
        prefix = args.output / sample.pair.frame_id
        Image.fromarray(sample.rgb).save(f"{prefix}_rgb.png")
        save_depth_preview(sample.depth_mm, f"{prefix}_gt_preview.png", 0, 100)
        info = {"frame_id": sample.pair.frame_id, "rgb_path": str(sample.pair.rgb_path),
                "depth_path": str(sample.pair.depth_path), "rgb_shape": list(sample.rgb.shape),
                "depth_shape": list(sample.depth_mm.shape), "raw_dtype": str(sample.depth_raw.dtype),
                "depth_dtype": str(sample.depth_mm.dtype),
                "raw_min": int(sample.depth_raw.min()), "raw_max": int(sample.depth_raw.max()),
                "depth_min_mm": float(sample.depth_mm.min()), "depth_max_mm": float(sample.depth_mm.max()),
                "zero_codes": int(np.count_nonzero(sample.depth_raw == 0)),
                "max_codes": int(np.count_nonzero(sample.depth_raw == 65535)),
                "storage_valid_pixels": int(sample.valid_mask.sum()),
                "mask_note": "Endpoints retained; final evaluation ROI not yet specified"}
        report["samples"].append(info)
    write_json(args.output / "inspection.json", report)
    print(f"Checked {len(report['samples'])} development pairs; report: {args.output / 'inspection.json'}")


if __name__ == "__main__":
    main()
