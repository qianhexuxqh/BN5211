"""Single cached-prediction pass for depth bins, delta transitions, and error tails."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import numpy as np

from src.baseline import ConstantMedianDepth
from src.calibration import GlobalScaleCalibration
from src.contrast_calibration import LogAffineCalibration
from src.c3vd_dataset import C3VDDataset
from src.error_analysis import DEPTH_EDGES_MM, DEPTH_LABELS, depth_bin_indices, delta_transitions, tail_statistics
from src.evaluation import evaluation_mask, file_sha256
from src.metrics import DepthMetricAccumulator
from src.project import load_config, write_json

METHODS = ("constant_median", "original_depth_anything_v2", "global_scale", "log_affine")
METRICS = ("abs_rel", "rmse_mm", "delta_1_25")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe_frames(rows):
    descriptive, cases = [], []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        ranked = sorted(selected, key=lambda r: (r["abs_rel"], r["rmse_mm"], int(r["frame_id"])))
        for label, row in (("best_by_absrel", ranked[0]), ("worst_by_absrel", ranked[-1])):
            cases.append({"selection": label, **row})
        for metric in METRICS:
            values = np.array([r[metric] for r in selected], dtype=np.float64)
            quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
            descriptive.append({"method": method, "metric": metric, "frames": len(values),
                "mean": float(values.mean()), "std_population": float(values.std()),
                **dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), map(float, quantiles)))})
    return descriptive, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/error_analysis"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Analysis output already exists; refusing to overwrite")
    root = Path("outputs")
    sources = [root / "results" / "protocol.json", root / "results" / "baseline_fit.json",
               root / "calibration" / "scale_fit.json", root / "contrast_calibration" / "log_affine_fit.json",
               root / "contrast_calibration" / "test_summary.json", root / "contrast_calibration" / "test_per_frame.csv"]
    fingerprints = {str(path): file_sha256(path) for path in sources}
    protocol, baseline_fit, scale_fit, log_fit, existing = [json.loads(p.read_text(encoding="utf-8")) for p in sources[:5]]
    if protocol["protocol_id"] != "c3vd-interior-v1":
        raise ValueError("Unexpected frozen evaluation protocol")
    config = load_config(args.config)
    dataset = C3VDDataset(config["data_root"], "trans_t2_b")
    if [p.frame_id for p in dataset.pairs] != existing["frame_ids"]:
        raise ValueError("Test frame set differs from the existing evaluation")
    baseline = ConstantMedianDepth(baseline_fit["median_mm"])
    scale, contrast = GlobalScaleCalibration(scale_fit["scale_factor"]), LogAffineCalibration(log_fit["a"], log_fit["b"])
    args.output.mkdir(parents=True)
    rules = {"analysis_type": "descriptive analysis of one continuous test video; not independent clinical cases",
        "sequence": "trans_t2_b", "frames": 103, "protocol_id": protocol["protocol_id"],
        "depth_bin_edges_mm": DEPTH_EDGES_MM.tolist(), "bin_convention": "left-closed, right-open; raw 0 and 65535 excluded by existing mask",
        "aggregation": "pooled valid pixels, never simple frame-metric means",
        "case_ranking": "ascending per-frame AbsRel, then RMSE, then integer frame ID; first/last per method",
        "error_tail_definition": "absolute depth error > 50 mm and > 100 mm; strict thresholds fixed before analysis",
        "delta_transition_definition": "pixelwise max(pred/GT, GT/pred) < 1.25, strict; global-only pass is a loss, log-only pass is a gain",
        "no_fitting_or_inference": True, "formal_results_modified": False}
    write_json(args.output / "analysis_rules.json", rules)
    accumulators = [[DepthMetricAccumulator() for _ in METHODS] for _ in DEPTH_LABELS]
    counts = np.zeros(5, dtype=np.int64)
    diagnostic = {(j, m): {"gt_sum": 0., "pred_sum": 0., "signed_error_sum": 0., "underestimate_pixels": 0}
                  for j in range(5) for m in METHODS}
    transition_counts = {key: np.zeros(5, dtype=np.int64) for key in ("both_pass", "global_only_pass", "log_only_pass", "both_fail")}
    tails = {method: {threshold: {"pixels": 0, "squared_error_sum_mm2": 0.} for threshold in (50., 100.)} for method in METHODS}
    max_errors = {method: 0. for method in METHODS}
    bin_tails = {(j, method, threshold): {"pixels": 0, "squared_error_sum_mm2": 0.}
                 for j in range(5) for method in METHODS for threshold in (50., 100.)}
    for index, pair in enumerate(dataset.pairs):
        raw, gt, _ = dataset.load_depth(index)
        mask = evaluation_mask(raw)
        valid_gt = gt[mask].astype(np.float64)
        bin_index = depth_bin_indices(valid_gt)
        selectors = [bin_index == j for j in range(5)]
        frame_counts = np.bincount(bin_index, minlength=5)
        counts += frame_counts
        original = np.load(root / "results" / "predictions_mm" / f"{pair.frame_id}_prediction_mm.npy", mmap_mode="r", allow_pickle=False)
        predictions = {"constant_median": baseline.predict(gt.shape), "original_depth_anything_v2": original,
                       "global_scale": scale.apply(original), "log_affine": contrast.apply(original)}
        valid_predictions = {method: np.asarray(predictions[method][mask], dtype=np.float64) for method in METHODS}
        transitions = delta_transitions(valid_gt, valid_predictions["global_scale"], valid_predictions["log_affine"])
        for key, flags in transitions.items():
            transition_counts[key] += np.bincount(bin_index[flags], minlength=5)
        for method_index, method in enumerate(METHODS):
            pred = valid_predictions[method]
            max_errors[method] = max(max_errors[method], float(np.abs(pred - valid_gt).max()))
            for threshold, values in tail_statistics(valid_gt, pred).items():
                for key, value in values.items():
                    tails[method][threshold][key] += value
            for j, selector in enumerate(selectors):
                if not frame_counts[j]:
                    continue
                g, p = valid_gt[selector], pred[selector]
                accumulators[j][method_index].update(g.reshape(1, -1), p.reshape(1, -1), np.ones((1, g.size), dtype=bool))
                d = diagnostic[j, method]
                d["gt_sum"] += float(g.sum())
                d["pred_sum"] += float(p.sum())
                d["signed_error_sum"] += float((p - g).sum())
                d["underestimate_pixels"] += int(np.count_nonzero(p < g))
                for threshold, values in tail_statistics(g, p).items():
                    for key, value in values.items():
                        bin_tails[j, method, threshold][key] += value
        if index == 0 or (index+1) % 20 == 0 or index == len(dataset)-1:
            print(f"Analyzed cached test frame {index+1}/{len(dataset)}", flush=True)
    total = int(counts.sum())
    rows, tail_rows, bin_tail_rows = [], [], []
    for j, label in enumerate(DEPTH_LABELS):
        for mi, method in enumerate(METHODS):
            result = accumulators[j][mi].result()
            d = diagnostic[j, method]
            rows.append({"depth_bin_mm": label, "lower_mm": float(DEPTH_EDGES_MM[j]),
                "upper_mm_exclusive": float(DEPTH_EDGES_MM[j+1]), "method": method,
                "valid_pixel_fraction": int(counts[j])/total, **result,
                "mean_gt_mm": d["gt_sum"]/counts[j], "mean_prediction_mm": d["pred_sum"]/counts[j],
                "mean_signed_error_mm": d["signed_error_sum"]/counts[j],
                "underestimate_fraction": d["underestimate_pixels"]/counts[j]})
            for threshold in (50., 100.):
                t = bin_tails[j, method, threshold]
                bin_tail_rows.append({"depth_bin_mm": label, "method": method, "absolute_error_threshold_mm": threshold,
                    **t, "fraction_of_bin_pixels": t["pixels"]/counts[j],
                    "fraction_of_method_total_sse": t["squared_error_sum_mm2"]/existing["results"][method]["squared_error_sum_mm2"]})
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        for key in ("valid_pixels", "abs_rel_sum", "squared_error_sum_mm2", "delta_1_25_count"):
            np.testing.assert_allclose(sum(r[key] for r in method_rows), existing["results"][method][key], rtol=1e-12, atol=1e-8)
        sse = sum(r["squared_error_sum_mm2"] for r in method_rows)
        for threshold in (50., 100.):
            t = tails[method][threshold]
            tail_rows.append({"method": method, "absolute_error_threshold_mm": threshold, **t,
                "fraction_of_valid_pixels": t["pixels"]/total, "fraction_of_method_sse": t["squared_error_sum_mm2"]/sse,
                "tail_mse_contribution_mm2": t["squared_error_sum_mm2"]/total,
                "sqrt_tail_mse_component_mm": float(np.sqrt(t["squared_error_sum_mm2"]/total)),
                "max_absolute_error_mm": max_errors[method]})
    transition_rows = []
    for j, label in enumerate(DEPTH_LABELS):
        values = {key: int(value[j]) for key, value in transition_counts.items()}
        if sum(values.values()) != counts[j]:
            raise RuntimeError("Delta transition categories do not partition the valid pixels")
        net = values["log_only_pass"] - values["global_only_pass"]
        transition_rows.append({"depth_bin_mm": label, "valid_pixels": int(counts[j]), **values,
            "net_delta_pass_change": net, "net_within_bin_delta_change": net/int(counts[j]),
            "contribution_to_global_delta_change_pp": 100*net/total})
    expected_change = existing["results"]["log_affine"]["delta_1_25_count"] - existing["results"]["global_scale"]["delta_1_25_count"]
    if sum(r["net_delta_pass_change"] for r in transition_rows) != expected_change:
        raise RuntimeError("Delta transition analysis differs from existing scores")
    with sources[-1].open(newline="", encoding="utf-8") as handle:
        frame_rows = [{"method": r["method"], "frame_id": r["frame_id"], "valid_pixels": int(r["valid_pixels"]),
                       **{metric: float(r[metric]) for metric in METRICS}} for r in csv.DictReader(handle)]
    descriptive, cases = describe_frames(frame_rows)
    frame_maps = {method: {r["frame_id"]: r for r in frame_rows if r["method"] == method} for method in METHODS}
    pairwise = {}
    for metric in METRICS:
        difference = np.array([frame_maps["log_affine"][p.frame_id][metric] - frame_maps["global_scale"][p.frame_id][metric]
                               for p in dataset.pairs])
        better = difference > 0 if metric == "delta_1_25" else difference < 0
        worse = difference < 0 if metric == "delta_1_25" else difference > 0
        pairwise[metric] = {"log_affine_better_frames": int(better.sum()), "log_affine_worse_frames": int(worse.sum()),
                            "equal_frames": int(np.count_nonzero(difference == 0))}
    for path in sources:
        if file_sha256(path) != fingerprints[str(path)]:
            raise RuntimeError("An existing parameter or result file was modified")
    write_csv(args.output / "depth_bin_metrics.csv", rows)
    write_csv(args.output / "delta_transitions.csv", transition_rows)
    write_csv(args.output / "large_error_contributions.csv", tail_rows)
    write_csv(args.output / "large_error_by_depth_bin.csv", bin_tail_rows)
    write_csv(args.output / "frame_metrics.csv", frame_rows)
    write_csv(args.output / "frame_distributions.csv", descriptive)
    write_csv(args.output / "representative_cases.csv", cases)
    summary = {"status": "complete", "completed_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "protocol_id": protocol["protocol_id"], "sequence": "trans_t2_b", "frames": len(dataset),
        "valid_pixels": total, "depth_bin_pixel_counts": dict(zip(DEPTH_LABELS, map(int, counts))),
        "delta_transition_totals": {key: int(values.sum()) for key, values in transition_counts.items()},
        "net_delta_pass_change": expected_change, "delta_change_pp": expected_change/total*100,
        "frame_pairwise_log_affine_vs_global_scale": pairwise, "representative_cases": cases,
        "source_fingerprints": fingerprints, "parameters_refitted": False, "model_inference_performed": False,
        "formal_results_modified": False, "bin_statistics_reconstruct_existing_global_sufficient_statistics": True,
        "temporal_caution": "103 sequential frames from one video, not 103 independent clinical cases; descriptive statistics only"}
    write_json(args.output / "analysis_summary.json", summary)
    print(f"Depth-bin counts: {summary['depth_bin_pixel_counts']}")
    print(f"Delta transitions: {summary['delta_transition_totals']}; net {expected_change}")
    print(f"Frame comparison: {pairwise}")


if __name__ == "__main__":
    main()
