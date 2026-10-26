# Figure captions

**Figure 1.** Completed experimental workflow. Development GT fits the constant baseline and both calibrations; the frozen network supplies image-dependent predictions. Parameters are then fixed for test evaluation. Test GT contributes only to the evaluation path. Original DAV2 is zero-shot with respect to project development labels; the calibrations are development-supervised.

**Figure 2.** Fixed test frame 0102: RGB, GT, original DAV2, both calibrations, and their absolute error maps. GT and both calibrated predictions share viridis at 0–100 mm. Original prediction uses an explicitly separate 0–1300 mm scale; all error maps share magma at 0–100 mm. Gray denotes the same GT-excluded pixels in every map. Display saturation is marked and does not clip numerical predictions or scores. The central excluded GT component contains raw code 65535, the 100 mm encoding boundary. No per-image normalization is used.

**Figure 3.** Formal test scores for all four methods. AbsRel and RMSE use clearly labeled logarithmic axes to accommodate the original model's much larger errors. δ uses a linear axis and is a fraction. Bar annotations are rounded; Table 1 retains eight decimal places. There are no inferential error bars because this experiment contains one continuous test sequence.

**Figure 4.** Pixel-pooled test performance in the fixed GT depth bins for all four methods. Tick labels include each bin's proportion of valid pixels. AbsRel and RMSE use logarithmic axes; δ is linear. The 10–20 mm penalty contrasts with improvements at greater depths. All bins use the frozen GT mask, and no large-error pixels are removed.

**Figure 5.** Development depth distribution summarized by saved exact pooled quantiles. Lines span P05–P95 and diamonds mark medians on one shared depth axis. This is a quantile summary rather than an estimated density. Log-affine expands the compressed distribution but does not fully match GT. Values outside these intervals remain included in all formal metrics.

**Supplementary Figure S1.** Reused frame-level analysis across the 103 consecutive test frames. Left: metrics in temporal order. Right: descriptive boxplots with median, interquartile range, and 1.5-IQR whiskers. Points beyond whiskers remain in the scores. AbsRel and RMSE use log axes; delta is linear. The original analysis title 'D' is retained in this unchanged figure. Frames are correlated observations, not independent clinical cases; no confidence intervals or significance tests are implied.
