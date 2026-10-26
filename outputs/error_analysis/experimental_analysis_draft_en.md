# Experimental error analysis — draft

## Scope and aggregation

We analyzed 136,655,314 GT-valid pixels from 103 consecutive frames of the C3VD trans_t2_b sequence. Depth bins were fixed as [0,10), [10,20), [20,40), [40,60), and [60,100) mm. The existing mask, 0 < raw depth < 65535, and all model/calibration parameters were unchanged. Bin metrics pool pixels across frames; frame distributions are descriptive summaries, not substitutes for the pooled scores. No model inference, fitting, prediction clipping, or test-driven parameter selection was performed. These frames are temporally correlated observations from one video, not 103 independent clinical cases.

## Observed depth dependence

The original model had an absolute error above 100 mm at every valid test pixel, indicating a pervasive magnitude mismatch in this sequence rather than an error restricted to a few outliers. Its AbsRel ranged from 54.095 in the 0–10 mm bin to 5.890 in the 60–100 mm bin. This numerical pattern does not establish a specific physical, imaging, or training-related cause.

Global scale calibration concentrated predictions near the middle of the depth range. It performed best in the 10–20 mm bin (AbsRel 0.1935, RMSE 3.0885 mm, delta 0.6930), which contained 38.76% of valid pixels, but strongly underestimated larger depths. In the 60–100 mm bin, mean GT was 76.10 mm whereas the global-scale mean prediction was 19.64 mm. Log-affine calibration raised that mean to 36.27 mm, but a mean signed error of -39.83 mm and 99.47% underestimation remained.

Log-affine calibration improved 40–60 mm RMSE from 30.0464 to 19.4660 mm and 60–100 mm RMSE from 57.5270 to 42.7622 mm. Its effects were not uniformly favorable: 10–20 mm AbsRel increased from 0.1935 to 0.3502 and RMSE from 3.0885 to 6.2021 mm. In 20–40 mm, RMSE and delta improved but AbsRel rose slightly (0.3317 to 0.3364). In 0–10 mm, AbsRel and delta improved, while RMSE rose slightly from 7.9638 to 8.0017 mm. The fixed log-affine exponent expands contrast, but the remaining errors cannot be reduced to a single scale mismatch.

## Why threshold accuracy decreased

Pixelwise transitions directly locate the delta trade-off. Across the full sequence, 29,304,468 pixels passed only under global scale, while 26,628,349 passed only under log-affine calibration: a net loss of 2,676,119, or 1.9583 percentage points. The 10–20 mm bin alone lost 20,498,249 net passes, contributing -15.0000 points to the global change. The other four bins all gained and offset most of that loss. In 10–20 mm, residual SD increased from 2.8150 to 6.1457 mm even as the magnitude of mean signed bias decreased. Thus reduced average bias did not imply more predictions inside the multiplicative 1.25 threshold. Minimizing development log-depth squared error does not guarantee threshold accuracy on the test sequence.

## Large errors and frame heterogeneity

For log-affine calibration, errors above 50 mm occurred in 1.1930% of pixels but contributed 24.9109% of total squared error. Errors above 100 mm occurred in 3,958 pixels (0.002896%) and contributed 1.1289% of squared error; all were in the 0–10 mm GT bin. The maximum absolute error was 1931.15 mm. These tails matter, but errors above 100 mm alone do not dominate the total RMSE. All remained in the formal scores.

Frame behavior also varied. Log-affine calibration improved AbsRel in 83/103 frames and RMSE in 96/103, but improved delta in 53 and worsened it in 50. Its frame-level AbsRel median [IQR] was 0.4128 [0.3799, 0.4594], compared with 0.4594 [0.4487, 0.4738] for global scale. Its RMSE median was 11.6739 versus 16.5002 mm, while delta had a much wider frame range (0.0663–0.5370 versus 0.1927–0.3843). By the prespecified AbsRel ranking, log-affine frame 0075 was best (0.2764) and frame 0036 worst (0.6369); these are metric-specific illustrative cases, not overall rankings of clinical difficulty.

## Interpretation limits

The evidence supports correction of scale and part of the compressed depth contrast, with persistent depth-dependent bias, residual spread, and occasional large local errors. It does not identify reflection, texture, illumination, or domain shift as causal explanations. The descriptive analysis neither establishes clinical generalization nor changes the frozen methods. Figures A–C summarize pixel-pooled bin metrics; Figure D shows frame distributions and temporal order without independence assumptions or inferential error bars.
