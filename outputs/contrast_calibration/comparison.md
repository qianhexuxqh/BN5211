# Global log-affine calibration comparison

Frozen development-only OLS in natural-log mm depth; no inference or clipping.

| Method | AbsRel ↓ | RMSE (mm) ↓ | δ < 1.25 ↑ |
| --- | ---: | ---: | ---: |
| Constant-Median-Depth | 0.54988718 | 17.71215573 | 0.27518633 |
| Original Depth Anything V2 | 27.06870370 | 418.08559985 | 0.00000000 |
| Global Scale Calibration | 0.46851230 | 16.69069083 | 0.30924857 |
| Log-Affine Calibration | 0.41913838 | 13.12062839 | 0.28966559 |
