# Development-set global-scale calibration

One factor from all valid development pixels; original test predictions reused without inference.

| Method | AbsRel ↓ | RMSE (mm) ↓ | δ < 1.25 ↑ |
| --- | ---: | ---: | ---: |
| Constant-Median-Depth | 0.54988718 | 17.71215573 | 0.27518633 |
| Original Depth Anything V2 | 27.06870370 | 418.08559985 | 0.00000000 |
| Depth Anything V2 + Global Scale Calibration | 0.46851230 | 16.69069083 | 0.30924857 |
