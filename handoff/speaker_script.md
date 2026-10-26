# 12-minute script and 3-minute Q&A

Speak in English. Times include short pauses on the figures. Do not claim a new network, clinical accuracy, or independent patients. Numbers are the frozen test scores.

## Slide 1 — Title (0:00–0:40)

This is our BN5211 project on monocular depth estimation for endoscopic images. We did not train a new depth network. We took one released general-purpose model, Depth Anything V2 Metric Indoor Small, and asked how it behaves on C3VD colonoscopy images. We then compared two calibrations that were fitted only on the development sequence.

## Slide 2 — Why this task (0:40–1:40)

Endoscopic robotics needs a sense of how far the tissue is from the camera. A single image does not determine that distance. C3VD lets us measure the error because it pairs real colonoscope video of a physical colon model with registered depth. These are phantom recordings, not patient examinations, and the ground truth is rendered from the registered model. The clinical motivation is spatial perception. The experiment itself is a controlled transfer test.

## Slide 3 — Protocol (1:40–2:45)

The course split is fixed. `trans_t1_a` has 61 development frames. `trans_t2_b` has 103 test frames. Depth is camera Z, decoded as raw times 100 over 65535, in millimetres, over 0 to 100 mm. The evaluation mask keeps codes strictly between 0 and 65535. Every method uses that same mask. There are about 136.7 million valid test pixels. We report AbsRel, RMSE in millimetres, and the fraction of pixels whose ratio is strictly below 1.25. Scores pool pixels. We do not average frame RMSE, and we do not tune anything on the test labels.

## Slide 4 — Four methods (2:45–4:20)

The baseline paints every pixel with the development median, 17.26 mm. It has no spatial structure. The original model is the frozen indoor metric checkpoint. Only this output is zero-shot with respect to our development labels. Its development median is about 452 mm, roughly 26 times the true median, so the scale factor is 0.03821. Global scale multiplies every prediction by that one number. Log-affine is `exp(a + b ln D)`, with a about −26.37 and b about 4.79, fitted by least squares in log depth on all valid development pixels, then frozen. Both calibrations use development ground truth. Neither one changes the network.

## Slide 5 — Figure 2, frame 0102 (4:20–5:40)

This is the last test frame, chosen by position, not because it looks good. Ground truth and the two calibrated maps share 0 to 100 mm. The original prediction uses a separate 0 to 1300 mm scale, because it lives on the wrong scale. Gray pixels are excluded by the mask. The central gray region is raw code 65535, the 100 mm encoding boundary, not a hole in the prediction. Global scale looks flatter than the ground truth. Log-affine restores more contrast, but it does not recreate the folds. A constant-median map is not drawn here. It would be one colour, at 17.26 mm. Figure 3 is where that baseline is compared numerically.

## Slide 6 — Figure 3 (5:40–7:10)

Read the three numbers, not a single winner. The original model has AbsRel 27.07, RMSE 418 mm, and delta 0. It is worse than predicting 17.26 mm everywhere. Global scale reaches AbsRel 0.469, RMSE 16.69 mm, and delta 0.309. Log-affine reaches AbsRel 0.419 and RMSE 13.12 mm, the best of those two, but delta falls to 0.290. Both calibrations beat the constant baseline on all three metrics. Between the two calibrations, lower error and higher threshold accuracy are different goals.

## Slide 7 — Figure 4 (7:10–8:40)

The 10 to 20 mm bin holds 38.8% of the valid pixels. Global scale is strong there. Log-affine makes that bin worse: RMSE rises from about 3.1 mm to 6.2 mm, and delta falls from 0.69 to 0.31. Farther bins move the other way. In the 60 to 100 mm bin, mean ground truth is 76 mm, while log-affine still predicts about 36 mm. So the lumen is still too shallow. The aggregate RMSE can fall while many already-acceptable near pixels leave the 1.25 band.

## Slide 8 — What the trade-off means (8:40–10:00)

Improving global depth error does not necessarily put more pixels inside the threshold. The net loss is about 2.7 million pixels, and the 10 to 20 mm bin loses about 20.5 million passes before the other bins add some back. Development Pearson correlation averages 0.784. That means the model has some large-scale near-far structure. It does not mean the folds are correct. Because both maps are increasing functions, neither calibration can repair a locally wrong ordering.

## Slide 9 — Limits (10:00–11:05)

Do not describe this as label-free deployment. Do not call 103 frames 103 patients. They are one continuous video. Development scores reuse the fitting data, so they are not a second test. We did not isolate specular highlights, camera intrinsics, or domain shift as separate causes. Far depth remains underestimated, and a few near pixels have very large log-affine errors. Those pixels were kept in the scores.

## Slide 10 — Close (11:05–12:00)

A frozen general model transferred some relative structure and failed on absolute scale. One development scale factor repaired most of the magnitude and beat the constant baseline. A log-affine map improved AbsRel and RMSE further, widened the depth range, and paid for that with a mid-range delta loss. The useful result is the trade-off, not a claim of clinical depth. I can take questions on the supervision, the metrics, and what we deliberately did not do.

## Q&A rehearsal (3 minutes)

Answer in two or three sentences. If several are asked, skip the repeated caveat about patients.

**Is the deep model zero-shot?** Only the untouched checkpoint is zero-shot relative to our development labels. The constant baseline and both calibrations are fitted on development ground truth and then frozen. Test labels were not used to choose parameters.

**Why is delta lower if RMSE is better?** RMSE penalises large millimetre errors, especially far away. Delta counts pixels inside a ratio of 1.25. Log-affine stretches depth, which helps the far bins and pushes some 10–20 mm pixels out of the band. That bin is 39% of the pixels, so the overall delta falls by about two percentage points.

**Can we use this clinically?** No. The sequences are a physical colon model, not patients. Even the best delta is about 0.31, and 60–100 mm is still predicted at about half the true mean depth.

**Why not fit a scale on each test frame?** A scale from that frame's ground truth would leak the test labels. We report one global map fitted on `trans_t1_a` only.

**Did the gray patch in frame 0102 fail?** No. Those pixels are raw value 65535. The protocol excludes them before scoring. The same mask is used for every method.

**Which frame fails?** For log-affine AbsRel, frame 0075 is the lowest and frame 0036 is the highest. That is one metric on one video, not a clinical ranking. Frame 0102 is shown because it is the predetermined last frame.

**Why this model?** The assignment asks for a released general monocular depth model, compared with a constant-median baseline. We used the small indoor metric checkpoint and did not fine-tune it.

**What would you do next?** Evaluate on another sequence that is not used for fitting, and judge errors against a robotic tolerance. That experiment is not in these results.
