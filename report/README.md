# Report preparation package

**Current status (24 October 2026):** the English draft and report assets are complete and awaiting team review. The remaining work is to supply team identifiers and official course requirements, review the draft, and then decide the final submission layout. Existing experiments are complete.

Quick navigation: [project overview](../README.md) · [current engineering handover](../PROJECT_CONTEXT.md) · [report draft](draft/report_draft.md) · [figure captions](draft/figure_captions.md) · [BibTeX references](draft/references.bib) · [pending information](draft/user_information_checklist.md).

Start with [the complete English report draft](draft/report_draft.md). The report follows the six sections requested by the team and contains an abstract, four numbered tables, five main figures, and seven verified references. [Standalone figure captions](draft/figure_captions.md) and a [user information checklist](draft/user_information_checklist.md) are supplied separately. No final report PDF or presentation has been created.

The draft is about 3,300 whitespace-delimited words including tables and captions but excluding references; it is an editable first draft, not a submission-ready formatted manuscript. The actual course page/word limit has not been supplied. The original engineering request's core experiment requirements are covered. No official `BN5211_Final_Project_Instruction_AY2026-2027.docx` was found in the workspace, so its Submission Requirements, Recommended Steps, Report Structure, and Grading Notes could not be checked directly.

## Figures and sources

| Figure | Content | Existing evidence used | Format |
| --- | --- | --- | --- |
| 1 | Experimental workflow; development supervision and test-only evaluation paths | Frozen protocol, median/scale/log-affine parameter files | 300 dpi PNG + editable SVG |
| 2 | Fixed test frame 0102; RGB, GT, original, two calibrations, errors | Byte-identical copy of `outputs/final_visualizations/0102_final_comparison.png` | Original 2800 × 2240 PNG |
| 3 | Four methods, three formal test metrics | `outputs/contrast_calibration/test_summary.json` | 300 dpi PNG + editable SVG |
| 4 | Five GT depth bins, four methods, three metrics | `outputs/error_analysis/depth_bin_metrics.csv` | 300 dpi PNG + editable SVG |
| 5 | Development GT/global-scale/log-affine P05, median, P95 | `outputs/contrast_calibration/development_summary.json` | 300 dpi PNG + editable SVG |
| S1 | Existing temporal and boxplot frame analysis | Byte-identical copy of the existing Figure D PNG in `outputs/error_analysis/figures/` | Original PNG |

Figure 5 intentionally shows saved quantiles, not an invented histogram or density. Its intervals are not full ranges or confidence intervals. Figure 2 uses a shared 0–100 mm scale for GT and both calibrations, a labeled independent 0–1300 mm scale for original prediction, and one shared error scale. Gray is the frozen GT exclusion mask. Numerical predictions are never normalized or clipped.

Other fixed qualitative cases are available as [frame 0000](../outputs/final_visualizations/0000_final_comparison.png) and [frame 0051](../outputs/final_visualizations/0051_final_comparison.png). Figure 2 uses the predetermined last frame rather than a metric-ranked success case.

## Tables and numerical provenance

`tables/` contains four Markdown/CSV table pairs matching the manuscript numbering: overall test results, development results, detailed calibration-by-depth results, and development quantiles. Filenames explicitly state their contents. Table 3's standalone assets retain separate columns and eight decimals, while the manuscript uses compact metric triplets for readability.

Five existing full-precision CSVs are also copied unchanged: all four methods' bin metrics, frame distributions, representative cases, delta transitions, and large-error contributions. In addition to the consolidated four-method test summary, the source protocol and original result summaries are listed with hashes in `asset_manifest.json`. These hashes track the small source artifacts used for this report; no new dataset-wide or system audit is involved.

| Claim or artifact | Authoritative file |
| --- | --- |
| Original baseline and zero-shot scores | `outputs/results/summary.json` |
| Frozen mask, units, aggregation | `outputs/results/protocol.json` |
| Baseline depth | `outputs/results/baseline_fit.json` |
| Single scale factor | `outputs/calibration/scale_fit.json` |
| Development raw distribution and Pearson summary | `outputs/calibration/development_summary.json` |
| Log-affine fit and units | `outputs/contrast_calibration/log_affine_fit.json` |
| Development metrics and exact quantiles | `outputs/contrast_calibration/development_summary.json` |
| Four-method test comparison | `outputs/contrast_calibration/test_summary.json` |
| Bin trade-offs, threshold changes, tails, frames | `outputs/error_analysis/report_digest.json` and accompanying CSVs |
| Original English analysis | `outputs/error_analysis/experimental_analysis_draft_en.md` |
| Display ranges and central gray-region explanation | `outputs/final_visualizations/visualization.json` |

## Rebuild report graphics only

From the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_report_assets
```

This command reads saved JSON/CSV files and existing PNGs. It rewrites only its generated files under `report/`; it never imports the depth model, reads the full prediction arrays, fits parameters, or reevaluates metrics. The English manuscript and user checklist are manually authored and are not overwritten. Figure captions are extracted from the manuscript when rebuilding.

The recorded inference environment is `.venv-gpu` (RTX 3060 Laptop, PyTorch 2.11.0+cu126); this report-only command uses the existing CPU plotting environment `.venv` with Matplotlib 3.10.8. No package or CUDA configuration was changed.

## Scientific wording that must be preserved

- Only original released-checkpoint inference is zero-shot with respect to project development labels. Both calibrations, and the constant baseline, use development GT.
- Scores pool pixels, and delta is a fraction. A delta change of −0.01958299 equals −1.9583 percentage points.
- Log-affine improves test AbsRel/RMSE but reduces delta relative to global scale. It is a standard calibration, not a novel model.
- C3VD RGB is captured from physical colon models; GT is rendered after registration. These are not patient cases or entirely synthetic RGB sequences.
- Correlation is descriptive evidence of depth association, not proof of a learned mechanism or clinical reliability.
- No unseen confirmatory dataset or statistical independence claim is available. Formal predictions and results remain unchanged.
