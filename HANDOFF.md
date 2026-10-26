# 交给下一位同学

改进实验已经完成。请从本仓库继续，不必再要网盘。仓库是私有的，用现有 collaborator 权限打开即可：<https://github.com/qianhexuxqh/BN5211>。

截止 **2026-11-02 10:00**（新加坡时间）。下一位负责 **文献与报告**。**分析与展示** 可以同时做幻灯片，不用等新实验。

## 你要交的文件

Canvas 由组长交一次，三个文件：

- `GroupNumber_report.pdf`
- `GroupNumber_code.zip`
- `GroupNumber_slides.pdf`

组号、姓名和学号还不在仓库里，所以上面的 `GroupNumber` 和报告标题页都是占位。上传前换成真实信息。不要编造组员。

## 先读这些

1. 本文件。
2. 英文初稿：[report/draft/report_draft.md](report/draft/report_draft.md)。六章已经按课程要求写好。
3. 草稿 PDF、10 页幻灯片和 12 分钟讲稿：`handoff/`。数字没有重算。
4. 协议和参数细节：[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)。数据恢复：[GITHUB_BACKUP.md](GITHUB_BACKUP.md)。

原始图像和预测数组不在 Git 里。需要时到 Release `backup-2026-10-26` 下载 6 个 `bn5211-data-part-*.zip`，核对 `SHA256SUMS.txt` 后解压到仓库根目录。只改报告和幻灯片时不必下载。

## 正式结果，不要改

协议 `c3vd-interior-v1`。`trans_t1_a` 61 帧只拟合，`trans_t2_b` 103 帧、136,655,314 个有效像素只评价。模型是 Depth Anything V2 Metric Indoor Small，revision `8078d68a9c75a972131914f6afd0c1723be0da7f`。没有训练或微调。

权威文件是 `outputs/contrast_calibration/test_summary.json`。AbsRel 和 δ 无量纲，RMSE 单位是 mm。δ 要求预测与真值之比严格小于 1.25。

| Method | AbsRel | RMSE (mm) | δ < 1.25 |
| --- | ---: | ---: | ---: |
| Constant-Median-Depth | 0.54988718 | 17.71215573 | 0.27518633 |
| Original Depth Anything V2 | 27.06870370 | 418.08559985 | 0.00000000 |
| Global Scale | 0.46851230 | 16.69069083 | 0.30924857 |
| Log-Affine | 0.41913838 | 13.12062839 | 0.28966559 |

常数中位深度是 17.25947966735332 mm。Global Scale 的系数是 `s = 0.03821439952684151`。Log-Affine 是 `exp(a + b * ln(prediction_mm))`，`a = -26.367917618101938`，`b = 4.791517002825514`。深度解码是 `raw * 100 / 65535`，正式掩码是 `0 < raw < 65535`。

相对 Global Scale，Log-Affine 的测试 AbsRel 降低 10.54%，RMSE 降低 21.39%，δ 下降 1.958 个百分点。损失集中在 10–20 mm，约占有效像素 38.76%。60–100 mm 的平均真值约 76.10 mm，Log-Affine 平均预测约 36.27 mm。

只有原始模型相对开发集标签是 zero-shot。Baseline 和两种校准都用了开发集 ground truth。校准不是新网络。103 帧是同一段视频，不是 103 个病人。RGB 来自结肠模型，ground truth 是配准后的渲染深度。

不要覆盖 `outputs/results/`、`outputs/calibration/`、`outputs/contrast_calibration/`、`outputs/error_analysis/`，也不要重跑推理、拟合或测试评价。不要用测试集 ground truth 做逐帧尺度。不要再加第二个模型、高光处理或微调。

## 文献与报告

报告占 15 分，其中 5 分是说明本方法和已有工作的差别。初稿已有 Title and team、Background、Methods、Experiments and results、Conclusion and discussion、References。

- 填组号、姓名、学号、机构、日期。清单在 [report/draft/user_information_checklist.md](report/draft/user_information_checklist.md)。
- 文献用 [report/draft/references.bib](report/draft/references.bib)。C3VD 引 Bobrow et al., Medical Image Analysis, 2023。模型引 Depth Anything V2，NeurIPS 2024。
- 正文可以说校准改善了尺度，不能说提出了新的深度网络。Discussion 的主句是：整体误差下降不等于更多像素达到 δ < 1.25。
- 表保持八位小数。0102 图中央灰色区域是原始深度 65535，按协议排除，不是预测失败。
- 草稿在 `handoff/GroupNumber_report.pdf`（11 页）。它还不是 Canvas 终稿。

## 分析与展示

演示 12 分钟，问答 3 分钟。图已经画好，不用再跑模型。

- 幻灯片：`handoff/GroupNumber_slides.pdf`（10 页，含 Figure 2、3、4）。
- 讲稿和问答：`handoff/speaker_script.md`。
- 定量图已经包含 Constant Median。定性图没有常数深度列，因为那一列会是一整片 17.26 mm。
- Log-Affine 按 AbsRel，帧 0075 较低、帧 0036 较高。这只是一个指标，不是临床难度排序。

讲这三句：原始预测中位数约为真值的 26 倍；Global Scale 只修正尺度；Log-Affine 降低 AbsRel 和 RMSE，但在近处损失 δ。
