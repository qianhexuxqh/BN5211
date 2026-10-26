# BN5211 当前工程交接

更新日期：2026-10-24。工程根目录：`E:\BN5211小组作业`。

## 当前阶段

算法实验、两种开发集校准、误差分析及完整英文报告初稿均已完成，当前等待团队审阅。最新一轮仅整理 README 等说明，没有修改报告正文、代码、图表、预测、冻结参数或正式成绩，没有执行推理、拟合、评价或环境检查。

课程核心要求已按用户提供内容完成：C3VD、公开通用单目模型、Constant Median Baseline、AbsRel/RMSE/δ、针对性改进比较。官方 `BN5211_Final_Project_Instruction_AY2026-2027.docx` 不在工程，尚不能直接确认篇幅、模板、Submission Requirements 和 Grading Notes。没有生成最终报告 PDF 或 Presentation。

## 报告入口和组织

优先阅读 [英文报告](report/draft/report_draft.md)、[报告说明](report/README.md)和[用户待补充清单](report/draft/user_information_checklist.md)。

报告题目：General-Purpose Monocular Depth Estimation for Endoscopic Images: Evaluation and Target-Domain Calibration。正文含表格/图注约 3,325 个空白分隔词，含 7 项参考文献约 3,540 词。

六章为团队信息、背景、方法、实验与结果、结论与讨论、参考文献，另含摘要。方法给出单位、掩码、监督范围、两校准公式及池化指标；结果涵盖整体成绩、深度分段、δ 转移、分布、长尾及帧统计；讨论区分观测与推测。

| 产物 | 路径与内容 |
| --- | --- |
| 主图 | `report/figures/`：1 流程；2 复用 0102 修正总览；3 四方法三指标；4 深度分段；5 开发分位数 |
| 补充图 S1 | 复用已有逐帧时序/箱线图，原图标题 D 保留 |
| 表格 | `report/tables/`：1 测试成绩；2 开发成绩；3 深度段对比；4 开发分位数；各 Markdown/CSV，另有 5 份分析 CSV |
| 图注与文献 | `report/draft/figure_captions.md`、`references.bib` |
| 产物来源与核对 | `report/asset_manifest.json`、`preparation_checks.json`，记录报告准备阶段的核对 |

新绘 4 图提供 300 dpi PNG 和 SVG，复用 2 图保留原 PNG；报告准备阶段均已查看。引用包括 C3VD 2023、Eigen 2014、Ranftl 跨数据集迁移预印本 2019/修订 2020、DAV2 NeurIPS 2024，以及 3 项官方技术资料。Ranftl 按已核实预印本引用，没有编造期刊卷页。

## 固定数据、模型与环境

- 开发集：`trans_t1_a/trans_t1_a/`，61 帧，80,952,223 个有效像素。
- 测试集：`trans_t2_b/trans_t2_b/`，103 帧，136,655,314 个有效像素；连续视频帧，不是 103 个独立临床病例。
- 原始预测：开发 `outputs/calibration/development/predictions_mm/`；测试 `outputs/results/predictions_mm/`；1080 × 1350 float32 毫米数组。
- 模型：`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`；revision `8078d68a9c75a972131914f6afd0c1723be0da7f`；权重缓存 `.cache/huggingface/`。
- 既有推理环境：`.venv-gpu\Scripts\python.exe`，Python 3.13.5、PyTorch 2.11.0+cu126、CUDA 12.6、Transformers 4.57.6、NumPy 2.4.4、Pillow 12.1.1，RTX 3060 Laptop 6 GB；batch size 1、float32、无 autocast。
- 报告绘图环境：`.venv\Scripts\python.exe`，CPU，Matplotlib 3.10.8。两个环境依赖 `D:\miniconda\python.exe`，现有环境可复用。

C3VD RGB 为实体结肠模型的真实光学录像，GT 为配准后三维模型的渲染深度。不能将本数据描述成患者数据或全合成 RGB。

## 冻结评价与参数

协议为 `outputs/results/protocol.json` 中的 `c3vd-interior-v1`。GT 是相机 Z 深度，`raw * 100 / 65535` 转 mm；模型米输出乘 1000。正式掩码为 `0 < raw < 65535`，是项目约定，不是官方强制规则；数据模块默认存储掩码不能代替它。

全部有效像素池化，float64 指标累积。RMSE 为总平方误差除以总像素数后开方，不平均帧 RMSE；δ 严格 `<1.25`。各方法使用同一 GT 掩码，不裁剪预测、不做逐图归一化或预测依赖过滤。

- Baseline：开发 GT 全局中位数 **17.25947966735332 mm**，来自 `outputs/results/baseline_fit.json`。
- Global Scale：开发 GT 中位数除以预测中位数 **451.6485900878906 mm**，得到 **0.03821439952684151**，来自 `outputs/calibration/scale_fit.json`。应用 float64 乘法后转 float32。
- Log-Affine：`corrected_mm = exp(a + b * ln(original_prediction_mm))`，自然对数，毫米输入；**a=-26.367917618101938**、**b=4.791517002825514**，来自 `outputs/contrast_calibration/log_affine_fit.json`。用全部开发有效像素、流式 float64 中心矩拟合对数空间最小二乘，应用保留 float64。

参数只由开发 GT 和对应预测拟合，先冻结再用于测试，未使用测试 GT 调整参数。原始推理是相对于本项目开发标签的 zero-shot；Baseline 和两种校准均有开发监督。没有训练或微调。校准全帧数组没有另存，可由原始预测与冻结参数重建。

## 正式成绩与关键发现

测试集（AbsRel / RMSE mm / δ）：

| 方法 | AbsRel | RMSE (mm) | δ < 1.25 |
| --- | ---: | ---: | ---: |
| Constant Median | 0.54988718 | 17.71215573 | 0.27518633 |
| Original DAV2 | 27.06870370 | 418.08559985 | 0.00000000 |
| Global Scale | 0.46851230 | 16.69069083 | 0.30924857 |
| Log-Affine | 0.41913838 | 13.12062839 | 0.28966559 |

开发集：Original 为 28.67227614 / 430.62626127 / 0；Global Scale 为 0.50868805 / 16.98462983 / 0.29631624；Log-Affine 为 0.36418687 / 11.36445892 / 0.43914866。开发成绩使用拟合数据，不能当成独立验证。

开发 P05/Median/P95（mm）：GT 6.21194781 / 17.25947967 / 60.33417258；Global 14.07882023 / 17.25947952 / 20.46521759；Log 7.00191592 / 18.58152590 / 42.03386409。原始预测中位数约为 GT 的 26.17 倍；开发逐帧 Pearson 均值 0.78373139，61 帧全正，无零方差帧。相关性只支持相对深度关联，不证明精确几何或学习机制。

Log 相对 Global 的测试 AbsRel 降低 10.54%、RMSE 降低 21.39%，δ 下降 1.9583 个百分点。10–20 mm 占有效像素 38.76%，该段 δ 从 0.69297 降至 0.30600，净损失 20,498,249 个通过像素；其他四段净增加，抵消后全局净损失 2,676,119。该段残差 SD 从 2.8150 增至 6.1457 mm，尽管平均偏差幅度减小。不能声称三指标全面改善。

40–60、60–100 mm 的 Log RMSE 改善，但远深度仍低估：60–100 mm 段 GT 均值 76.10 mm，Global 19.64 mm、Log 36.27 mm。Log 绝对误差 >100 mm 的 3,958 像素占 0.002896%，贡献总 SSE 的 1.1289%，全部位于 GT 0–10 mm 段；保留这些像素，没有删尾替代成绩。Log 在 83/103 帧改善 AbsRel、96/103 帧改善 RMSE，δ 为 53 帧改善、50 帧恶化。以上为描述分析，不作独立性显著性推断或临床性能声明。

## 结果来源与显示约定

- `outputs/results/`：原始 Baseline/零样本正式成绩、协议、103 帧预测及逐帧 CSV。
- `outputs/calibration/`：开发预测、尺度参数、分布、Pearson 和尺度校准成绩。
- `outputs/contrast_calibration/`：a/b、开发与测试正式成绩及四方法比较。
- `outputs/error_analysis/`：深度段、δ 转移、长尾、帧统计与原英文分析草稿。
- `outputs/final_visualizations/`：固定帧 0000、0051、0102 的修正总览。GT/两校准共用 0–100 mm；原始预测独立明确标 0–1300 mm；误差统一 0–100 mm；排除区域灰色，显示饱和不改数值。

0102 中央灰区此前针对性核实为 uint16 全部 65535，对应 100 mm 饱和边界。四连通区域 23,325 像素，种子 (x=750,y=400)，记录在 `outputs/final_visualizations/visualization.json`。未发现读取、配对或单位错误。报告优先用修正图，早期 GT 共用 0–1300 mm 的图保留为历史产物。

## 当前命令、验证与下一步

在工程根目录重建报告图表：

```powershell
.\.venv\Scripts\python.exe -m scripts.build_report_assets
```

该命令在报告准备阶段已成功执行，只读保存的 JSON/CSV/PNG，重建自己的报告图表及独立图注，不覆盖手写稿；不调用模型、不读完整预测、不拟合、不重新评价。报告准备阶段已核对数值、链接、复用图与来源文件，详情见 `report/preparation_checks.json`。本轮仅文档更新，未重跑实验、测试或图表生成。

下一步：用户补齐成员姓名/学号/组号、机构、提交日期、官方模板/篇幅/引用格式和课程要求的贡献/披露信息；团队审稿后再决定最终排版与导出。清单见 `report/draft/user_information_checklist.md`。现有实验足以支持当前初稿，不自动开展新实验或依据已见测试误差调参。

当前无未解决工程阻碍。本轮文档整理未遇到实际问题。
