# BN5211 单目内镜深度估计

BN5211 Medical Robotics Intelligence 课程项目：**Depth Estimation from Monocular or Stereo Endoscopic Images**。

已完成基础实验、Global Scale Calibration、Log-Affine Calibration、最终可视化修正、误差分析及完整英文报告初稿。当前进入团队审阅阶段，尚未生成最终报告 PDF 或 Presentation。

## 从这里开始

- [GitHub 完整备份与恢复](GITHUB_BACKUP.md)：代码、报告与结果保存在仓库；原始数据和预测数组保存在同仓库的 Release 附件。
- [完整英文报告初稿](report/draft/report_draft.md)：六章正文及摘要，含表格/图注约 3,325 词，含参考文献约 3,540 词。
- [报告目录说明](report/README.md)：图表与数据来源、重建方法和科学表述约定。
- [独立 Figure Captions](report/draft/figure_captions.md)及[BibTeX 参考文献](report/draft/references.bib)：7 项真实论文与官方资料。
- [待补充信息与审阅清单](report/draft/user_information_checklist.md)：团队信息、课程格式和最终审稿事项。
- [工程交接](PROJECT_CONTEXT.md)：继续开发所需的固定协议、参数、结果及当前工作边界。

下一步是补齐姓名、学号、组号等信息，并按课程官方模板审稿和排版。工程中未找到官方项目说明 DOCX，页数/字数限制、提交格式及 Grading Notes 尚待确认。现阶段无需重跑推理或校准。

## 正式测试结果

测试集为 `trans_t2_b`，103 帧，136,655,314 个有效像素。下表采用全有效像素汇总；AbsRel、δ 无量纲，RMSE 单位为 mm。

| Method | AbsRel ↓ | RMSE (mm) ↓ | δ < 1.25 ↑ |
| --- | ---: | ---: | ---: |
| Constant-Median-Depth | 0.54988718 | 17.71215573 | 0.27518633 |
| Original Depth Anything V2 | 27.06870370 | 418.08559985 | 0.00000000 |
| Global Scale Calibration | 0.46851230 | 16.69069083 | **0.30924857** |
| Log-Affine Calibration | **0.41913838** | **13.12062839** | 0.28966559 |

完整精度见 [四方法正式结果](outputs/contrast_calibration/test_summary.json)。[原始 Baseline/零样本结果](outputs/results/summary.json)与[Global Scale 结果](outputs/calibration/test_summary.json)独立保留。

原始模型存在明显尺度失配。Log-Affine 相对 Global Scale 的测试 AbsRel 降低 10.54%、RMSE 降低 21.39%，但 δ 下降 1.9583 个百分点，主要损失来自 10–20 mm 区间。深度动态范围有所恢复，远深度仍明显低估，局部仍存在大误差。

只有原始预训练推理属于相对于本项目开发标签的 zero-shot。Baseline 和两种校准均使用开发集 GT；没有训练或微调模型。103 帧来自同一连续测试视频，不能当成独立临床病例。

## 工程与数据位置

工程根目录：`E:\BN5211小组作业`。配置中的相对路径均相对于工程根目录解释，数据原地使用。

```text
trans_t1_a/trans_t1_a/       开发集，61 帧
trans_t2_b/trans_t2_b/       测试集，103 帧
src/                        数据读取、模型、Baseline、指标、校准与误差分析
configs/default.json        数据路径、模型 revision 和设备配置
scripts/                    已完成实验及绘图脚本
scripts/build_report_assets.py  仅重建报告图表
tests/                      人工数组和临时文件检查
outputs/results/            原始正式结果、协议、103 帧原始预测
outputs/calibration/        尺度因子、61 帧开发预测及 Global Scale 结果
outputs/contrast_calibration/   Log-Affine 参数和正式结果
outputs/error_analysis/      深度段、δ 转移、长尾、逐帧分析及英文简稿
outputs/final_visualizations/   已修正色标的 0000、0051、0102 总览
report/draft/               英文报告、图注、参考文献与待补充清单
report/figures/             5 张主图、1 张补充图
report/tables/              4 组 Markdown/CSV 表格及 5 份分析 CSV
.cache/huggingface/         已缓存模型权重，约 99 MB
PROJECT_CONTEXT.md          当前工程交接
```

原始预测为毫米浮点数组：测试位于 `outputs/results/predictions_mm/`，开发位于 `outputs/calibration/development/predictions_mm/`。两种校准由原始预测及冻结参数重建，未额外保存完整校准预测副本。

C3VD RGB 来自实体结肠模型的光学录像，GT 来自配准后的三维模型渲染；这些数据不是患者病例，也不是全合成 RGB。

## 冻结模型、协议与参数

模型为 [Depth Anything V2 Metric Indoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)，固定 revision：

```text
8078d68a9c75a972131914f6afd0c1723be0da7f
```

协议为 [c3vd-interior-v1](outputs/results/protocol.json)：

- GT 为相机 Z 轴深度，`depth_mm = depth_raw * 100 / 65535`；模型原始米输出乘 1000。
- 正式掩码为 `0 < depth_raw < 65535`，是本项目约定；数据模块默认存储掩码不等于正式评价掩码。
- 所有方法使用相同 GT 掩码；无额外裁剪、预测依赖过滤或逐图归一化。
- AbsRel 为平均相对绝对误差；RMSE 从全体像素平方误差和开方；δ 采用严格 `< 1.25`。
- 全局成绩按有效像素池化，以 float64 累积，不直接平均逐帧 RMSE。

仅由开发集拟合的冻结参数：

| 方法 | 参数/公式 | 来源 |
| --- | --- | --- |
| Constant Median | 17.25947966735332 mm | [baseline_fit.json](outputs/results/baseline_fit.json) |
| Global Scale | `prediction_mm * 0.03821439952684151` | [scale_fit.json](outputs/calibration/scale_fit.json) |
| Log-Affine | `exp(a + b * ln(prediction_mm))`；`a=-26.367917618101938`，`b=4.791517002825514` | [log_affine_fit.json](outputs/contrast_calibration/log_affine_fit.json) |

Log-Affine 使用自然对数，输入单位必须为 mm。参数冻结后统一应用于全部测试帧，没有逐帧尺度拟合或测试 GT 调参。

## 环境与当前可用命令

已完成推理使用 `.venv-gpu\Scripts\python.exe`：Python 3.13.5、PyTorch 2.11.0+cu126、CUDA 12.6、Transformers 4.57.6、NumPy 2.4.4、Pillow 12.1.1，RTX 3060 Laptop GPU（6 GB），batch size 1、float32、不使用 autocast。已记录环境见 [gpu_environment_freeze.txt](configs/gpu_environment_freeze.txt)，不需要重新配置 CUDA。

`.venv\Scripts\python.exe` 为现有 CPU 备用及绘图环境，使用 Matplotlib 3.10.8；它继承 Miniconda 的部分包。两个环境均依赖基础解释器 `D:\miniconda\python.exe`。现有环境可直接复用，不需要重建。

在工程根目录执行，无需激活环境：

```powershell
# 只读取已有 JSON/CSV/PNG，重建报告图表和独立图注；已验证成功
.\.venv\Scripts\python.exe -m scripts.build_report_assets

# 如修改数值代码，可运行人工数据检查；本轮文档更新未运行
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -v
```

报告图表脚本只改写自己的 `report/` 产物，不运行模型、不读取完整预测数组、不拟合或重新评价，也不覆盖手写报告与待补充清单。无需为了阅读结果重跑准备、推理、拟合或评价脚本；这些实验均已完成。

## 报告图表与可视化约定

| 图 | 内容 |
| --- | --- |
| Figure 1 | 开发监督、参数冻结、测试评价流程 |
| Figure 2 | 固定测试帧 0102，复用已修正的 RGB/GT/原始预测/两校准/误差总览 |
| Figure 3 | 四方法三个正式指标 |
| Figure 4 | 五个 GT 深度区间的三指标 |
| Figure 5 | 开发 GT 与两校准的 P05/Median/P95 区间，不是概率密度或置信区间 |
| Supplementary S1 | 复用逐帧时间曲线与箱线图 |

GT 与两种校准预测统一使用 0–100 mm，原始预测独立使用明确标注的 0–1300 mm；误差图统一使用 0–100 mm。排除区域统一灰色，显示饱和不改变实际数值或成绩。早期 `outputs/results/visualizations/` 共用 0–1300 mm 的 GT 图仅保留为历史产物，报告优先使用修正版本。

0102 中央灰区已在之前核实为原始 uint16 值 65535（100 mm 饱和边界），由冻结协议排除；详情见 [visualization.json](outputs/final_visualizations/visualization.json)。

当前没有未解决工程阻碍。下一步仅团队审稿、补全信息及按官方要求排版；已有预测、参数、协议和正式成绩应保持不变。
