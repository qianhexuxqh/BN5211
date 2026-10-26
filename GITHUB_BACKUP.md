# GitHub 项目备份与恢复

本项目采用「Git 仓库 + 同仓库 Release 附件」备份。

- Git 仓库：源码、配置、依赖清单、报告、图表，以及实验 JSON/CSV/Markdown 结果。
- Release 附件：`trans_t1_a/`、`trans_t2_b/` 中的全部原始数据，以及 `outputs/` 中的全部 `.npy` / `.npz` 预测数组。
- 本地保留：`.venv/`、`.venv-gpu/`、`.cache/`、字节码和登录凭据。环境通过依赖清单重建，模型按配置中的固定 revision 重新下载。

因此，仅下载仓库源码不包含完整数据；恢复实验文件还需下载 Release 附件。

## 下载和恢复

1. 克隆仓库或下载源码 ZIP，并进入工程根目录。
2. 打开仓库的 **Releases** 页面，下载备份版本的全部 `bn5211-data-part-*.zip`、`backup-manifest.json` 和 `SHA256SUMS.txt`。
3. 用 PowerShell 的 `Get-FileHash <压缩包路径> -Algorithm SHA256` 核对每个 ZIP 的哈希与 `SHA256SUMS.txt` 一致。
4. 将每个 ZIP 解压到工程根目录。这些是可独立解压的 ZIP，不需要拼接，内部已保留相对路径。
5. 检查 `trans_t1_a/trans_t1_a/`、`trans_t2_b/trans_t2_b/` 以及 `outputs/results/predictions_mm/`、`outputs/calibration/development/predictions_mm/` 已恢复。文件级 SHA-256 见 manifest。
6. 根据 [README](README.md) 和 `requirements*.txt` 配置运行环境。阅读已有报告无需安装模型依赖。

## 更新代码和报告

```powershell
git add .
git commit -m "Update project results and report"
git push
```

原始数据和预测数组受 `.gitignore` 保护，不会随普通提交重复上传。

## 创建新的数据备份

在数据已停止写入的情况下执行：

```powershell
python -m scripts.create_github_backup
```

脚本将 ZIP 写入忽略目录 `.github-backup/assets/`，为每个文件及压缩包生成 SHA-256，并逐文件读取 ZIP 校验内容。若该输出目录非空，脚本会停止，避免覆盖已有备份。将旧附件移到另一个本地目录后可重新运行。

将新附件上传至新的 Release，并将该 Release 对应到当次代码提交。每个 ZIP 的原始内容不超过 800 MiB，满足 GitHub Release 单附件小于 2 GiB 的限制：<https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases>。
