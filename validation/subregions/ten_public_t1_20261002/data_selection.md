# 10 例公开 T1 的固定选择与数据来源

本轮采用 OpenNeuro `ds000114` 的全部 10 名受试者，按 `sub-01` 至 `sub-10` 的编号顺序选择，每人只取 `ses-test` 的 T1。选择在运行分割、计算 Dice 之前固定；未根据图像处理结果更换受试者。

## Snapshot、许可及输入

- Snapshot：`1.0.2`，Git 标签对应提交 `6299834614e9ae7df1e2fc5922545331b5c7f022`。
- 数据 DOI：[10.18112/openneuro.ds000114.v1.0.2](https://doi.org/10.18112/openneuro.ds000114.v1.0.2)。
- [该提交的 dataset_description.json](https://github.com/OpenNeuroDatasets/ds000114/blob/6299834614e9ae7df1e2fc5922545331b5c7f022/dataset_description.json) 明确标记 `CC0`。
- [同一提交的 participants.tsv](https://github.com/OpenNeuroDatasets/ds000114/blob/6299834614e9ae7df1e2fc5922545331b5c7f022/participants.tsv) 包含 `sub-01` 至 `sub-10`；该 snapshot 中没有 `sub-11`。
- 影像直接从 OpenNeuro 的公开 S3 对象下载到服务器共享目录；逐个锁定 S3 version ID，并核对 snapshot 中的 Git-annex `MD5E` 字节大小和 MD5，再记录 SHA-256、体素数组哈希、几何、方向及数据类型。
- 两种实现使用完全相同的公开 snapshot 输入字节，不替换为仓库现有 example 文件。`data_manifest.json` 是逐例输入清单，`data_download_status.json` 保存下载校验状态。

这里的“原始 T1 输入”指公开 snapshot 的 T1，未追加本轮图像处理。[数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3641991/)说明公开 T1 已经通过 `mri_deface` 去脸。因此不能将这些文件称为未去脸扫描。原始影像只保留在服务器；Git 仅包含脚本、数值指标、清单和裁切后的脑区图。

10 例共 92,390,227 字节，已于 2026-10-02 05:56:21 UTC 完成下载校验。随后在 `gpucw1` 从共享目录独立重读全部影像，再次核对文件 MD5/SHA-256、体素数组哈希、shape 与 affine，全部通过；10 个文件哈希和 10 个体素数组哈希均互不相同。结果见 [data_independent_audit.json](data_independent_audit.json)，逐例摘要见 [data_summary.tsv](data_summary.tsv)。

最终 [data_manifest.json](data_manifest.json) 的 SHA-256 为 `9e79b962537ec7ada6d3e817765d0cc41b03cce84e791e9668cecfd4693fffc0`。本机只取回清单等文本，传输后的大小与 SHA-256 逐项校验，见 [data_metadata_fetch.json](data_metadata_fetch.json)。

## 开发样本与独立样本

`sub-01` 在此前单例开发中使用过一个不同的缓存衍生输入，因此标记 `development_seen=True`。其余 9 名受试者标记 `False`。预先计划分别汇总全部 10 例，以及排除 `sub-01` 的 9 例新受试者；不将总体 10 例称为 10 例独立测试样本。

服务器现有 `examples/data/sub-01_T1w.nii.gz`、`sub-02_T1w.nii.gz`、`sub-03_T1w.nii.gz` 与本轮上游 snapshot 的体素并不相同。实测 shape 和 affine 相同，但分别有 3,627,021、4,533,864、3,026,658 个体素变化；原因未进一步归因，不能仅根据压缩大小推断其处理方法。记录见 `cached_input_comparison.json`。本轮重新下载 `sub-01` 并重新运行两种实现，不复用其旧官方分割结果。

## 复现下载

服务器需有 `numpy`、`nibabel`；它们属于项目既有 Conda 环境依赖。脚本只用于 benchmark 数据准备，不属于 FNIT 生产运行时。

```bash
python download_public_t1.py \
  --output /path/to/ten_public_t1_20261002 \
  --subjects sub-01 sub-02 sub-03 sub-04 sub-05 \
             sub-06 sub-07 sub-08 sub-09 sub-10 \
  --workers 3
```

如目标文件已存在，只有其大小和 MD5 与固定 snapshot 完全一致才复用；不覆盖来源不明的旧影像。不同受试者的轴顺序可能不同，调用方须使用各自 NIfTI affine，不以固定数组轴代替物理空间。

首次下载在读取 GitHub 的两份 Git-annex 文本元数据时网络超时。为完成同一固定列表，第二次使用 [public_metadata_cache.json](public_metadata_cache.json)：它只含同一提交的 12 份公开 Git 文本及来源 URL、大小、哈希，不含影像。影像仍由 headcw 直接从 S3 获取。可在上述命令追加 `--metadata-cache public_metadata_cache.json`；[data_download_history.json](data_download_history.json)保留两次状态。最终运行的脚本 SHA-256 为 `8c6954efb9a006ed6ae101bb2a720d94f80ed58683447030fe34267b7e09e864`。

## 参考文献

Gorgolewski KJ, Storkey A, Bastin ME, Whittle IR, Wardlaw JM, Pernet CR. A test-retest fMRI dataset for motor, language and spatial attention functions. *GigaScience*. 2013;2:6. [doi:10.1186/2047-217X-2-6](https://doi.org/10.1186/2047-217X-2-6)。
