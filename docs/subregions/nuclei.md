# 核团旧接口兼容

[统一脑亚区接口](README.md) · [当前真实数据验证](../../validation/subregions/unified.md) · [历史核团验证](../../validation/subregions/nuclei.md)

从一张原始 T1 运行 `segment_subregions`，默认处理脑干、丘脑、左侧海马/杏仁核、右侧海马/杏仁核四项结构。旧 `segment_nuclei` 保留原参数及嵌套路径字典，内部只调用一次统一函数，再导出兼容文件。两者使用主页 Conda 环境中的 PyTorch/Nibabel 实现。

## 旧 Python 调用保持可用

```python
from fnit.gems import segment_nuclei

legacy_files = segment_nuclei(
    norm="/absolute/path/subject/mri/norm.mgz",       # 输入：旧强度图
    aseg="/absolute/path/subject/mri/aseg.mgz",       # 输入：同网格粗分割
    wmparc="/absolute/path/subject/mri/wmparc.mgz",   # 输入：同网格白质分区
    atlas_root="/absolute/path/subregion_atlases",    # 输入：旧或统一图谱目录
    output_dir="/absolute/path/subject/subregions",   # 输出：统一结果及兼容文件
    device="cuda:0", optimization="fast",           # 输入：新增的可选配置
)
right_labels_path = legacy_files["hippo-right"]["labels"]  # 输出：旧右侧 ID 标签路径
```

| 旧参数或用法 | 当前兼容行为 |
|---|---|
| `norm`、`aseg`、`wmparc`、`atlas_root`、`output_dir` | 原五个位置参数和关键字均保留；映射到统一函数的强度图、粗分割、白质分区及输出目录。 |
| `structures=("thalamus", "hippo-left", "hippo-right")` | 原结构名与三项默认值保留；内部转换为统一名称。 |
| `threads=4` | 设置 PyTorch CPU 线程数，必须至少为 1。 |
| `{结构名: {文件类型: Path}}` | 保留 `labels`、`high_resolution_labels`、丘脑 `volumes`、海马 `hippocampal_volumes` 和杏仁核 `amygdala_volumes`。 |
| `prepare_nuclei_atlas(atlas_root=..., asset_dir=...)` | 原关键字保留，调用已校验的统一图谱准备。 |

兼容标签文件使用 NIfTI，返回路径可继续用 `nibabel.load` 读取；右侧兼容标签减去 10000，恢复旧 HippoSF ID。软体积文本保留两列名称/mm³ 及整结构汇总，使用统一拟合的后验积分体积。统一输出保持左右侧独立 ID。原始 T1 和内存结果请用下面的主接口。

旧 Python 接口省略 `device` 时自动选择可用的 `cuda:0`，否则使用 CPU；显式设备值直接透传。新 `segment_subregions` 仍默认 `cuda:0`。

## 原始 T1 主接口

```python
from fnit import segment_subregions

subregion_result = segment_subregions(
    t1="/absolute/path/sub-01_T1w.nii.gz",          # 输入：原始三维 T1
    atlas_root="/absolute/path/subregion_atlases",  # 输入：统一图谱根目录；None 使用默认缓存
    output_dir="/absolute/path/sub-01_subregions", # 输出：自动保存结果的目录
    structures="all",                            # 输入：默认全部四项结构
    coarse_segmentation=None,                    # 输入：可选的同网格 aseg
    wmparc=None,                                 # 输入：可选的同网格白质分区
    device="cuda:0",                             # 输入：计算设备，也支持 cpu
    optimization="fast",                         # 输入：默认速度配置；另有 balanced
    save_highres=True,                           # 输出：保存四项最终工作网格标签
    save_posteriors=False,                       # 输出：默认省去较大的后验图 I/O
)
native_labels_path = subregion_result.output_files["labels"]  # 输出：原 T1 网格标签路径
```

`output_dir` 自动保存 `subregions_native.nii.gz`（原 T1 网格 `int32` 标签）、`labels.tsv`（标签及所属结构）、`volumes.tsv`（硬体积与后验积分软体积，mm³）、`report.json`（处理报告），以及 `highres/{结构名称}.nii.gz`。脑干和丘脑最终分辨率为 0.5 mm，每侧海马/杏仁核约 0.333 mm。

不传 `output_dir` 时保留内存结果，稍后可调用 `subregion_result.save(output_dir, save_highres=True, save_posteriors=False)`。需要写后验图时显式设 `save_posteriors=True`，生成 `highres/{结构名称}_posterior.nii.gz`；返回路径键为 `posterior/{结构名称}`。

## 图谱与兼容命令行

统一图谱根目录含 `brainstem/`、`thalamus/`、`hippo-amygdala-left/`、`hippo-amygdala-right/`。旧目录缺少统一包时，兼容函数会准备这些包；`prepare_nuclei_atlas(atlas_root=...)` 同样准备此布局。资源逐文件核对大小和 SHA-256，来源与许可见[图谱准备说明](README.md#准备图谱)。

```bash
# 兼容 setup：--atlas-root 转为统一图谱准备的 output_root
fnit-nuclei setup --atlas-root /absolute/path/subregion_atlases --device cpu

# 原始 T1；不传 --structure 时运行全部四项结构
fnit-nuclei run --t1 /absolute/path/sub-01_T1w.nii.gz \
  --atlas-root /absolute/path/subregion_atlases \
  --output-dir /absolute/path/sub-01_subregions --device cuda:0 --optimization fast

# 旧三图输入和结构名仍可用；这里显式限定旧的三项结构
fnit-nuclei run --norm /absolute/path/subject/mri/norm.mgz \
  --aseg /absolute/path/subject/mri/aseg.mgz \
  --wmparc /absolute/path/subject/mri/wmparc.mgz \
  --atlas-root /absolute/path/subregion_atlases \
  --output-dir /absolute/path/subject/subregions \
  --structure thalamus --structure hippo-left --structure hippo-right \
  --device cuda:0 --optimization fast --threads 4
```

`--t1` 与 `--norm` 二选一；`--aseg`、`--wmparc` 可选。重复 `--structure` 选择结构，旧 `hippo-left/right` 自动映射到统一名称。`--threads` 设置 PyTorch CPU 线程数，默认 4 且必须至少为 1。`--t1` 默认使用 `cuda:0`；旧 `--norm` 未指定设备时自动选可用 GPU，否则用 CPU。`--device cpu` 可显式选择 CPU，`setup --asset-dir` 可指定资源缓存。

CLI 保存统一目录的标签、表格、报告与 `highres/` 结果。旧 Python `segment_nuclei` 另外写出 `thalamus/`、`hippo-left/`、`hippo-right/` 兼容文件，并返回对应的嵌套路径字典。

## 历史验证

旧 C++/ITK 核团运行的机器可读精度、耗时和软体积结果保存在[历史验证页](../../validation/subregions/nuclei.md)。它们对应已被替换的计算路径；当前统一流程的真实 T1 结果见[统一验证](../../validation/subregions/unified.md)。官方命令、原实现链接和参考文献见[统一功能页](README.md#官方对照与参考文献)。
