# 旧核团阶段验证接口：`segment_nuclei`

[统一原始 T1 接口](README.md) · [逐区验证](../../validation/subregions/nuclei.md)

普通用户请使用[统一 `segment_subregions`](README.md)，从一张 T1 生成脑干、丘脑、海马和杏仁核的合并标签。本页记录旧的 `segment_nuclei`，供与 FreeSurfer 8.2 的 `norm/aseg/wmparc` 相同输入进行参考验证；已有脚本仍可使用。其历史精度和耗时仅属于旧路径，不能写成新 TorchGEMS recipe 的测试结果。

`segment_nuclei` 接收同一受试者、同一体素网格上的强度图 `norm`、粗结构分割 `aseg` 和白质分区 `wmparc`。它按 FreeSurfer GEMS 横断面流程拟合两个图谱：ThalamicNuclei 和 HippoSF。海马与杏仁核分别处理左右半球。运行时不调用 FreeSurfer 程序，也不需要安装 FreeSurfer；图谱约 30 MB，单独下载。当前只有使用相同 `norm/aseg/wmparc` 的阶段对照，不能当成原始 T1 到最终核团的独立全流程验证。

四面体网格载入、平滑、概率栅格化和变形优化由 Conda 内编译的 C++/ITK 扩展执行。插值及配准的密集体素循环也在 C++ 中；Python 负责组织各阶段，PyTorch 参与小矩阵运算。此旧路径目前主要在 CPU 上运行，不宣称 GPU 加速。仓库已裁去未编译、未链接的 GEMS 文件，包括未使用的原版配准类和其他优化器源码。它复用 FNIT 已有的图谱数据下载器。

## 安装与图谱

在仓库根目录执行：

```bash
# environment-gems-native.yml：Python 3.11、PyTorch、Nibabel、Surfa、ITK、pybind11 和 Conda C++ 编译器
conda env create -f environment-gems-native.yml
# 编译本仓库内的 GEMS、重采样和配准扩展；不链接 FreeSurfer 运行库
conda run -n fnit-gems-native python tools/build_gems_native.py
# --atlas-root：保存 average/ThalamicNuclei/atlas 与 average/HippoSF/atlas 的目录
# --asset-dir：可选的下载缓存；不填时使用已配置的 recon-all 资产目录
conda run -n fnit-gems-native fnit-nuclei setup \
  --atlas-root /absolute/path/nuclei_atlases \
  --asset-dir /absolute/path/asset_cache
```

图谱和 `FreeSurferColorLUT.txt` 下载会逐文件检查大小和 SHA-256。`prepare_nuclei_atlas` 的 Python 形式：

```python
from fnit.gems import prepare_nuclei_atlas

atlas_root = prepare_nuclei_atlas(
    atlas_root="/absolute/path/nuclei_atlases",  # 输出：图谱根目录
    asset_dir="/absolute/path/asset_cache",     # 输入：下载缓存；也可填 None
)
```

返回的 `atlas_root` 为 `Path`。根目录有 `FreeSurferColorLUT.txt`（输出标签配色）；其下两个 `average/<图谱名>/atlas/` 子目录各有 `AtlasMesh.gz`（四面体和标签先验）、`AtlasDump.mgz`（配准目标）及 `compressionLookupTable.txt`（标签号和名称）。

## 分割输入与输出

| 参数 | 输入含义 |
|---|---|
| `norm` | 单受试者三维 T1 强度图路径。阶段对照使用官方 `mri/norm.mgz`；支持 Nibabel 可读取的影像路径。 |
| `aseg` | 同一受试者的粗结构整数标签图路径；用于配准、初始网格拟合及裁剪。必须与 `norm` 的尺寸和仿射一致。 |
| `wmparc` | 白质分区整数标签图路径；海马/杏仁核的强度模型使用它估计参数。同样必须与 `norm` 在同一网格。仅运行丘脑时仍须传入此路径作网格检查。 |
| `atlas_root` | 上一步的图谱根目录。 |
| `output_dir` | 结果根目录；按结构创建子目录。 |
| `structures` | 运行 `thalamus`、`hippo-left`、`hippo-right` 中的哪些结构；默认三者。 |
| `threads` | C++/ITK 线程数，默认 4。 |

```python
from fnit.gems import segment_nuclei

outputs = segment_nuclei(
    norm="/absolute/path/subject/mri/norm.mgz",       # 输入：T1 强度图
    aseg="/absolute/path/subject/mri/aseg.mgz",       # 输入：粗结构标签
    wmparc="/absolute/path/subject/mri/wmparc.mgz",   # 输入：白质分区
    atlas_root="/absolute/path/nuclei_atlases",       # 输入：已校验的两套 GEMS 图谱
    output_dir="/absolute/path/nuclei_result",        # 输出：结果根目录
    structures=("thalamus", "hippo-left", "hippo-right"),  # 输入：需要的结构
    threads=4,                                        # 输入：C++/ITK 线程数
)
left_labels = outputs["hippo-left"]["labels"]  # 输出：左侧原 aseg 网格标签图路径
```

`outputs` 是 `{结构名: {文件类型: Path}}`。`thalamus/` 下生成 `ThalamicNuclei.mgz`（0.5 mm 工作网格）、`ThalamicNuclei.FSvoxelSpace.mgz`（原 `aseg` 网格）和 `ThalamicNuclei.volumes.txt`（各核团软体积，mm³）。`hippo-left/`、`hippo-right/` 下分别生成 `lh`、`rh` 前缀的 `hippoAmygLabels.mgz`、`hippoAmygLabels.FSvoxelSpace.mgz`、`hippoSfVolumes.txt` 和 `amygNucVolumes.txt`；另有 `HBT`、`FS60`、`CA` 合并标签版本，每种均有工作网格和原网格图。返回字典包含基础标签、工作网格标签与两类体积表路径。`.FSvoxelSpace.mgz` 的标签是硬分割；体积表使用原算法的后验软体积，不能直接由硬标签体素数替代。

命令行运行同一单受试者：

```bash
# --norm：T1 强度；--aseg：粗分割；--wmparc：白质分区
# --atlas-root：图谱根目录；--output-dir：结果根目录
# --structure：可重复，限定输出结构；省略时运行全部三项
# --threads：C++/ITK 线程数
fnit-nuclei run --norm /absolute/path/subject/mri/norm.mgz \
  --aseg /absolute/path/subject/mri/aseg.mgz \
  --wmparc /absolute/path/subject/mri/wmparc.mgz \
  --atlas-root /absolute/path/nuclei_atlases \
  --output-dir /absolute/path/nuclei_result \
  --structure thalamus --structure hippo-left --structure hippo-right \
  --threads 4
```

## 官方命令和实测状态

官方同输入命令需事先有包含上述三个文件的 subject：

```bash
# --cross：已有 subject 名；--sd：subject 父目录；--threads：线程数
segment_subregions thalamus --cross fs_sub01 --sd /absolute/path/subjects --threads 4
segment_subregions hippo-amygdala --cross fs_sub01 --sd /absolute/path/subjects --threads 4
```

在 gpucw1 的同一真实 T1 上，官方双侧海马/杏仁核总墙钟 1005.17 秒、丘脑 596.84 秒。FNIT 全新 Conda 环境的左右两项并发运行、丘脑单独运行；各自墙钟、阶段日志、逐区 Dice 和后验软体积见[验证报告](../../validation/subregions/nuclei.md)。前景 Dice 为 0.9819（左）、0.9906（右）和 0.9957（丘脑）；按每区 Dice≥0.95 且硬体积差≤5% 的约定，仅 7/28、24/28、32/44 区达标。后验软体积表为 110/116 条在 5% 内。此前另一 Python 数值环境得到较高但仍未达标的结果，详见报告。当前不能声称与官方逐核团等价，也不能把这组非等价运行的墙钟差称为等价提速。

## 参考文献与原实现代码

- [Iglesias 等，2018，丘脑核团概率图谱](https://pmc.ncbi.nlm.nih.gov/articles/PMC6215335/)。
- [Iglesias 等，2015，海马亚区图谱](https://doi.org/10.1016/j.neuroimage.2015.04.042)。
- [Saygin 等，2017，杏仁核图谱](https://doi.org/10.1016/j.neuroimage.2017.04.046)。
- [FreeSurfer 原实现代码库](https://github.com/freesurfer/freesurfer)；[原流程说明](https://surfer.nmr.mgh.harvard.edu/fswiki/SubregionSegmentation)。
