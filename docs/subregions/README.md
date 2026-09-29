# TorchGEMS：脑干亚区分割

[返回首页](../../README.md) · [真实数据对照](../../validation/subregions/README.md)

`segment_subregions` 从三维 T1、同网格的粗结构分割和 GEMS 四面体图谱，输出原 T1 网格上的脑干标签。图谱配准、Gaussian 参数估计、网格变形和后验计算由 Python/PyTorch 执行；读写采用 Nibabel。运行时不调用 FreeSurfer 可执行程序。两例真实 T1 的四区 Dice 均超过 0.95，体积差均不超过 5%；逐区结果和显存见验证记录。丘脑核团、海马和杏仁核图谱目前只能解析，尚无分割精度验收。

## 安装图谱

先按[首页安装说明](../../README.md)安装 FNIT。下列命令下载 FreeSurfer 8.2 BrainstemSS 图谱的 `AtlasMesh.gz`（四面体及标签先验）和 `AtlasDump.mgz`（仿射配准掩膜），按 SHA-256 校验；1,291 字节的 `compressionLookupTable.txt`（标签编号和名称）随 FNIT 分发。然后用 PyTorch 生成两层平滑先验。整个图谱包约 2.3 MB，遵循[FreeSurfer 许可](../../licenses/FreeSurfer.txt)。

```bash
# --output-root：最终图谱根目录；程序会创建其 brainstem 子目录
# --asset-dir：原始图谱下载缓存；省略时使用已配置的 recon-all 资产目录
# --device：平滑先验生成设备，可选 cpu 或 cuda:0
fnit-setup-brainstem-atlas --output-root /absolute/path/atlases \
  --asset-dir /absolute/path/asset_cache --device cuda:0
```

Python 中可调用：

```python
from fnit.gems import prepare_brainstem_atlas

atlas_dir = prepare_brainstem_atlas(
    output_root="/absolute/path/atlases",   # 输出根目录；返回 /absolute/path/atlases/brainstem
    asset_dir="/absolute/path/asset_cache", # 校验后的官方图谱缓存目录；也可设为 None
    device="cuda:0",                        # 平滑先验计算设备；无 GPU 时填 "cpu"
)
```

输出结构为 `brainstem/AtlasMesh.gz`、`AtlasDump.mgz`、`compressionLookupTable.txt`、`sigma2.npy`、`sigma1.npy` 和 `config.json`。两个 `.npy` 文件均为 `[4432, 13]` 的 `float32` Gaussian 类先验；`config.json` 固定标签分组、0.5 mm 工作分辨率和三阶段网格拟合参数。每次更换图谱版本需重新生成先验。

## 分割输入和输出

| 参数 | 含义 |
|---|---|
| `t1` | 单幅三维 T1 路径或 Nibabel image。输出保持其 shape、体素仿射和方向。 |
| `atlas_root` | 上一步的根目录，包含 `brainstem/` 子目录。 |
| `structures` | 要运行的子目录名；脑干填 `"brainstem"` 或 `["brainstem"]`。`"all"` 会运行根目录下的全部图谱包。 |
| `coarse_segmentation` | 与 `t1` **shape 和 affine 相同**的粗标签图路径、Nibabel image 或三维 NumPy 数组。脑干编号 16 用于配准；还使用脑干周围的粗标签进行初始网格拟合。省略时运行 FNIT PyTorch SynthSeg。 |
| `synthseg_weights` | 未提供粗标签时，指定 33 类 SynthSeg 权重文件；也可使用已配置的 FNIT 权重目录。有粗标签时不用此参数。 |
| `auto_initialize` | 未提供显式 `atlas_to_native_voxel.npy` 时是否自动拟合图谱仿射；脑干一般填 `True`。 |
| `device` | `"cuda:0"` 或 `"cpu"`；GPU 计算默认允许 TF32。 |
| `em_iterations` | Gaussian EM 最大迭代数。标准脑干图谱包在 `config.json` 中设为 25，优先于此函数参数。 |
| `deform_iterations` | 未设置 `fit_alpha_files` 的通用图谱使用的网格迭代数；标准脑干包由 `fit_stage_iterations` 控制。 |

```python
from fnit.gems import segment_subregions

result = segment_subregions(
    t1="/absolute/path/sub-01_T1w.nii.gz",          # 输入：三维 T1
    atlas_root="/absolute/path/atlases",            # 输入：含 brainstem/ 的图谱根目录
    structures="brainstem",                         # 输入：只运行脑干图谱
    coarse_segmentation="/absolute/path/aseg.nii.gz", # 输入：与 T1 同网格的粗标签；也可填 None
    synthseg_weights=None,                           # 粗标签为 None 时使用已配置的 SynthSeg 权重
    auto_initialize=True,                            # 根据脑干掩膜拟合图谱到 T1 的仿射
    device="cuda:0",                                # PyTorch 设备
    em_iterations=25,                                # Gaussian EM 次数；标准图谱配置已固定为 25
    deform_iterations=0,                             # 标准图谱由三阶段配置决定变形步数
)
result.labels.save("/absolute/path/sub-01_brainstem.nii.gz")
scp_mask = result.mask("SCP")
```

`result.labels` 是 `int32` NIfTI 标签图，体素编号为 173 Midbrain、174 Pons、175 Medulla、178 SCP，其他体素为 0。`result.label_table` 是 `{编号: 名称}`；`result.mask(label)` 返回同网格布尔掩膜；`result.confidence` 是已选标签的最大后验张量；`result.initialization["brainstem"]` 记录仿射、裁剪范围、阶段耗时和峰值 GPU 显存；`result.structure_results["brainstem"]` 保留工作网格裁剪区的先验、后验、网格顶点、Gaussian 参数、目标函数历史及最小 Jacobian。后三项用于诊断，不在原 T1 网格上。

单病例命令行：

```bash
# --i：输入 T1；--o：输出原 T1 网格标签 NIfTI
# --atlas-root：图谱根目录；--structure：图谱子目录名
# --coarse-segmentation：同网格粗标签；省略时加 --synthseg-weights 指定权重
# --device：PyTorch 设备；--em-iterations：Gaussian EM 次数
fnit subregions --i /absolute/path/sub-01_T1w.nii.gz \
  --o /absolute/path/sub-01_brainstem.nii.gz \
  --atlas-root /absolute/path/atlases --structure brainstem \
  --coarse-segmentation /absolute/path/aseg.nii.gz \
  --device cuda:0 --em-iterations 25
```

标准脑干配置采用粗标签掩膜仿射配准、40 步初始网格拟合、0.5 mm 工作图像、13 个强度类及三层平滑先验（每层 20 步、每 10 步更新一次 Gaussian 参数）。最终输出经最大连通域筛选，再以最近邻插值回到原 T1 网格。`config.json` 中的 `include_label_ids` 决定保留的亚区，不应人为限制在粗标签 16 内：官方的中脑和 SCP 有部分体素位于 16 之外。

## 官方命令和数值对照

```bash
# --cross：已有官方 subject 名；--sd：subjects 根目录；--threads：官方 CPU 线程数
segment_subregions brainstem --cross fs_sub01 \
  --sd /absolute/path/official_subjects --threads 4
```

官方命令从该 subject 的 `mri/norm.mgz` 和 `mri/aseg.mgz` 读入。严格阶段对照应把同一对文件交给 FNIT；对比脚本会按仿射将官方 `brainstemSsLabels.FSvoxelSpace.mgz` 最近邻重采样到 FNIT 网格。原始 T1 加 FNIT SynthSeg 的完整独立输入属于另一项对照，不应与官方 `norm/aseg` 阶段结果混为同输入比较。逐区 Dice、硬体积差、阶段计时、峰值显存和限制见[验证记录](../../validation/subregions/README.md)。

## Reference

- 参考文献：Iglesias et al., *Bayesian segmentation of brainstem structures in MRI*, NeuroImage (2015), [doi:10.1016/j.neuroimage.2015.02.065](https://doi.org/10.1016/j.neuroimage.2015.02.065)。
- 原实现代码库：[FreeSurfer 主代码库](https://github.com/freesurfer/freesurfer)。
