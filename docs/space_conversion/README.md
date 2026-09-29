# MNI152、fsaverage 与 fsLR 皮层图转换

`convert_space` 转换标准空间中的皮层标量图或整数标签图。MNI152 使用 NIfTI，fsaverage 和 fsLR 使用左右半球各一份 GIFTI。支持双半球单帧图和多帧图。

体积到 fsaverage 使用 Wu 等发布的 RF-ANTs 映射；fsaverage 与 fsLR 之间使用 HCP 2017 年修订的球面对应关系，并按两侧平均顶点面积运行 Workbench `ADAP_BARY_AREA`。fsaverage 回到体积时使用 CBIG 发布的 RF-ANTs 最近顶点映射与皮层掩膜。这个路径与官方映射的体素位置一致。体积转表面会压缩皮层深度，表面回体积只填充皮层掩膜；这些步骤不是数学上的可逆变换。

## 安装与模板文件

主页的 `environment.yml` 安装 PyTorch、Nibabel、SciPy 和 Connectome Workbench。模板文件按官方版本下载，逐个核验 SHA-256。程序运行时不调用 FSL、FreeSurfer 或 MATLAB。

```bash
conda env create -f environment.yml
conda activate fnit
fnit-setup-space-assets --output-dir /data/fnit_space_assets
```

资产目录包含：

```text
/data/fnit_space_assets/
├── rf_ants/                 # CBIG 1490 人 RF-ANTs 正反向映射及 MNI 皮层掩膜
└── hcp_2017/                # HCP 2017 年 fsaverage/fsLR 球面和平均顶点面积
    └── resample_fsaverage/
```

MNI152 输入必须已经在 FSL MNI152 模板坐标中；函数不会从个体 T1 或 BOLD 自动估计配准。它通过 NIfTI 仿射矩阵读取世界坐标，因而可接受同一模板坐标下的 0.5、1、2 mm 或其他体素网格。不同的 MNI 模板变体不能只靠修改像素大小互换。

## Python 用法

```python
from fnit import convert_space

surface_files = convert_space(
    source="/data/atlas_MNI152_1mm.nii.gz",  # 输入：已经配准到 FSL MNI152 的 3D/4D NIfTI
    source_space="MNI152",                     # 输入空间；体积图固定写 MNI152
    target_space="fsLR",                       # 输出空间：fsaverage 或 fsLR
    output_dir="/data/out/fsLR32k",            # 输出目录
    assets_dir="/data/fnit_space_assets",      # 已通过 fnit-setup-space-assets 安装的资产目录
    source_density=None,                       # 体积输入没有表面顶点密度
    target_density="32k",                     # fsLR 输出密度：32k、59k 或 164k
    reference=None,                            # 仅目标为 MNI152 时使用；此处不使用
    device="cuda:0",                           # PyTorch 体积采样设备；也可设为 cpu
    label=False,                                # 连续值图；整数标签图设为 True
    wb_command="wb_command",                   # Conda 环境中的 Connectome Workbench 命令
)
# surface_files == (左侧 GIFTI 路径, 右侧 GIFTI 路径)

volume_file = convert_space(
    source=surface_files,                      # 输入：(左侧 GIFTI, 右侧 GIFTI)，顺序固定
    source_space="fsLR",                      # 输入表面空间
    target_space="MNI152",                    # 输出为 MNI 皮层 NIfTI
    output_dir="/data/out/mni05",             # 输出目录
    assets_dir="/data/fnit_space_assets",     # 同上
    source_density="32k",                     # 输入 fsLR 顶点密度
    target_density=None,                       # MNI 输出用 reference 决定网格
    reference="/data/MNI152_ref_0p5mm.nii.gz", # 输出网格及仿射；可为 0.5/1/2 mm 等
    device="cuda:0",                           # 最近顶点查表和体素网格映射使用 PyTorch
    label=False,                               # 连续值；标签图设为 True
    wb_command="wb_command",                   # fsLR 到 fsaverage 的面积校正重采样
)
# volume_file == /data/out/mni05/space-MNI152_cortex.nii.gz
```

不传 `reference` 时，MNI 输出使用 CBIG 附带的 1 mm、256×256×256 皮层掩膜网格。函数返回的是路径，不是影像数组。

| 参数 | 含义 |
|---|---|
| `source` | MNI152 时为一个 NIfTI 路径；表面空间时为 `(左, 右)` GIFTI 路径。每侧顶点数必须与 `source_density` 一致。 |
| `source_space`、`target_space` | `MNI152`、`fsaverage`、`fsLR` 三选一。 |
| `source_density`、`target_density` | fsaverage：`3k`、`10k`、`41k`、`164k`；fsLR：`32k`、`59k`、`164k`。MNI152 一端设为 `None`。 |
| `output_dir` | 输出位置。表面图为 `L/R.<空间>.<密度>.func.gii`，标签为 `label.gii`；体积为 `space-MNI152_cortex.nii.gz`。 |
| `assets_dir` | 下载并校验过的 HCP/CBIG 文件目录。 |
| `reference` | 目标为 MNI152 时指定 3D 参考 NIfTI。只取前三维的网格与仿射；未指定时使用 CBIG 1 mm 掩膜。 |
| `device` | `cpu` 或可用的 `cuda:<编号>`。体积采样由 PyTorch 计算，GPU 默认 TF32 且使用 float32 数据。表面球面重采样由 Workbench 在 CPU 完成。 |
| `label` | `True` 使用最近邻体积采样及 Workbench `-label-resample`；`False` 使用三线性体积采样及 `-metric-resample`。回体积使用 CBIG 发布的最近顶点表。 |
| `wb_command` | Workbench 可执行文件路径或命令名，默认从环境 `PATH` 搜索。 |

输出体积只有皮层掩膜内赋值。多帧输入保持帧次序：NIfTI 为 `(X,Y,Z,T)`，GIFTI 每帧一个数据数组。标签值超过 float32 精确整数范围（2²⁴）时不适用。MNI 与表面之间的体素分辨率及表面顶点密度是两个独立参数。

命令行示例：

```bash
fnit-space-convert \
  --volume /data/atlas_MNI152_1mm.nii.gz \
  --source-space MNI152 --target-space fsaverage --target-density 41k \
  --assets-dir /data/fnit_space_assets --output-dir /data/out/fsaverage41k \
  --device cpu

fnit-space-convert \
  --left /data/L.fsaverage.41k.func.gii --right /data/R.fsaverage.41k.func.gii \
  --source-space fsaverage --source-density 41k --target-space MNI152 \
  --reference /data/MNI152_ref_2mm.nii.gz \
  --assets-dir /data/fnit_space_assets --output-dir /data/out/mni2mm \
  --device cpu
```

## 原实现的对应命令

CBIG 官方 MATLAB 正向投影使用 `CBIG_RF_projectMNI2fsaverage`，反向投影使用 `CBIG_RF_projectfsaverage2Vol_single`。示意命令如下；`lh_map`、`rh_map`、`reverse_map` 和 `mask` 均来自本功能安装的 CBIG 文件。

```matlab
[lh, rh] = CBIG_RF_projectMNI2fsaverage('/data/atlas_MNI152_1mm.nii.gz', 'linear', lh_map, rh_map);
[mni_cortex, mni_hemi] = CBIG_RF_projectfsaverage2Vol_single(lh, rh, 'nearest', reverse_map, mask);
```

HCP 2017 年的 fsaverage6 到 fsLR32k 左半球连续值命令为：

```bash
wb_command -metric-resample L.fsaverage6.func.gii \
  fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii \
  fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
  ADAP_BARY_AREA L.fsLR.32k.func.gii \
  -area-metrics fsaverage6.L.midthickness_va_avg.41k_fsavg_L.shape.gii \
                fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii
```

右半球把 `L` 换为 `R`。反向交换源、目标球面及面积文件；标签图将 `-metric-resample` 换为 `-label-resample`。

## 真实数据对照

输入为 FSL 随附的实际 MNI152 T1 1 mm 模板及 fsaverage6 平均皮层沟深图；映射使用 CBIG FS5.3 的 1490 人 RF-ANTs 发布文件和 HCP 2017 年球面。计时在本地 WSL CPU 上，包含函数内部文件读写，排除 MATLAB 进程启动。官方 MATLAB 对照调用 CBIG 函数；Workbench 对照单独运行相同官方命令。输出位于测试工作目录，未将原软件数据或代码复制到仓库。

| 转换 | FNIT | 官方实现 | 精度 |
|---|---:|---:|---|
| MNI152 1 mm T1 → fsaverage164k | 0.41 s | CBIG MATLAB 2.55 s | 左/右 MAE 0.00131/0.00109；相关系数均 >0.99999999999 |
| fsaverage164k → MNI152 1 mm | 4.11 s | CBIG MATLAB 20.73 s | 1,029,656 个皮层体素逐体素相同；最大绝对差 0 |
| fsaverage6 沟深 → fsLR32k | 0.77 s | Workbench 0.77 s | 双半球逐顶点相同；最大绝对差 0 |

另用同一真实 MNI T1 图的 0.5 mm 重采样版本测试输入，生成 fsaverage3k；用 0.5 mm 与 2 mm 参考网格测试表面回体积，输出分别为 `(363,435,363)` 和 `(91,109,91)`。0.5 mm 图是原始模板的重采样，不是新的独立扫描。GPU 计时及占用记录见 `validation/space_conversion/README.md`。

## 参考文献和原实现代码库

1. Wu J, et al. Accurate nonlinear mapping between MNI volumetric and FreeSurfer surface coordinate systems. *Human Brain Mapping* 39:3793–3808, 2018. [DOI](https://doi.org/10.1002/hbm.24213)；[CBIG RF-ANTs 原代码与映射](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/registration/Wu2017_RegistrationFusion)。
2. Coalson TS, Van Essen DC, Glasser MF. Resampling between FreeSurfer and HCP fsLR spaces, 2017. [HCP 官方操作说明](https://wiki.humanconnectome.org/docs/assets/Resampling-FreeSurfer-HCP_5_8.pdf)；[HCPpipelines 模板资产](https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases/resample_fsaverage)。
3. Glasser MF, et al. The minimal preprocessing pipelines for the Human Connectome Project. *NeuroImage* 80:105–124, 2013. [DOI](https://doi.org/10.1016/j.neuroimage.2013.04.127)；[Connectome Workbench 原代码](https://github.com/Washington-University/workbench)。
