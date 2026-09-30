# Fudan Neuroimaging Toolkit (FNIT)

**本工具包仍处于开发阶段，部分算法尚未完成严格验证，目前不宜作为常规数据处理与分析的默认工具。**

FNIT 提供人脑磁共振（MRI）处理和群体分析的 Python 与命令行接口。主要计算由 PyTorch 实现，NIfTI 读写使用 Nibabel。Python 包名为 `fnit`，统一命令行入口为 `fnit`。各功能的运行依赖与安装要求见对应功能页；当前 Connectome 的自动结构重建仍需用户安装并许可的官方 FreeSurfer `recon-all`。

CUDA 路径默认启用 NVIDIA TF32 矩阵乘法和 cuDNN 内核；BWAS 为匹配原版统计结果，使用普通 float32 连接计算和 QR 正交化回归。模型、影像张量与 NIfTI 输出保持 float32，不自动使用 float16 或 bfloat16。

影像处理以单被试 Python API 和命令行接口为主；多被试任务可在包外通过任务调度器、进程池或作业系统分配 CPU/GPU。recon-all、Connectome、BigFLICA 和 BWAS 的处理范围与调用方式见对应功能页。

## 功能

### 一般功能

| 函数名 | 原软件函数名 | 功能 |
|---|---|---|
| [run_fslmaths](docs/fslmaths/README.md) | FSL `fslmaths` | 3D/4D NIfTI 算术、阈值、滤波、形态学与时间统计。 |

### 多模态

| 函数名 | 原软件函数名 | 功能 |
|---|---|---|
| [TorchFLIRT](docs/flirt/README.md) | FSL `flirt` | 线性配准；`applyxfm` 应用已知矩阵或按 qform/sform 在 MNI152 不同分辨率间重采样。 |
| [TorchFNIRT](docs/fnirt/README.md) | FSL `fnirt` | 非线性配准、Jacobian 与系数场。 |
| [TorchApplyWarp](docs/applywarp/README.md) | FSL `applywarp` | 应用形变场及前后仿射矩阵。 |
| [TorchConvertWarp](docs/convertwarp/README.md) | FSL `convertwarp` | 组合线性与非线性变换，转换 MMORF 场。 |
| [TorchInvWarp](docs/invwarp/README.md) | FSL `invwarp` | 在指定网格上计算位移场的反场。 |
| [convert_space](docs/space_conversion/README.md) | CBIG RF-ANTs；HCP Workbench `-metric-resample` | MNI152、fsaverage 与 fsLR 皮层标量或标签图互转，支持多种体素网格与表面密度。 |

### sMRI

| 函数名 | 原软件函数名 | 功能 |
|---|---|---|
| [SynthStrip](docs/synthstrip/README.md) | FreeSurfer `mri_synthstrip` | 脑图、脑掩膜和有符号距离场。 |
| [SynthMorph](docs/synthmorph/README.md) | FreeSurfer `mri_synthmorph` | 刚性、仿射和非线性配准。 |
| [WMHSynthSeg](docs/wmh_synthseg/README.md) | FreeSurfer `mri_WMHsynthseg` | 脑结构与白质高信号标签、软体积。 |
| [SynthSeg](docs/synthseg/README.md) | FreeSurfer `mri_synthseg` | 33 类脑结构标签与软体积。 |
| [SynthSegPlus](docs/synthseg_plus/README.md) | FreeSurfer `mri_synthseg --parc` | 33 类结构与 68 区皮层分区。 |
| [segment_subregions](docs/subregions/README.md) | FreeSurfer `segment_subregions brainstem` | 脑干四亚区标签。 |
| [segment_nuclei](docs/subregions/nuclei.md) | FreeSurfer `segment_subregions thalamus/hippo-amygdala` | 丘脑核团、海马亚区和杏仁核标签与体积；逐区阈值尚未全部通过。 |
| [SynthSR](docs/synthsr/README.md) | FreeSurfer `mri_synthsr` | 合成 1 mm T1w 图像。 |
| [TorchFAST](docs/fast/README.md) | FSL `fast` | 三组织分割、部分体积分数与偏置场。 |
| [FastVBM](docs/fast_vbm/README.md) | FSL `fslvbm` | 从 T1w 生成标准空间灰质、Jacobian 与调制灰质图。 |
| [run_recon_all_python](docs/recon_all/README.md) | FreeSurfer `recon-all` | 从 T1w 生成脑分割、皮层表面、顶点指标与脑区统计。 |

### fMRI

| 函数名 | 原软件函数名 | 功能 |
|---|---|---|
| [parcellate](docs/mshbm/README.md) | CBIG `CBIG_MSHBM_parcellation_single_subject.m` | 生成个体 fsLR32k 17 网络标签。 |
| [fMRIVolume_pipeline](docs/fmri/README.md) | FSL FEAT、ICA-AROMA | 原始 BIDS 单 run 到 BIDS Derivatives 体积 BOLD。 |
| [fMRISurface_pipeline](docs/fmri/surface.md) | fMRIPrep fsLR 重采样、Workbench | 读取已完成的 volume 与 T1 recon-all，写出 fsLR32k GIFTI 和 91k CIFTI。 |
| [fnit.msm.run_msmsulc](docs/msm/README.md) | newMSM MSMSulc | 独立的 HOCR/FastPD 脑沟球面配准。 |
| [fnit.melodic.run_melodic_bids](docs/melodic/README.md) | FSL MELODIC | 独立的 PyTorch 单被试空间 PICA，输入和输出均为 BIDS Derivatives。 |

### dMRI

| 函数名 | 原软件函数名 | 功能 |
|---|---|---|
| [TorchTOPUP](docs/topup/README.md) | FSL `topup` | AP/PA b0 畸变场估计与校正。 |
| [TorchEDDY](docs/eddy/README.md) | FSL `eddy` | DWI 运动及涡流校正、bvec 旋转。 |
| [TorchDTIFIT](docs/dtifit/README.md) | FSL `dtifit` | FA、MD、特征值/向量、S0 与张量拟合。 |
| [TorchAMICONODDI](docs/amico_noddi/README.md) | AMICO `NODDI`；NODDI Toolbox `WatsonSHStickTortIsoV_B0` | AMICO 或经典连续 Watson 拟合；输出 NDI、ODI、FWF、方向与拟合误差。 |
| [TorchMMORF](docs/mmorf/README.md) | FSL `MMORF` | 多标量与扩散张量联合配准，自动估计线性初始化。 |
| [TorchBEDPOSTX](docs/bedpostx/README.md) | FSL `bedpostx` | 纤维方向、体积分数与后验不确定性。 |
| [TorchProbtrackX](docs/probtrackx/README.md) | FSL `probtrackx2` | 概率纤维追踪、路径密度和连接矩阵。 |
| [DMRIPipeline](docs/dmri_pipeline/README.md) | UK Biobank dMRI pipeline（FSL `topup`、`eddy`、`dtifit`、TBSS） | 读取单被试原始 BIDS 或 UKB 格式 DWI；无 T1w 用 TBSS，有 T1w 可选 MMORF；生成九张标准空间扩散参数图。 |
| [UKBConnectome_pipeline](docs/connectome/README.md) | BIDS DWI/T1 结构连接组网 | 从原始 BIDS 自动执行 TOPUP、EDDY、必要时的官方 recon-all，并从一次追踪输出单套或多套 atlas 矩阵；[真实数据对照](validation/connectome/ds004666/README.md)。 |

### 后续分析（Post analysis）

| 函数名 | 原软件函数名 | 功能 |
|---|---|---|
| [run_bigflica / apply_model](docs/bigflica/README.md) | [BigFLICA](https://github.com/weikanggong/BigFLICA) mMIGP、DicL、FLICA | 从每人一目录的多模态标准空间 NIfTI 提取跨模态成分，输出被试 course、各模态成分 z 图和 top-voxel 脑图；CUDA 路径逐被试建库，分块处理 mMIGP 并用增量 LARS 训练 DicL，可选跳过两者直接拟合体素 FLICA，并投影新被试。 |
| [run_bwas / plot_bwas_connectivity](docs/bwas/README.md) | [weikanggong/BWAS](https://github.com/weikanggong/BWAS) | 对多被试 2 mm BIDS volume BOLD 的逐体素连接做表型 GLM、6D 连接簇校正、MA 图和多视角连接可视化。 |

各功能页说明输入、输出、参数与调用示例，并汇总已有的真实数据验证结果、原软件命令、参考文献和原实现链接。统一入口中的子命令用 `fnit <子命令> --help` 查看；fMRI 使用 `fnit-fmri --help`，MS-HBM 使用 `fnit-mshbm --help`，recon-all 使用 `fnit-recon-all --help`。全部独立入口见 [pyproject.toml](pyproject.toml)。

## 安装

推荐从仓库根目录创建独立 Conda 环境：

```bash
git clone https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit.git
cd Fudan-Neuroimaging-toolkit
conda env create -f environment.yml
conda activate fnit
python -c "import fnit, torch; print(fnit.__version__, torch.__version__, torch.cuda.is_available())"
fnit --help
```

`environment.yml` 固定 Python 3.11、PyTorch 2.5.1、CUDA 11.8、Triton 3.1.0、Connectome Workbench 2.1.0、构建 MSMSulc 扩展所需的 C++ 编译器，以及 AMICO 逐体素对照使用的 NumPy 1.26.4 x86_64 wheel。目标 GPU 节点为 glibc 2.17 时，可在共享文件系统上按该 ABI 求解：

```bash
FNIT_ENV_PREFIX=/path/on/shared-storage/fnit-conda
CONDA_OVERRIDE_GLIBC=2.17 conda env create -p "$FNIT_ENV_PREFIX" -f environment.yml
conda activate "$FNIT_ENV_PREFIX"
```

[Conda 环境验证](validation/environment/README.md)和[机器可读报告](validation/environment/report.public.json)记录了依赖解析、目标 ABI、固定 wheel 的下载校验与实际导入结果。具体安装结果以目标机器上的环境创建和导入检查为准。

也可使用 Python 虚拟环境安装；需预先安装支持 C++17 的编译器，以构建表面 MSMSulc 的 FastPD 扩展：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

recon-all 的 Python 依赖和原生编译工具链已列入主页 Conda 环境；创建环境后按 [recon-all 安装说明](docs/recon_all/CONDA_CPP_BUILD.md)编译固定源码程序并获取外置权重、图谱。Connectome 的兼容依赖及其安装边界见功能页。

## 下载和配置权重

Git 仓库与 wheel 不包含模型权重。配置脚本优先从 [FNIT 固定版本 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1)下载；Release 不可用时回退到原作者地址。每个文件均检查大小和 SHA-256，再保存到默认权重目录。推理过程不会自动联网。

下载全部模型：

```bash
python tools/setup_weights.py --all
```

只下载所需模型：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
python tools/setup_weights.py --model wmh-synthseg
python tools/setup_weights.py --model synthseg
python tools/setup_weights.py --model synthseg-plus
python tools/setup_weights.py --model synthsr
python tools/setup_weights.py --model fast-vbm
python tools/setup_weights.py --model fmri
```

安装包后可将 `python tools/setup_weights.py` 替换为 `fnit-setup-weights`。自定义存放位置与离线校验：

```bash
fnit-setup-weights --all --dest /path/to/weights
fnit-setup-weights --all --dest /path/to/weights --verify-only
```

fsLR32k 表面投影的 HCP 公开模板不属于模型权重，同样优先从固定版本 Release 下载并逐文件校验；`--fmriprep` 的 TemplateFlow dseg 仍从原站获取：

```bash
fnit-setup-fmri-surface-assets --output-dir /absolute/path/hcp_surface_assets --fmriprep
```

该命令包含 fsLR32k 投影和 MSMSulc 配准所需的球面、脑沟参考图、ROI、官方 MSMSulc 对照配置，以及生成 91k CIFTI 的 HCP 皮层下分区。被试需提供已生成的 white、pial、sphere、sphere.reg、sulc 和 thickness；这条 fMRIPrep 表面路径不使用 FLAIR、髓鞘图或 wmparc。完整用法见 [fMRI 表面投影](docs/fmri/surface.md)。

脑干亚区另需约 2.3 MB 图谱包，可独立下载和生成 PyTorch 平滑先验：

```bash
# --output-root：生成 brainstem/ 图谱目录的根路径；--device：先验计算设备
fnit-setup-brainstem-atlas --output-root /absolute/path/atlases --device cuda:0
```

输入、输出及 FreeSurfer 同输入对照见[脑干亚区说明](docs/subregions/README.md)。

丘脑、海马与杏仁核使用独立的 Conda 原生扩展环境，需从仓库根目录编译一次：

```bash
conda env create -f environment-gems-native.yml
conda run -n fnit-gems-native python tools/build_gems_native.py
conda run -n fnit-gems-native fnit-nuclei setup --atlas-root /absolute/path/nuclei_atlases
```

运行输入为同网格的 `norm.mgz`、`aseg.mgz` 和 `wmparc.mgz`；用法、实际精度和时间见[核团分割说明](docs/subregions/nuclei.md)。

API 的显式 `weights=`、CLI 的 `--weights`、`FNIT_WEIGHTS` 环境变量、已保存目录和默认缓存按此顺序解析。TorchFAST、TorchFLIRT、TorchFNIRT、TorchApplyWarp、TorchConvertWarp、TorchInvWarp、TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI、TorchMMORF、TorchBEDPOSTX、TorchProbtrackX 与 dMRI pipeline 的 TBSS 分支没有预训练权重；从原始 T1w 启动的流程可能仍需 SynthStrip。文件清单、官方 URL、SHA-256、许可和离线部署见[权重说明](docs/WEIGHTS.md)。

## 验证、样例与许可

[验证索引](validation/README.md)汇总当前源码对应的真实数据精度、运行时间、峰值显存和示意图，并链接机器可读报告。公开样例见 [T1w](examples/README.md)与 [FLAIR](examples/WMH.md)。各功能的验证范围和运行依赖以对应功能页为准。

FSL 派生代码及随包保存的上游源码受 [FSL Software Licence 6.0](licenses/FSL-6.0.txt) 的非商业使用条款约束；其他第三方来源、许可与引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [来源记录](docs/provenance.json)。
