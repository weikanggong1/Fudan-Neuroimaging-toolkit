# Fudan Neuroimaging Toolkit (FNIT)

FNIT 提供单被试脑 MRI 处理的 Python 与命令行接口。主要计算由 PyTorch 实现，NIfTI 读写使用 Nibabel；除各功能页明确列出的参考对照外，运行 FNIT 不需要安装 FSL、FreeSurfer、SPM、MRtrix3、AFNI、DIPY 或工作流封装包。Python 包名为 `fnit`，统一命令行入口为 `fnit`。

CUDA 路径默认允许 NVIDIA TF32 matmul 和 cuDNN 内核。模型、影像张量与 NIfTI 输出保持 float32；不会自动使用 float16 或 bfloat16。除 recon-all 与 Connectome 专页另行维护的范围外，本页审计的功能只提供单被试 Python API 和单被试命令行接口。多个病例由调用方在包外通过任务调度器、进程池或作业系统分配 CPU/GPU；这些功能不提供多被试调度层。

## 功能

| 模态 | 功能 | 主要输出 | 用法、输入输出、原软件命令与 benchmark |
|---|---|---|---|
| sMRI | SynthStrip | 脑图、脑掩膜、有符号距离场 | [SynthStrip](docs/synthstrip/README.md) |
| sMRI | SynthMorph | 刚性、仿射、非线性配准结果及变换 | [SynthMorph](docs/synthmorph/README.md) |
| sMRI | WMH-SynthSeg | 脑结构标签、WMH 标签与软体积 | [WMH-SynthSeg](docs/wmh_synthseg/README.md) |
| sMRI | 33 类 SynthSeg | T1w 结构标签与软体积 | [SynthSeg](docs/synthseg/README.md) |
| sMRI | SynthSeg+ | 33 类结构与 68 区体积皮层分区 | [SynthSeg+](docs/synthseg_plus/README.md) |
| sMRI | TorchGEMS 实验接口 | 皮下亚区图谱的概率分割；官方核团精度未验收 | [皮下亚区](docs/subregions/README.md) |
| sMRI | SynthSR | 1 mm T1w 合成图 | [SynthSR](docs/synthsr/README.md) |
| sMRI | TorchFAST | 三组织分割、PVE 与偏置场 | [TorchFAST](docs/fast/README.md) |
| sMRI | FastVBM | 标准空间 GM、Jacobian 与 modulated GM | [FastVBM](docs/fast_vbm/README.md) |
| sMRI、fMRI、dMRI | TorchFLIRT | reference-grid 图像与 FSL scaled-mm `.mat` | [FLIRT](docs/flirt/README.md) |
| sMRI、dMRI | TorchFNIRT | warped image、Jacobian 与 intent-2007 coefficients | [FNIRT](docs/fnirt/README.md) |
| sMRI、fMRI、dMRI | TorchApplyWarp | 应用 warp、premat 与 postmat 后的 reference-grid 图像 | [applywarp](docs/applywarp/README.md) |
| fMRI | BIDS→MNI152 2 mm 体积流程 | 单 run FEAT、FAST/BBR、PICA/AROMA、可选混杂回归与 MNI 影像 | [fMRI](docs/fmri/README.md) |
| fMRI | MNI 2 mm→fsLR32k 表面投影 | 双侧 GIFTI、皮层下 4D 影像与 CIFTI dense timeseries | [fMRI 表面投影](docs/fmri/surface.md) |
| dMRI | TorchTOPUP | Hz 场、校正图与 FSL 兼容输出 | [TOPUP](docs/topup/README.md) |
| dMRI | TorchEDDY | 运动/涡流校正 DWI、旋转 bvec 与参数 | [EDDY](docs/eddy/README.md) |
| dMRI | TorchDTIFIT | FA、MD、L1–L3、V1–V3、S0 与 tensor | [DTIFIT](docs/dtifit/README.md) |
| dMRI | TorchAMICONODDI | NDI、ODI、FWF、方向与拟合误差 | [AMICO-NODDI](docs/amico_noddi/README.md) |
| dMRI | TorchMMORF | T1/DTI 联合配准的 pull warp、Jacobian 与 warped maps | [MMORF](docs/mmorf/README.md) |
| dMRI | 参数图 pipeline | TOPUP/EDDY/DTIFIT/NODDI 后接 TBSS 或 MMORF 的九张标准空间图 | [dMRI pipeline](docs/dmri_pipeline/README.md) |
| dMRI | TorchBEDPOSTX | 纤维方向、体积分数与不确定性 | [BEDPOSTX](docs/bedpostx/README.md) |
| dMRI | TorchProbtrackX | 路径密度与 voxel/ROI 连接矩阵 | [ProbtrackX](docs/probtrackx/README.md) |
| fMRI | MS-HBM 17 网络 | fsLR32k 个体网络标签 | [MS-HBM](docs/mshbm/README.md) |
| dMRI | UKBConnectome | 由校正 DWI 与已完成的 FreeSurfer subject 目录生成 84 区四张结构连接矩阵 | [Connectome](docs/connectome/README.md)；[真实数据对照](validation/connectome/fs_aparc84_subject_dir_20260928.md) |
| sMRI | recon-all | 核心分割、皮层表面、顶点指标与脑区统计 | [recon-all](docs/recon_all/README.md) |

各功能页均给出带参数名和逐项注释的 Python 单被试示例、等价命令行、输入/输出结构、原软件命令、真实数据精度与计时结果。统一入口中的子命令用 `fnit <子命令> --help` 查看；fMRI 使用 `fnit-fmri --help`，MS-HBM 使用 `fnit-mshbm --help`，recon-all 使用 `fnit-recon-all --help`。全部独立入口见 [pyproject.toml](pyproject.toml)。

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

`environment.yml` 固定 Python 3.11、PyTorch 2.5.1、CUDA 11.8、Triton 3.1.0、Connectome Workbench 2.1.0，以及 AMICO 逐体素对照使用的 NumPy 1.26.4 x86_64 wheel。目标 GPU 节点为 glibc 2.17 时，可在共享文件系统上按该 ABI 求解：

```bash
FNIT_ENV_PREFIX=/path/on/shared-storage/fnit-conda
CONDA_OVERRIDE_GLIBC=2.17 conda env create -p "$FNIT_ENV_PREFIX" -f environment.yml
conda activate "$FNIT_ENV_PREFIX"
```

[Conda 环境验证](validation/environment/README.md)和[机器可读报告](validation/environment/report.public.json)记录了依赖 dry-run、目标 ABI、固定 wheel 的下载校验与实际导入结果。dry-run 证明依赖可解析，不代表任意机器已经完成环境创建。

只需基础 Python 安装时也可使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

recon-all 的 Python/C++ 阶段和 Connectome 的兼容依赖不属于基础安装；安装方法与实现边界见各自功能页。

## 下载和配置权重

Git 仓库与 wheel 不包含模型权重。配置脚本从 FreeSurfer 官方地址下载文件、检查大小和 SHA-256，并保存默认权重目录。推理过程不会自动联网。

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

fsLR32k 表面投影的 HCP 公开模板不属于模型权重，单独下载并逐文件校验：

```bash
fnit-setup-fmri-surface-assets --output-dir /absolute/path/hcp_surface_assets --msmall
```

`--msmall` 增加公开的 MSMAll 群体模板和配置；被试者的 white、pial、sphere.reg 和 wmparc 由用户提供。个体 MSMAll 配准仍需另行计算；完整输入与许可见 [fMRI 表面投影](docs/fmri/surface.md)。

API 的显式 `weights=`、CLI 的 `--weights`、`FNIT_WEIGHTS` 环境变量、已保存目录和默认缓存按此顺序解析。TorchFAST、TorchFLIRT、TorchFNIRT、TorchApplyWarp、TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI、TorchMMORF、TorchBEDPOSTX、TorchProbtrackX 与 dMRI pipeline 的 TBSS 分支没有预训练权重；从原始 T1w 启动的流程可能仍需 SynthStrip。文件清单、官方 URL、SHA-256、许可和离线部署见[权重说明](docs/WEIGHTS.md)。

## 验证、样例与许可

[验证索引](validation/README.md)汇总当前源码对应的真实数据精度、运行时间、峰值显存和示意图，并链接机器可读报告。公开样例见 [T1w](examples/README.md)与 [FLAIR](examples/WMH.md)。官方软件只用于生成参考结果，不是 FNIT 候选运行时依赖。

FSL 派生代码及随包保存的上游源码受 [FSL Software Licence 6.0](licenses/FSL-6.0.txt) 的非商业使用条款约束；其他第三方来源、许可与引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)和[来源记录](docs/provenance.json)。
