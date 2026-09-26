# Fudan Neuroimaging Toolkit (FNIT)

`fudan-neuroimaging-toolkit` 是独立的脑 MRI 推理包，在 CPU 或 CUDA 上运行。单项推理无需安装 FreeSurfer、FSL、TensorFlow、VoxelMorph 或 Neurite。现有 `fnit-recon-all` 混合流程需要 FreeSurfer 8.2 原生运行包和个人 license；实验性 `fnit-recon-all-python` 使用外置权重与模板，不调用 FreeSurfer 程序。Python 导入名为 `fnit`，单项功能的命令行入口为 `fnit`。

| 模态 | 功能 | 输出与用途 | 用法、原版对照与验证 |
|---|---|---|---|
| sMRI | SynthStrip | 脑图、脑掩膜、有符号距离场 | [SynthStrip 文档](docs/synthstrip/README.md) |
| sMRI | SynthMorph | 刚性、仿射、非线性及联合配准；应用已有变换 | [SynthMorph 文档](docs/synthmorph/README.md) |
| sMRI | WMH-SynthSeg | 脑结构标签、白质高信号及软体积 | [WMH-SynthSeg 文档](docs/wmh_synthseg/README.md) |
| sMRI | 33 类 SynthSeg | T1 结构标签和软体积 | [独立 SynthSeg 文档](docs/synthseg/README.md) |
| sMRI | SynthSR | 从单幅 MRI 或 CT 合成 1 mm T1w | [SynthSR 文档](docs/synthsr/README.md) |
| sMRI | TorchFAST | T1 三组织分割、PVE 与偏置场校正 | [TorchFAST 文档](docs/fast/README.md) |
| sMRI | GPU FAST VBM | 原始 T1w 到 warped GM、Jacobian 和 modulated GM | [FastVBM 文档](docs/fast_vbm/README.md) |
| sMRI、fMRI、dMRI 通用 | PyTorch FLIRT | 12-DOF 仿射；reference-grid image 和 FSL scaled-mm `.mat` | [FLIRT 文档](docs/flirt/README.md) |
| sMRI（当前 GM 配置） | PyTorch FNIRT | GM 非线性配准；intent-2007 coefficients、warped image 和 Jacobian | [FNIRT 文档](docs/fnirt/README.md) |
| sMRI、fMRI、dMRI 通用 | GPU applywarp | 应用 FSL dense warp、FNIRT coefficient、premat 和 postmat | [applywarp 文档](docs/applywarp/README.md) |
| dMRI | PyTorch TOPUP | UKB AP/PA b0 选择、Hz 场估计、畸变校正和 FSL 输出 | [TOPUP 文档](docs/topup/README.md) |
| dMRI | PyTorch EDDY | 逐 volume 运动、二次 EC、TOPUP 场、Jacobian 和离群切片校正 | [EDDY 文档](docs/eddy/README.md) |
| dMRI | PyTorch DTIFIT | FSL 默认 OLS tensor、FA、MD、eigenvalue 和 eigenvector | [DTIFIT 文档](docs/dtifit/README.md) |
| dMRI | PyTorch AMICO-NODDI | NDI、ODI、FWF、方向和拟合误差 | [AMICO-NODDI 文档](docs/amico_noddi/README.md) |
| dMRI | PyTorch BEDPOSTX | 估计体素内纤维方向及不确定性，供概率追踪使用 | [BEDPOSTX 文档](docs/bedpostx/README.md) |
| dMRI | PyTorch ProbtrackX | 种子到体素的概率追踪与脑区间连接矩阵 | [ProbtrackX 文档](docs/probtrackx/README.md) |
| dMRI | UKBConnectome | 已校正 DWI 和 T1w 到四张结构连接矩阵；追踪和 SIFT2 为近似 | [connectome 文档](docs/connectome/README.md) |
| sMRI | GPU recon-all 混合流程 | T1w 到结构分割、皮层表面、顶点指标和脑区统计 | [GPU recon-all 文档](docs/recon_all/README.md) |
| sMRI | 实验性纯 Python recon-all | T1w 到核心分割、双侧皮层表面、顶点指标和脑区统计；近似结果 | [纯 Python 入口](docs/recon_all/README.md#实验性纯-python-整例入口) |
| fMRI | MS-HBM 17 网络 | fsLR32k 静息态时序到个体网络划分，纯 CPU | [MS-HBM 文档](src/fnit/mshbm/README.md) |

混合 recon-all 与实验性纯 Python 入口分别提供独立的单被试命令。其余功能只提供单被试 Python 和单被试命令行接口；需要处理多个病例时，由调用方在包外组织任务与设备。仓库提供 [T1w 样例](examples/README.md)和 [FLAIR 样例](examples/WMH.md)。

混合 recon-all 的 SynthStrip、33 类 SynthSeg、SynthMorph 及三项辅助神经分割使用 PyTorch/CUDA；其他阶段运行于打包的原生 CPU 程序。实验性 `fnit-recon-all-python` 不调用这些程序，其拓扑修复、表面放置和球面配准仍使用近似算法，不能等同于官方结果。

相关 CUDA 路径允许 NVIDIA TF32 matmul 和 cuDNN 内核；模型与影像张量仍保持 float32，本包不会自动改用 float16 或 bfloat16。各验证报告记录实际开关。

FastVBM 的两个分支共用仿射配准、FSL 坐标转换、GPU 重采样、仅非线性
Jacobian 和调制步骤；差别只在非线性形变由 SynthMorph 或 TorchFNIRT 估计。

FLIRT、FNIRT 和 FastVBM 的 0.9 十例验证状态见 [FastVBM 验证页](validation/fast_vbm/README.md)。TOPUP、EDDY、DTIFIT 和 AMICO-NODDI 的真实 UKB 格式 dMRI 对照见上表各子页；输入、输出、原命令对应关系、计时和数值边界均在各自页面。BEDPOSTX 和 ProbtrackX 的验证边界见其功能页。connectome 的固定端点矩阵赋值，以及公开 ds004666 的原始和 TOPUP/EDDY 校正 DWI 同输入 benchmark 均已记录；完整追踪和 SIFT2 结果尚未达到原软件输出一致性，见 [connectome 文档](docs/connectome/README.md)。

FLIRT、FNIRT、applywarp、TOPUP、EDDY、DTIFIT、BEDPOSTX 和 ProbtrackX 的 FSL 派生实现及随包提供的上游源码受 [FSL Software Licence 6.0](licenses/FSL-6.0.txt) 的非商业使用条款约束。各功能的移植范围和验证边界见对应子页面。

## 安装

需要 Python ≥ 3.10。使用 GPU 时，请安装与本机驱动兼容的 CUDA 版 PyTorch。

```bash
git clone https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit.git
cd Fudan-Neuroimaging-toolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Conda GPU 环境

仓库提供独立的 [`environment.yml`](environment.yml)，固定本项目在 gpucw1 验证的 Python 3.11、PyTorch 2.5.1 和 CUDA 11.8 组合。必须从仓库根目录创建环境，因为配置最后以 editable 模式安装当前源码：

```bash
git clone https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit.git
cd Fudan-Neuroimaging-toolkit
conda env create -f environment.yml
conda activate fnit
python -c "import fnit, torch; print(fnit.__version__, torch.__version__, torch.cuda.is_available())"
fnit --help
```

`environment.yml` 同时固定 `pytorch-cuda=11.8` 和 `cuda-version=11.8`，防止 Conda 求解器混入需要更新系统 glibc 的 CUDA 组件。该环境保持模型和影像张量为 float32，并允许 NVIDIA TF32 matmul 与 cuDNN 内核；不会自动启用 float16 或 bfloat16。修改配置后可用 `conda env update -n fnit -f environment.yml --prune` 同步环境。

如果联网登录节点的 glibc 比离线 GPU 节点更新，应按 GPU 节点版本在共享文件系统创建环境。例如 GPU 节点为 glibc 2.17 时，在联网节点运行：

```bash
FNIT_ENV_PREFIX=/path/on/shared-storage/fnit-conda
CONDA_OVERRIDE_GLIBC=2.17 conda env create -p "$FNIT_ENV_PREFIX" -f environment.yml
conda activate "$FNIT_ENV_PREFIX"
```

然后在 GPU 节点激活同一路径。这样 Conda 会按目标节点 ABI 选择二进制包；环境仍由同一份 `environment.yml` 完整构建。

## 下载和部署权重

Git 仓库及 wheel 均不包含模型权重。下面从 FreeSurfer 官方地址下载各功能的默认权重，校验 SHA-256，并记录权重目录；推理时不会自动联网。GPU recon-all 的 33 类 SynthSeg 与 WMH-SynthSeg 是不同模型，分别用 `--model synthseg` 和 `--model wmh-synthseg` 安装。

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint \
  --model wmh-synthseg --model synthsr
```

FastVBM 的 SynthMorph 分支从原始 T1w 开始时需要 `synthstrip.1.pt` 和 `synthmorph.deform.3.h5`；`python tools/setup_weights.py --model fast-vbm` 安装这两个后端的权重超集。TorchFNIRT 分支只需 SynthStrip；已有脑 mask 时该分支无需 checkpoint。GM 模板由用户提供，不由配置脚本下载。TorchFAST、TorchFLIRT、TorchFNIRT、TorchApplyWarp、TorchTOPUP、TorchEDDY、TorchDTIFIT、TorchAMICONODDI、TorchBEDPOSTX 和 TorchProbtrackX 不使用预训练权重。

混合版 GPU recon-all 还需要与固定 FreeSurfer 8.2 流程匹配的本地原生运行包；上述权重命令不提供它。运行包和个人 license 均不随仓库或 wheel 发布。构建、调用、输出和验收见[GPU recon-all 文档](docs/recon_all/README.md)。

GPU recon-all 的 13 个权重和辅助资源文件可单独配置：`python tools/setup_weights.py --model recon-all`。该组复用 SynthStrip 和 SynthMorph 的通用权重，也包含 33 类 SynthSeg、EntoWM、MCA 和静脉窦模型及其查找表；混合流程的原生运行包仍需单独准备；实验性纯 Python 入口另需 `fnit-setup-recon-all-assets` 安装模板。

独立使用 33 类 SynthSeg 时只需 `python tools/setup_weights.py --model synthseg`，随后运行 `fnit synthseg --i T1.nii.gz --o seg.nii.gz --csv-vols seg.vol.csv`，或使用 Python 的 `SynthSeg` 类；详见[独立接口](docs/synthseg/README.md)。

`--all` 下载全部模型变体；`--dest /path/to/weights` 指定本地目录；`--verify-only` 检查已有文件。安装后也可使用 `fnit-setup-weights`。SynthStrip、SynthMorph、WMH-SynthSeg、33 类 SynthSeg 和 SynthSR 可通过 Python 的 `weights=`、CLI 的 `--weights` 或 `FNIT_WEIGHTS` 指定权重；FastVBM 分别使用 `synthstrip_weights=` / `--synthstrip-weights` 和 `synthmorph_weights=` / `--synthmorph-weights`。官方地址、文件大小、SHA-256、许可和离线部署方法见[权重说明](docs/WEIGHTS.md)。

## 项目资料

- 各功能的调用、输出、原版对应和数值比较见上表各子页；[代码结构](docs/ARCHITECTURE.md)说明共享接口。
- [功能 benchmark 索引](validation/README.md)、[FastVBM 验证](validation/fast_vbm/README.md)、[TOPUP 验证](docs/topup/README.md)、[模型及源码来源](docs/provenance.json)。
- [第三方许可与引用](THIRD_PARTY_NOTICES.md)。
