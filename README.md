# Fudan Neuroimaging Toolkit（FNIT）

FNIT 提供 MRI 处理、连接组构建和群体分析的 Python API 与命令行工具。先选择功能页，按输入要求运行，再查看该功能的真实数据验证结果。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit.git
cd Fudan-Neuroimaging-toolkit
conda env create -f environment.yml
conda activate fnit
python -c "import fnit, torch; print(fnit.__version__, torch.cuda.is_available())"
fnit --help
```

环境定义见 [environment.yml](environment.yml)，安装验证见 [Conda 环境报告](validation/environment/README.md)。需要独立编译程序的功能，按对应功能页完成安装。

## 开始使用

以脑提取为例，先配置外置权重：

```bash
fnit-setup-weights --model synthstrip
fnit synthstrip -i /data/sub-01_T1w.nii.gz \
  -o /data/results/sub-01_brain.nii.gz \
  -m /data/results/sub-01_mask.nii.gz --device cuda:0
```

```python
from fnit import SynthStrip

brain_extraction_model = SynthStrip(device="cuda:0")  # 加载已配置模型
brain_extraction_result = brain_extraction_model("/data/sub-01_T1w.nii.gz")  # 原始结构像
brain_extraction_result.image.save("/data/sub-01_brain.nii.gz")  # 保存脑图
```

Python 示例和参数说明见 [SynthStrip 手册](docs/synthstrip/README.md)；其他功能的输入、输出、Python 和 CLI 示例均在下表链接中。

权重、图谱和模板按[运行资源安装](docs/ASSETS.md)配置，已获许可并收录目录的文件优先从固定 [assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1) 下载。个人 FreeSurfer 许可证由用户提供，许可尚待确认的少量资源继续从原作者来源获取。

## 功能

### 通用与配准

| FNIT 函数 / 类 | 对应原软件包函数 / 命令 | 用途 |
|---|---|---|
| [run_fslmaths](docs/fslmaths/README.md) | FSL `fslmaths` | 对 NIfTI 影像执行算术、滤波、形态学和时间统计。 |
| [fnit.flirt.run_flirt](docs/flirt/README.md) | FSL `flirt` | 估计或应用影像间的刚性与仿射变换。 |
| [TorchFNIRT](docs/fnirt/README.md) | FSL `fnirt` | 估计形变场并输出配准影像和 Jacobian。 |
| [TorchApplyWarp](docs/applywarp/README.md) | FSL `applywarp` | 将已保存变换应用于影像。 |
| [TorchConvertWarp](docs/convertwarp/README.md) | FSL `convertwarp` | 合并仿射与形变并转换场的表示。 |
| [TorchInvWarp](docs/invwarp/README.md) | FSL `invwarp` | 在指定网格上计算位移场的反场。 |
| [convert_space](docs/space_conversion/README.md) | CBIG `CBIG_RF_projectMNI2fsaverage` / `CBIG_RF_projectfsaverage2Vol_single`；Workbench `-metric-resample` | 在 MNI、fsaverage 和 fsLR 之间转换标量或标签脑图。 |
| [run_mmorf](docs/mmorf/README.md) | FSL MMORF `mmorf` | 联合标量影像和扩散张量估计配准。 |

### 结构 MRI

CPU 的功能覆盖、同节点精度与耗时、GPU 保持检查见[本轮 sMRI 报告](validation/smri_cpu/README.md)。SynthSR 默认 CPU 完整浮点输出、FAST 已测 CPU 模式已与官方匹配；SynthMorph joint 的 192/256 固定门已通过，精确 NumBa 采样的六次完整 CPU 输出与验收版逐位相同。同八核默认 256 的 ABBA 中位数 163.094→155.086 秒，本组缩短 4.91%；局部首次 JIT 仍有成本，共享节点时间不代表稳定吞吐。完整左半球 sphere.reg 的坐标、有序面和解码几何与同输入官方相同；正式 C++ 平均器 ABBA 同源码及八核资源门通过，平均步骤快 2.09 倍，完整配准墙钟仍慢 5.27%，不能视为整步提速。WMH 的 GPU crop 已在 20 GB 内保持旧输出，no-crop 仍需更多显存。丘脑/海马亚区、FNIRT 非线性估计、原始 T1 完整 recon-all 仍有差异，见[剩余清单](validation/smri_cpu/REMAINING.md)。

| FNIT 函数 / 类 | 对应原软件包函数 / 命令 | 用途 |
|---|---|---|
| [SynthStrip](docs/synthstrip/README.md) | FreeSurfer `mri_synthstrip` | 提取脑图、脑掩膜和脑边界距离场。 |
| [SynthMorph](docs/synthmorph/README.md) | FreeSurfer `mri_synthmorph register` | 使用外置模型完成刚性、仿射或非线性配准。 |
| [SynthSeg](docs/synthseg/README.md) | FreeSurfer `mri_synthseg` | 分割脑结构并统计体积。 |
| [SynthSegPlus](docs/synthseg_plus/README.md) | FreeSurfer `mri_synthseg --parc` | 分割脑结构和皮层分区。 |
| [SynthSR](docs/synthsr/README.md) | FreeSurfer `mri_synthsr` | 从结构影像合成 1 mm T1w 影像。 |
| [WMHSynthSeg](docs/wmh_synthseg/README.md) | FreeSurfer `mri_WMHsynthseg` | 分割脑结构和白质高信号。 |
| [TorchFAST](docs/fast/README.md) | FSL `fast` | 估计脑组织标签、部分体积分数和偏置场。 |
| [segment_4_subregions](docs/subregions/README.md) | FreeSurfer `segment_subregions` | 从 T1w 分割脑干、丘脑、海马和杏仁核亚区。 |
| [fnit.recon_all.native_free.run_recon_all_python](docs/recon_all/README.md) | FreeSurfer `recon-all` | 从 T1w 生成皮层表面、脑区标签和形态统计。 |
| [FastVBM.run](docs/fast_vbm/README.md) | FSL `fast` + `fsl_reg` + `fslmaths`（组合流程） | 生成标准空间组织图、Jacobian 和调制灰质图。 |
| [FastVBM.run](docs/ukb_vbm/README.md) | UKB `bb_struct_init` / `bb_vbm`（组合流程） | 使用固定标准模板生成单被试灰质 VBM 衍生图。 |

### 功能 MRI

| FNIT 函数 / 类 | 对应原软件包函数 / 命令 | 用途 |
|---|---|---|
| [TorchMCFLIRT](docs/mcflirt/README.md) | FSL `mcflirt` | 估计 BOLD 逐帧运动并保存校正影像和运动参数。 |
| [fMRIVolume_pipeline](docs/fmri/README.md) | fMRIPrep volume；FSL `feat` / `melodic`、ICA-AROMA（组合流程） | 从原始 BIDS BOLD 与 T1w 生成原生和 MNI 空间的预处理与去噪时序。 |
| [fMRISurface_pipeline](docs/fmri/surface.md) | fMRIPrep surface；Workbench `wb_command`（组合流程） | 自动准备 volume 和重建，生成 fsLR32k 时序及 91k CIFTI。 |
| [run_msmsulc / run_msmall](docs/msm/README.md) | newMSM `newmsm`（MSMSulc / MSMAll 配置） | 对皮层球面进行脑沟或多特征配准。 |
| [fnit.melodic.run_melodic_bids / decompose_spatial_ica](docs/melodic/README.md) | FSL `melodic` | 对单被试 BOLD 进行空间 PICA 分解。 |
| [fnit.mshbm.parcellate_volume / parcellate](docs/mshbm/README.md) | CBIG `CBIG_MSHBM_parcellation_single_subject` | 从 BOLD 估计个体网络分区和连接矩阵。 |

### 扩散 MRI 与连接组

| FNIT 函数 / 类 | 对应原软件包函数 / 命令 | 用途 |
|---|---|---|
| [TorchTOPUP](docs/topup/README.md) | FSL `topup` | 从反向相位编码影像估计和校正畸变。 |
| [TorchEDDY](docs/eddy/README.md) | FSL `eddy_cuda` | 校正 DWI 运动与涡流并旋转梯度方向。 |
| [TorchDTIFIT](docs/dtifit/README.md) | FSL `dtifit` | 拟合扩散张量并生成 FA、MD 等参数图。 |
| [TorchAMICONODDI](docs/amico_noddi/README.md) | AMICO `Evaluation.fit`（NODDI）；NODDI Toolbox `batch_fitting_single` | 估计神经突密度、方向离散度和自由水比例。 |
| [TorchBEDPOSTX](docs/bedpostx/README.md) | FSL `bedpostx` / `bedpostx_gpu` | 估计纤维方向及后验不确定性。 |
| [TorchProbtrackX](docs/probtrackx/README.md) | FSL `probtrackx2` / `probtrackx2_gpu` | 执行概率纤维追踪并生成路径密度与连接矩阵。 |
| [DMRIPipeline.run_bids](docs/dmri_pipeline/README.md) | FSL TOPUP / EDDY / DTIFIT / TBSS + AMICO 或 MMORF（组合流程） | 从原始 DWI 生成校正影像和原生、标准空间扩散参数图。 |
| [UKBConnectome_pipeline.run_bids](docs/connectome/README.md) | MRtrix3 `dwi2response` / `dwi2fod` / `mtnormalise` / `tckgen` / `tcksift2` / `tck2connectome`（组合流程） | 从 DWI 与 T1w 构建单模板或两模板结构连接矩阵。 |

### 群体分析

| FNIT 函数 / 类 | 对应原软件包函数 / 命令 | 用途 |
|---|---|---|
| [fit_dictionary_learning / fit_dictionary_learning_streaming](docs/dictionary_learning/README.md) | scikit-learn `MiniBatchDictionaryLearning.fit` | 从数组或分块数据拟合稀疏字典。 |
| [fnit.bigflica.run_bigflica](docs/bigflica/README.md) | BigFLICA `BigFLICA.BigFLICA_cpu.BigFLICA` | 从多模态影像提取群体共享成分。 |
| [run_superbigflica](docs/superbigflica/README.md) | SuperBigFLICA `SupBigFLICA_cpu.SupervisedFLICA` | 联合影像与表型学习共享成分并预测新被试。 |
| [run_bwas](docs/bwas/README.md) | BWAS `BWAS_main.py` | 对逐体素功能连接开展表型关联和连接簇推断。 |

## 运行与结果

各功能页说明 CPU/GPU 支持、运行资源、输出格式与最新真实数据验证结果。

原软件用于独立对照。生产示例采用 FNIT 实现或只读已有结果；现有兼容接口中显式调用官方 FreeSurfer 的选项仅用于独立参考实验。recon-all 的独立 native 程序及 surface 的 Workbench 依赖见相应手册。

真实数据 benchmark 摘要见各功能页，详细记录见 [validation](validation)；图像示例见 [真实脑图索引](docs/figures/README.md)。具体算法仍在持续验证，是否适合某项分析应以对应功能的当前结果为依据。

## 文档与许可

- [统一用户手册模板](docs/README_TEMPLATE.md)
- [运行资源安装与下载例外](docs/ASSETS.md)
- [模型权重配置与许可](docs/WEIGHTS.md)
- [本轮源码与文档核对记录](validation/documentation/readme_manual_20261005/README.md)
- [代码与依赖归属](THIRD_PARTY_NOTICES.md)

外部模型、模板和原始软件分别遵循原作者许可；Git 仓库和安装包不包含模型权重。
