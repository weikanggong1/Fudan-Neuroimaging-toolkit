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

模型、图谱和模板的配置见 [资源手册](docs/WEIGHTS.md)。资源优先从固定 [assets-v1 Release](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/tag/assets-v1) 获取；未获明确再分发许可的资源由安装器从原作者来源获取。

## 功能

### 通用与配准

| 功能 | 用途 |
|---|---|
| [影像计算](docs/fslmaths/README.md) | 对 NIfTI 影像执行算术、滤波、形态学和时间统计。 |
| [线性配准](docs/flirt/README.md) | 估计或应用影像间的刚性与仿射变换。 |
| [非线性配准](docs/fnirt/README.md) | 估计形变场并输出配准影像和 Jacobian。 |
| [应用形变](docs/applywarp/README.md) | 将已保存变换应用于影像。 |
| [组合变换](docs/convertwarp/README.md) | 合并仿射与形变并转换场的表示。 |
| [反向形变](docs/invwarp/README.md) | 在指定网格上计算位移场的反场。 |
| [空间转换](docs/space_conversion/README.md) | 在 MNI、fsaverage 和 fsLR 之间转换标量或标签脑图。 |
| [多模态配准](docs/mmorf/README.md) | 联合标量影像和扩散张量估计配准。 |

### 结构 MRI

| 功能 | 用途 |
|---|---|
| [SynthStrip](docs/synthstrip/README.md) | 提取脑图、脑掩膜和脑边界距离场。 |
| [SynthMorph](docs/synthmorph/README.md) | 使用外置模型完成刚性、仿射或非线性配准。 |
| [SynthSeg](docs/synthseg/README.md) | 分割脑结构并统计体积。 |
| [SynthSeg+](docs/synthseg_plus/README.md) | 分割脑结构和皮层分区。 |
| [SynthSR](docs/synthsr/README.md) | 从结构影像合成 1 mm T1w 影像。 |
| [WMH-SynthSeg](docs/wmh_synthseg/README.md) | 分割脑结构和白质高信号。 |
| [组织分割](docs/fast/README.md) | 估计脑组织标签、部分体积分数和偏置场。 |
| [脑亚区](docs/subregions/README.md) | 从 T1w 分割脑干、丘脑、海马和杏仁核亚区。 |
| [皮层重建](docs/recon_all/README.md) | 从 T1w 生成皮层表面、脑区标签和形态统计。 |
| [VBM](docs/fast_vbm/README.md) | 生成标准空间组织图、Jacobian 和调制灰质图。 |
| [UKB 风格 VBM](docs/ukb_vbm/README.md) | 使用固定标准模板生成单被试灰质 VBM 衍生图。 |

### 功能 MRI

| 功能 | 用途 |
|---|---|
| [运动校正](docs/mcflirt/README.md) | 估计 BOLD 逐帧运动并保存校正影像和运动参数。 |
| [Volume pipeline](docs/fmri/README.md) | 从原始 BIDS BOLD 与 T1w 生成原生和 MNI 空间的预处理与去噪时序。 |
| [Surface pipeline](docs/fmri/surface.md) | 自动准备 volume 和重建，生成 fsLR32k 时序及 91k CIFTI。 |
| [MSMSulc / MSMAll](docs/msm/README.md) | 对皮层球面进行脑沟或多特征配准。 |
| [MELODIC](docs/melodic/README.md) | 对单被试 BOLD 进行空间 PICA 分解。 |
| [MS-HBM](docs/mshbm/README.md) | 从 BOLD 估计个体网络分区和连接矩阵。 |

### 扩散 MRI 与连接组

| 功能 | 用途 |
|---|---|
| [TOPUP](docs/topup/README.md) | 从反向相位编码影像估计和校正畸变。 |
| [EDDY](docs/eddy/README.md) | 校正 DWI 运动与涡流并旋转梯度方向。 |
| [DTIFIT](docs/dtifit/README.md) | 拟合扩散张量并生成 FA、MD 等参数图。 |
| [NODDI](docs/amico_noddi/README.md) | 估计神经突密度、方向离散度和自由水比例。 |
| [BEDPOSTX](docs/bedpostx/README.md) | 估计纤维方向及后验不确定性。 |
| [ProbtrackX](docs/probtrackx/README.md) | 执行概率纤维追踪并生成路径密度与连接矩阵。 |
| [dMRI pipeline](docs/dmri_pipeline/README.md) | 从原始 DWI 生成校正影像和原生、标准空间扩散参数图。 |
| [Connectome pipeline](docs/connectome/README.md) | 从 DWI 与 T1w 构建单模板或两模板结构连接矩阵。 |

### 群体分析

| 功能 | 用途 |
|---|---|
| [字典学习](docs/dictionary_learning/README.md) | 从数组或分块数据拟合稀疏字典。 |
| [BigFLICA](docs/bigflica/README.md) | 从多模态影像提取群体共享成分。 |
| [SuperBigFLICA](docs/superbigflica/README.md) | 联合影像与表型学习共享成分并预测新被试。 |
| [BWAS](docs/bwas/README.md) | 对逐体素功能连接开展表型关联和连接簇推断。 |

## 运行与结果

各功能页说明 CPU/GPU 支持、运行资源、输出格式与最新真实数据验证结果。

原软件用于独立对照。生产示例采用 FNIT 实现或只读已有结果；现有兼容接口中显式调用官方 FreeSurfer 的选项仅用于独立参考实验。recon-all 的独立 native 程序及 surface 的 Workbench 依赖见相应手册。

真实数据 benchmark 摘要见各功能页，详细记录见 [validation](validation)；图像示例见 [真实脑图索引](docs/figures/README.md)。具体算法仍在持续验证，是否适合某项分析应以对应功能的当前结果为依据。

## 文档与许可

- [统一用户手册模板](docs/README_TEMPLATE.md)
- [资源下载、文件校验与许可](docs/WEIGHTS.md)
- [本轮源码与文档核对记录](validation/documentation/readme_manual_20261005/README.md)
- [代码与依赖归属](THIRD_PARTY_NOTICES.md)

外部模型、模板和原始软件分别遵循原作者许可；Git 仓库和安装包不包含模型权重。
