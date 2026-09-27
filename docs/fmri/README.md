# 体积静息态 fMRI：原始 BIDS 到 MNI152 2 mm

`run_fmri_pipeline` 一次处理一个 BIDS BOLD run。它先用 SynthStrip 提取 SBRef（缺失时取 BOLD 中间帧）与 T1 的脑掩膜，再运行运动校正和 FEAT 核心处理、TorchFAST 白质/脑脊液分割、EPI→T1 BBR、T1→MNI 非线性配准、单被试空间 PICA、ICA-AROMA 和可选 WM/CSF/运动回归。高层入口把 SBRef 掩膜作为 `brain_mask` 传给 FEAT；独立 `run_feat_core` 在未提供 `brain_mask` 时则从运动校正后的 EPI 均值提取掩膜，两条入口的默认掩膜输入不同。最后把清理后的 4D BOLD 通过合成变换一次插值到 MNI152 2 mm 网格。GPU 矩阵运算默认启用 TF32，影像以 float32 保存。

这条流程没有 GDC 和 B0 畸变估计。指定 UKB rfMRI ZIP 没有原始 B0 场图/幅度图，也没有 GDC warp；遇到与 BOLD 关联的 BIDS 场图但缺少已估计 warp 时，FEAT 核心会明确报错。清理使用 ICA-AROMA，不能与 UKB 的 FIX 输出逐体素相同。配准仍须和官方同输入结果对照；特别是当前 PyTorch FNIRT 的 T1 强度模型尚未覆盖官方 T1 配置的非线性强度/偏置项。

## 安装、权重和输入

在仓库根目录执行 `conda env create -f environment.yml`，激活 `fnit` 后执行 `fnit-setup-weights --model fmri`，即可准备 SynthStrip 和 SynthMorph deform 权重。选择 `registration_backend="fnirt"` 时只需 `fnit-setup-weights --model synthstrip`。运行时不调用 FSL、FreeSurfer 或其 Python 工作流包。MNI152 T1 2 mm 模板是单独的影像输入；传入模板绝对路径，推荐同时传入同网格的脑掩膜。两份 NIfTI 可从 [FSL 官方 `fsl-data_standard` 固定版数据包](https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/noarch/fsl-data_standard-2208.0-0.tar.bz2)的 `data/standard/` 目录取得，文件名分别为 `MNI152_T1_2mm.nii.gz` 和 `MNI152_T1_2mm_brain_mask.nii.gz`；使用前核对[官方模板说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/utilities/dataset_clitools.html)与[许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)。仅需模板数据，FNIT 运行时不调用 FSL。`mni_template` 必须为 3D、三个体素边长均为 2 mm，且尺寸和仿射与包内 ICA-AROMA 标准掩膜完全一致；不同来源的 2 mm MNI 网格即使体素大小相同，也会在运行前报错。

输入采用原始 BIDS 目录，至少有一份 4D BOLD、对应 JSON 的 `TaskName` 与秒单位 `RepetitionTime`，以及同被试 3D T1w。SBRef 可选；缺失时使用 BOLD 中间帧。多个 run/T1w 候选时明确填写实体或 `t1w_image`。功能不从 UKB ZIP 直接读取。

```text
bids_root/
├── dataset_description.json
└── sub-0001/
    ├── anat/sub-0001_T1w.nii.gz
    └── func/
        ├── sub-0001_task-rest_bold.nii.gz
        ├── sub-0001_task-rest_bold.json
        └── sub-0001_task-rest_sbref.nii.gz  # 可选
```

`locate_bids_inputs` 检查 BIDS 文件名、影像维度、JSON 和 NIfTI 的 TR，并返回 BOLD、T1w、SBRef、关联场图及元数据的路径。选中同一 run 的规则见 [FEAT 核心](feat.md)。

## Python 调用

下例每个路径都须替换为本机绝对路径。权重装入 FNIT 缓存或设定 `FNIT_WEIGHTS` 后，两个 `*_weights` 参数也可填 `None`。

```python
from fnit import run_fmri_pipeline

result = run_fmri_pipeline(
    bids_root="/absolute/path/bids",               # 原始 BIDS 数据集根目录
    output_dir="/absolute/path/sub-0001_fmri",    # 本次运行的输出目录
    subject="0001",                               # BIDS sub 标签，不含 sub-
    mni_template="/absolute/path/MNI152_T1_2mm.nii.gz",  # 3D MNI152 T1 2 mm 参考影像
    session=None,                                  # ses 标签；无 session 填 None
    task="rest",                                  # task 标签，与 BOLD 文件名对应
    run=None,                                     # run 标签；多 run 时填写具体值
    acquisition=None,                             # acq 标签；多候选时填写
    direction=None,                               # dir 标签；如 AP/PA
    reconstruction=None,                          # rec 标签；多候选时填写
    echo=None,                                    # echo 标签；多 echo 时填写
    t1w_image=None,                               # 多张 BIDS T1w 时指定其中一张的绝对路径
    mni_brain_mask="/absolute/path/MNI152_T1_2mm_brain_mask.nii.gz",  # 与模板同网格的 3D mask；None 时对模板做 SynthStrip
    registration_backend="synthmorph",            # T1→MNI 形变：synthmorph 或 fnirt
    synthstrip_weights="/absolute/path/synthstrip.1.pt",  # 脑提取权重；缓存就绪时可为 None
    synthmorph_weights="/absolute/path/synthmorph.deform.3.h5",  # SynthMorph 形变权重；fnirt 分支不用
    ica_n_components=None,                        # PICA 自动定阶；整数表示指定 IC 数
    aroma_mode="nonaggr",                          # ICA-AROMA 非积极回归；也可填 aggr
    regress_wm=False,                             # AROMA 后是否回归白质均值信号
    regress_csf=False,                            # AROMA 后是否回归脑脊液均值信号
    regress_motion=False,                         # AROMA 后是否回归运动参数
    motion_model=24,                               # 运动回归列数：6、12 或 24
    bandpass=None,                                # 可选 (低频Hz, 高频Hz)，如 (0.01, 0.1)
    global_signal=False,                          # 是否同时回归全脑均值
    highpass_cutoff_seconds=100.0,                 # FEAT 高通截止周期，秒
    device="cuda:0",                              # PyTorch 计算设备；None 自动选择
    batch_size=8,                                 # 每批重采样的 BOLD 帧数
    motion_iterations=(35, 25, 15),               # 运动优化三层迭代次数
    ica_max_iter=500,                             # PICA 独立成分迭代上限
    n_splits=1000,                                # AROMA 运动相关特征的随机抽样次数
    random_state=0,                               # ICA 初始化和 AROMA 抽样随机种子
    overwrite=False,                              # 已有最终文件时是否覆盖
)
print(result.clean_mni)                           # X×Y×Z×T，MNI152 2 mm 清理后 BOLD
print(result.mask_mni)                            # 同网格 3D 二值脑掩膜
print(result.report)                              # 参数、尺寸、耗时和显存的 JSON
print(result.timing_seconds)                      # 各阶段墙钟秒数
```

命令行处理同一个 BIDS run；`fnit-fmri run --help` 列出上述可选参数：

```bash
fnit-fmri run --bids-root /absolute/path/bids --subject 0001 \
  --mni-template /absolute/path/MNI152_T1_2mm.nii.gz \
  --mni-brain-mask /absolute/path/MNI152_T1_2mm_brain_mask.nii.gz \
  --registration-backend synthmorph --device cuda:0 \
  --output-dir /absolute/path/sub-0001_fmri
```

## 输出

| 路径 | 结构和含义 |
|---|---|
| `feat/filtered_func_data.nii.gz` | 原生 EPI 网格 X×Y×Z×T float32；运动校正、掩膜、整段强度缩放及高通后的 PICA 输入。FEAT 子文件见 [FEAT 核心](feat.md)。 |
| `masks/epi_synthstrip.nii.gz` | SBRef 或 BOLD 中间帧网格的 SynthStrip 二值脑掩膜；作为高层 FEAT 的显式脑掩膜。 |
| `T1_brain.nii.gz`、`masks/T1_synthstrip.nii.gz` | T1 原网格的 SynthStrip 脑影像及二值掩膜。 |
| `masks/T1_pve_wm.nii.gz`、`masks/T1_pve_csf.nii.gz` | T1 原网格的 TorchFAST 白质、脑脊液部分体积分数；BBR 使用 `masks/T1_wmseg.nii.gz`。 |
| `masks/csf_epi.nii.gz`、`masks/wm_epi.nii.gz` | EPI 网格的高概率 CSF 和 WM 掩膜，仅供相应的可选信号回归。 |
| `reg/example_func2highres.mat` | EPI→T1 的 4×4 FLIRT scaled-mm 矩阵。原理与实测见 [BBR](bbr.md)。 |
| `reg/T1_to_MNI152_2mm_affine.mat`、`reg/MNI152_2mm_to_T1_pull_ras.nii.gz` | T1→MNI 初始仿射与 MNI 网格上指向 T1 的 3 分量 RAS 毫米位移场。见 [非线性配准](normalization.md)。 |
| `aroma/ica/`、`aroma/ica_thresholded_MNI152_2mm.nii.gz` | 原生 EPI 空间的 PICA 成分、mixing 与频谱；另把阈值成分图经 BBR 和 T1→MNI 形变投到 MNI152 2 mm，供 ICA-AROMA 空间分类。见 [PICA](pica.md)。 |
| `aroma/aroma_features.tsv`、`aroma/aroma_noise_components.txt` | 四项 ICA-AROMA 特征，以及从 1 开始编号的噪声成分。见 [AROMA/回归](aroma_confounds.md)。 |
| `aroma/filtered_func_data_aroma.nii.gz` | 原生 EPI 网格的 AROMA 清理后 4D BOLD。启用额外回归时另有 `aroma/filtered_func_data_aroma_confounds.nii.gz`。 |
| `masks/brain_MNI152_2mm.nii.gz` | MNI 网格上 EPI 掩膜与 MNI 模板脑掩膜的交集；uint8。 |
| `filtered_func_data_clean_MNI152_2mm.nii.gz` | 最终 MNI152 2 mm float32 BOLD；脑掩膜外为 0，时间轴继承 BOLD 的 TR。 |
| `pipeline_report.json` | 不含被试编号的影像尺寸、TR、实际 IC 数、回归配置、各阶段耗时以及 PyTorch 峰值已分配/已保留显存；不代表整卡显存。 |

CSF/WM 组织掩膜由 TorchFAST 部分体积分数经 BBR 投到 EPI，再在 EPI 脑掩膜内以 0.8 阈值生成，仅用于可选回归。ICA-AROMA 分类采用[官方 ICA-AROMA](https://github.com/maartenmennes/ICA-AROMA)的三张 MNI152 2 mm CSF、edge、out 掩膜，文件随 `fnit` 安装；阈值 IC 图先从 EPI 空间经 BBR 和 T1→MNI 形变投到相同网格。分类结果仍受本包 PICA 成分和配准差异影响。`regress_wm`、`regress_csf` 和 `regress_motion` 只改变 AROMA 后的结果，不回写 `feat/filtered_func_data.nii.gz`。

## 实测对照和边界

本例真实 UKB BOLD 为 88×88×64×490、TR 0.735 秒。rfMRI ZIP 中的 BOLD/SBRef 是原始影像；该数据位置没有可核实的 scanner raw T1w。本次整链测试将同被试 FreeSurfer `orig/001.mgz` 用 NiBabel 转为 NIfTI 放入私有 BIDS 测试目录；它是真实 T1 衍生影像，不能作为原始 T1w 的验证证据。模板与权重也仅从服务器已有文件读取。私有影像不随仓库发布，公开结果仅有汇总标量。

各函数的同输入精度和耗时分别列于 [FEAT 核心](feat.md)、[BBR](bbr.md)、[PICA](pica.md)、[非线性配准](normalization.md)和[AROMA/回归](aroma_confounds.md)。整链的最终实测结果与官方 no-GDC/no-B0 FEAT 对照见 [fMRI 验证页](../../validation/fmri/README.md)。

同一例 490 帧 BOLD 的默认 SynthMorph 分支已从原始 BIDS 成功运行至 MNI152 2 mm：输出 91×109×91×490、float32、TR 0.735 秒，与模板网格完全相同；全部 442,288,210 个值有限，脑掩膜外最大绝对值为 0。PICA 得到 96 个成分，ICA-AROMA 用官方标准掩膜判定 48 个噪声成分。整链墙钟 520.77 秒，PyTorch 峰值已保留显存 17.58 GB。该例关闭额外 WM/CSF/运动回归，因而只验证这些选项的独立子函数测试；FNIRT 分支有独立的真实 T1→MNI 对照，未在这次 490 帧整链中重跑。以上是[可复核标量摘要](../../validation/fmri/e2e_summary.json)，不表示最终清理影像与 UKB FIX 逐体素相同。
