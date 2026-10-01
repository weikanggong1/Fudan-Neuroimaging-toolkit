# fMRI 体积流程

`fMRIVolume_pipeline` 每次处理一个原始 BIDS BOLD run，输出 BIDS Derivatives。结构支路完成 SynthStrip 脑提取、严格 TorchFAST 组织分割和 T1→MNI152NLin6Asym 2 mm 配准；功能支路完成 EPI 脑提取、本包 TorchMCFLIRT 运动校正与高通。随后执行 BBR、[FNIT MELODIC/PICA](../melodic/README.md)、ICA-AROMA 和可选 WM/CSF/运动回归。最后分别保存个体 EPI 网格与 MNI 网格的清理后 BOLD。运算时不调用 FSL、FreeSurfer 或 fMRIPrep。GPU 默认允许 TF32。最终 MNI BOLD 采用三次 B 样条空间插值；组织图、标签和 surface 皮层投影保留各自的采样方法。

## 流程策略

```mermaid
flowchart TD
    BIDS["原始 BIDS BOLD、JSON、同被试 T1w"] --> SEL["选择 subject、session 与 BOLD run"]
    SEL --> EPI["BOLD 或 SBRef 参考图"]
    SEL --> T1["同被试 T1w"]
    EPI --> MASK["SynthStrip：EPI 脑掩膜"] --> FEAT["TorchMCFLIRT → 强度缩放与高通滤波"]
    T1 --> BRAIN["SynthStrip：T1 脑图与掩膜"] --> FAST["TorchFAST execution=fsl：WM、CSF 部分体积分数"]
    FEAT --> BBR["BBR：EPI 到 T1w 的仿射"]
    FAST --> BBR
    BRAIN --> REG{"T1 到 MNI 配准后端？"}
    MNI["MNI152NLin6Asym 2 mm 模板；掩膜可选"] --> REG
    REG -- SynthMorph --> SM["FNIT SynthMorph 非线性配准"]
    REG -- FNIRT --> FN["TorchFNIRT 非线性配准"]
    SM --> XFM["T1 到 MNI 形变"]
    FN --> XFM
    FEAT --> ICA["FNIT MELODIC/PICA → ICA-AROMA"]
    BBR --> ICA
    XFM --> ICA
    ICA --> CONF{"启用 WM、CSF、运动等回归？"}
    FAST --> CONF
    CONF -- 是 --> REGRESS["回归选定混杂项"] --> CLEAN["清理后的原生 EPI BOLD"]
    CONF -- 否 --> CLEAN
    CLEAN --> NATIVE["保存原生 EPI 网格 BOLD"]
    BRAIN --> AUX["保存 T1 脑图与 BBR 矩阵"]
    BBR --> AUX
    CLEAN --> RESAMPLE["合成 BBR 与 T1→MNI 变换并用三次 B 样条重采样"]
    BBR --> RESAMPLE
    XFM --> RESAMPLE
    RESAMPLE --> OUT["保存 MNI 2 mm BOLD、脑掩膜与 BIDS JSON"]
    classDef default fill:#ffffff,stroke:#000000,color:#000000;
```

## 输入与安装

在仓库根目录运行 `conda env create -f environment.yml`，激活 `fnit`，再运行 `fnit-setup-weights --model fmri`。选 `registration_backend="fnirt"` 时只需要 SynthStrip 权重。`mni_template` 须为与 FNIT ICA-AROMA 掩膜同网格的 MNI152 T1 2 mm NIfTI；`mni_brain_mask` 可选，但需与模板同网格。原始 BIDS 至少包含 `dataset_description.json`、`sub-<label>/func/*_bold.nii.gz` 及含 `TaskName`、`RepetitionTime` 的 JSON、同被试 `anat/*_T1w.nii.gz`。SBRef 可选。多 run、echo 或 T1w 候选需明确选择。

```text
bids/
├── dataset_description.json
└── sub-0001/
    ├── anat/sub-0001_T1w.nii.gz
    └── func/
        ├── sub-0001_task-rest_bold.nii.gz
        └── sub-0001_task-rest_bold.json
```

## Python 调用

```python
from fnit import fMRIVolume_pipeline

result = fMRIVolume_pipeline(
    bids_root="/absolute/path/bids",                   # 原始 BIDS 数据集根目录
    derivatives_root="/absolute/path/bids/derivatives/fnit",  # FNIT BIDS Derivatives 根目录
    subject="0001",                                   # sub 标签，不含 sub-
    session=None,                                      # ses 标签；没有 session 时为 None
    task="rest",                                      # task 标签
    run=None,                                         # run 标签；多 run 时指定
    acquisition=None,                                 # acq 标签；多候选时指定
    direction=None,                                   # dir 标签；多候选时指定
    reconstruction=None,                              # rec 标签；多候选时指定
    echo=None,                                        # echo 标签；多 echo 时指定
    t1w_image=None,                                   # 同被试唯一 T1w；多张时传绝对路径
    mni_template="/absolute/path/MNI152_T1_2mm.nii.gz",  # MNI152 2 mm 参考 NIfTI
    mni_brain_mask=None,                              # 同网格模板掩膜；None 则用 SynthStrip
    registration_backend="synthmorph",                # T1→MNI：synthmorph 或 fnirt
    fnirt_config=None,                                # fnirt 分支配置；默认 T1 预设
    synthstrip_weights=None,                          # None 从 FNIT 权重缓存读取
    synthmorph_weights=None,                          # None 从 FNIT 权重缓存读取
    ica_n_components=None,                            # None 自动定阶；整数固定 IC 数
    aroma_mode="nonaggr",                              # AROMA 回归模式：nonaggr/aggr
    regress_wm=True,                                  # 回归白质均值信号
    regress_csf=True,                                 # 回归脑脊液均值信号
    regress_motion=True,                              # 回归运动参数
    motion_model=24,                                  # 运动项数：6、12、24
    bandpass=None,                                    # 可选频段 (低Hz, 高Hz)
    global_signal=False,                              # 是否回归全脑均值
    highpass_cutoff_seconds=100.0,                    # FEAT 高通截止周期，秒
    device="cuda:0",                                  # PyTorch 设备
    batch_size=8,                                     # BOLD 重采样每批帧数
    motion_iterations=(1, 1, 1),                      # 8/4/4 mm 各一次 Brent 坐标优化
    ica_max_iter=500,                                 # ICA 迭代上限
    n_splits=1000,                                    # AROMA 随机抽样次数
    random_state=0,                                   # ICA/AROMA 随机种子
    overwrite=False,                                  # 是否覆盖同名最终结果
    reuse_anatomical=True,                            # 同一 T1/模板/配置的解剖结果跨 run 复用
    bbr_execution="batched",                         # batched 批量搜索；reference 串行同算法
    fnirt_execution="optimized",                     # optimized GPU 算子；reference 保留原执行方式
)
print(result.clean_native)  # 个体 EPI 空间清理后 4D BOLD
print(result.clean_mni)     # MNI152 2 mm 清理后 4D BOLD
```

命令行：`fnit-fmri volume --bids-root /absolute/path/bids --derivatives-root /absolute/path/bids/derivatives/fnit --subject 0001 --mni-template /absolute/path/MNI152_T1_2mm.nii.gz --regress-wm --regress-csf --regress-motion --device cuda:0`。这条命令选择一个被试的 BOLD run，执行上图完整流程，并额外回归 WM、CSF 和运动项；将清理后原生及 MNI BOLD 写入指定的 derivatives 根目录。完整选项见 `fnit-fmri volume --help`。

运动校正直接调用独立 [`TorchMCFLIRT`](../mcflirt/README.md)：8/4/4 mm 三阶段、原 NCC 目标、相邻帧初值传播，最终 Constant 三次样条采样。`motion_iterations` 现在表示每阶段坐标优化轮数，默认 `(1,1,1)`，对应原 MCFLIRT 默认；原来的 Adam 步数已移除。原始 uint16 BOLD 的校正值先按 NEWIMAGE 向零截断为 int32，再转 float32 进入 FEAT。命令行对应 `--motion-iterations 1 1 1`。

解剖组织分割使用 [`TorchFAST(execution="fsl")`](../fast/README.md)，保留原 FAST 的 radiological 扫描方向、原位邻域更新、连续随机流与偏置场估计。独立 FAST 默认仍是兼容的 `tensor` 路径，pipeline 显式选择 `fsl`；实际配置写入输出 JSON，代码变更也会使旧解剖缓存失效。

## 输出

以 `sub-0001_task-rest` 为例，`derivatives_root/dataset_description.json` 声明 BIDS Derivatives 数据集；下列文件位于 `sub-0001/`。保留源 BIDS 的 ses、task、acq、dir、run、echo 等实体。NIfTI 为 float32；掩膜为 3D 二值图，BOLD 为 X×Y×Z×T。

| 路径（省略 `sub-0001/`） | 含义 |
|---|---|
| `func/sub-0001_task-rest_space-boldref_desc-clean_bold.nii.gz` | 原生 EPI 参考网格的清理后 BOLD，供 surface 皮层投影前重采样到 T1w。 |
| `func/sub-0001_task-rest_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz` | MNI152 2 mm 清理后 BOLD，供皮层下 CIFTI。相邻 JSON 保留源 TaskName，记录 TR、来源、回归配置、耗时及 `FNIT.MNIInterpolation="cubic-bspline-periodic"`。 |
| `func/sub-0001_task-rest_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz` | MNI 网格脑掩膜。 |
| `func/sub-0001_task-rest_from-boldref_to-T1w_mode-image_xfm.txt` | BBR FLIRT 4×4 矩阵；surface 流程用它把 EPI BOLD 采样到 T1w。 |
| `anat/sub-0001_desc-brain_T1w.nii.gz` | SynthStrip 提取的 T1w 脑影像；与源 T1w 同网格。 |

`FMRIVolumeResult` 返回上述五条绝对路径、MNI BOLD 的 JSON 路径及各阶段耗时。已有输出不会自动更新。重跑时用新的 `derivatives_root`，或设置 `overwrite=True`（命令行 `--overwrite`）；新版 MNI BOLD 的 JSON 应包含上述 `MNIInterpolation` 字段。`FNIT.Configuration` 记录本次全部处理参数、FAST 配置、模板与权重位置；`FNIT.Denoising` 记录 ICA-AROMA 模式与完成状态；`FNIT.Source` 记录 Python 源码清单 SHA-256 和 PyTorch、NumPy、nibabel、SciPy 版本。此源码清单不包含权重或模板文件内容，资源哈希由安装器及 benchmark 另行核对。WM/CSF 回归关闭时不生成对应原生掩膜；BBR 所需 T1 白质分割仍会生成。

`reuse_anatomical=True` 默认把 T1 SynthStrip、FAST、模板提取与 T1→MNI 结果保存在当前被试/会话 `anat/.fnit_anatomical/` 的内部缓存；BIDS 的公开输出仍采用上表命名。缓存核验输入、模板/掩膜、权重、配置、执行模式、实现源码及计算环境，每个输出再校验 SHA-256；改变其中任意一项会重新计算。缓存不完整或文件损坏时会重建。BBR、EPI 脑提取和 BOLD 处理仍按 run 运行。`reuse_anatomical=False`（命令行 `--no-anatomical-cache`）强制重新计算解剖步骤；`overwrite` 不负责清空缓存。

计时 JSON 分别记录 `bbr_initial_flirt`、`bbr_refinement`、`bbr_final_resampling`、`t1_to_mni_affine`、`t1_to_mni_nonlinear`、`warp_conversion` 和 `mni_resampling`。命中缓存时，已复用的计算阶段记为 0，校验与等待时间记为 `anatomical_cache_lookup`，报告的 `anatomical_cache.reused` 为 true。`total` 是此次调用至生成重采样结果的实际墙钟，包含初始化、校验、导入及加载；末尾最终 BIDS 文件复制和 JSON 写盘在该计时外。FEAT、PICA、AROMA 文件仍使用临时工作目录。表面处理随后调用 [`fMRISurface_pipeline`](surface.md)。

`bbr_execution="reference"` 和 `fnirt_execution="reference"` 用于逐项回归，选择同一算法的串行成本/原张量算子；它们保留本次坐标、边界与优化流程的修正。CLI 对应 `--bbr-execution`、`--fnirt-execution`。后者只在 `registration_backend="fnirt"` 时可切换。

WM/CSF 概率图投到 EPI 网格时，使用本包 `TorchFLIRT.applyxfm` 的默认降采样预滤波与三线性采样，再以概率≥0.8 交 EPI 脑掩膜。JSON 中 `FNIT.TissueInterpolation="flirt-trilinear-prefilter-float32-coordinates"` 标明这一步；坐标按 float32 逐项计算，其余 CUDA 计算仍默认允许 TF32。

## 全流程 benchmark

2026-10-01，从同一例真实 BOLD/SBRef 和匹配的存档 T1，完整运行冻结源码 `1eb9c417` 的 FNIT FNIRT 分支，与原 SynthStrip、FSL 和作者 ICA-AROMA 按相同步骤比较。双方处理全部 490 帧，100 秒高通、non-aggressive AROMA、WM/CSF 与 Friston-24 联合回归；不使用 GDC/B0、FIX 或空间平滑。存档 T1 经 nibabel 无误差转换，未核对更早的结构处理。

| 完整单例检查 | FNIT | 原软件参照 |
|---|---:|---:|
| 原生 / MNI 网格 | 88×88×64×490 / 91×109×91×490 | 相同 |
| API 含保存 / 原连续链墙钟 | **1318.04 s（21.97 min）** | **2570.47 s（42.84 min）** |
| 独立验证进程墙钟 | 1372.24 s | 2601.72 s |
| CUDA allocated / reserved，十进制 GB | 8.316 / 9.745 | 未单独记录 |
| ICA 成分 / 迭代 / AROMA 噪声成分 | 95 / 41 / 47 | 95 / 40 / 50 |
| EPI WM / CSF 回归 mask Dice | 0.990405 / 0.970970 | 比较基准 |

| 逐体素 490 帧时间 Pearson r | 均值 | 中位数 |
|---|---:|---:|
| 运动校正 BOLD | **0.999578** | 0.999834 |
| pre-ICA BOLD | **0.999356** | 0.999770 |
| 原生最终 clean BOLD | **0.940704** | 0.951029 |
| MNI 最终 clean BOLD | **0.938768** | 0.947436 |

本轮修复了 SynthStrip 的 conform/回采样几何、独立 TorchMCFLIRT 的原 NCC/Brent/初值传播与整数输出、MELODIC 的 PCA/ICA/混合模型，以及 FAST 的顺序扫描和 bias/PVE 数值路径。完整流程已直接调用这些本包子函数。各自的固定输入控制、示例脑图及原命令见 [SynthStrip](../synthstrip/README.md)、[TorchMCFLIRT](../mcflirt/README.md)、[MELODIC](../melodic/README.md)和 [TorchFAST](../fast/README.md)。此前完整版本的 MNI 时间 r 均值约为 0.8536；本表是更新后的完整复测。

固定原 pre-ICA 输入后，ICA 时间序列配对 r 中位数为 0.999999978，95 个分类标签全部匹配；完整链使用各自产生的 pre-ICA 数据，配对 r 中位数为 0.855702，噪声数为 47/50。固定同一 warp、只比较两份清理图时，MNI 时间 r 均值约 0.9449；固定同一清理图、只换 warp 为约 0.9935。原场下本包 sampler 对原 applywarp 的时间 r 均值为 0.999999999921、RMSE 0.002218。余差主要在分解与去噪的输入传播，完整流程尚未数值等价。

共享 H100、8 线程，FNIT 默认 TF32；PCA/ICA 的关键计算采用 float64，没有使用 float16。双方从新目录开始、不复用解剖缓存。FNIT API 扣除捕获中间影像的 27.05 s 额外复制；原连续链还包含阶段检查和 MELODIC HTML 输出。时间边界与共享负载不同，本表不据单次值计算稳定加速比。FEAT 占 973.12 s，包含顺序运动估计、最终采样、缩放、高通和阶段输出，尚未单独分离运动函数时间。

图中为同一 2 mm 模板切面：模板、FNIT temporal SD、原软件 temporal SD、逐体素时间 r。两行 SD 共用色阶，没有追加平滑。

![最新完整490帧FNIT与原软件对照](figures/fmri_matched_native.png)

[完整结果、原命令和复测方法](../../validation/fmri/matched_native.md) · [机器可读比较](../../validation/fmri/matched_pipeline.public.json) · [源码/输入/输出独立核对](../../validation/fmri/matched_fnirt_contract.public.json)。surface 与 MS-HBM 没有在本轮重跑，保留各自的日期和输入范围；[验证索引](../../validation/fmri/README.md)明确两类测量边界。

## 参考文献与原实现

- Smith 等，*Advances in functional and structural MR image analysis and implementation as FSL*，NeuroImage，2004，[DOI](https://doi.org/10.1016/j.neuroimage.2004.07.051)。
- Pruim 等，*ICA-AROMA*，NeuroImage，2015，[DOI](https://doi.org/10.1016/j.neuroimage.2015.02.064)。
- 原实现：[FSL FEAT](https://git.fmrib.ox.ac.uk/fsl/feat5)、[ICA-AROMA](https://github.com/maartenmennes/ICA-AROMA)；[BIDS Derivatives 规范](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)。

处理后的 MNI volume 与 fsLR32k surface 可继续运行 [MS-HBM 17 网络](../mshbm/README.md)。最新[官方 UKB release 下游对照](../../validation/mshbm/processed_release.md)分别报告体积标签、表面标签和网络连接差异。
