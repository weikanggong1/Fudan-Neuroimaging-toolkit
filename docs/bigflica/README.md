# BigFLICA：多模态标准空间成分

| 项目 | 内容 |
|---|---|
| 输入 | 每名被试的多模态3D标准空间影像与各模态mask |
| 输出 | 共享course、成分z图、固定模型及新被试course |
| 对应原软件 | 原BigFLICA/FLICA；字典CPU对照sklearn变体 |
| Python / CLI | run_bigflica / fnit-bigflica fit、apply |
| CPU / GPU | CPU/CUDA；直接体素模式要求CUDA |

## 1. 功能简介

`run_bigflica` 从已在标准空间的多模态影像提取共享成分，输出被试course、模态贡献、原模态网格z-stat脑图及可投影新被试的模型。默认压缩流程为逐体素标准化→联合mMIGP→每模态字典学习→FLICA；`use_mmigp_dicl=False`直接拟合体素。

CUDA采用项目PyTorch，CPU保留SciPy/sklearn对照；nibabel/HDF5承担读取、缓存和传输。字典学习复用[独立模块](../dictionary_learning/README.md)，运行时不调用外部神经影像软件。压缩完整默认C20尚未通过真实数据有效秩与脑图验收；遇到有效秩不足会保留诊断并停止输出成功模型。

```mermaid
flowchart LR
    A[标准空间多模态影像与mask] --> B[按被试核对并标准化]
    B --> C{启用压缩}
    C -->|是| D[mMIGP与每模态字典学习]
    C -->|否| E[原始体素矩阵]
    D --> F[FLICA及有效秩检查]
    E --> F
    F --> G[course、z脑图与固定模型]
    G --> H[可选新被试冻结投影]
```

## 2. Python 调用

```python
from pathlib import Path
from fnit.bigflica import run_bigflica, apply_model

subjects_root = Path("/data/cohort_images")  # 用户的被试影像根目录
mask_root = Path("/data/masks")  # 用户准备的标准空间mask
modality_specifications = {
    name: {"image": f"{name}.nii.gz", "mask": str(mask_root / f"{name}.nii.gz")}
    for name in ("vbm", "fa", "md")
}  # 每模态独立网格；同模态被试必须对齐mask
training_subject_ids = Path("/data/training_subjects.txt").read_text().splitlines()  # 真实训练ID
model_directory = run_bigflica(
    subjects_root=subjects_root,  # 输入影像根
    modalities=modality_specifications,  # 输入路径映射
    output_dir="/data/results/bigflica",  # 新的模型/缓存目录
    n_components=3,  # 示例C，用户须按真实规模与秩检查选择
    migp_dim=10,  # 压缩维度R，不超过被试数
    dicl_dim=40,  # 字典维度D，不超过体素样本数
    subjects=training_subject_ids,  # 固定被试顺序
    device="cuda:0",  # 计算设备
    max_gpu_gb=18.0,  # 保守18GiB预算，非实测峰值
)
new_subject_course = apply_model(
    model_dir=model_directory,  # 已通过有效秩检查的固定模型
    subject_dir=subjects_root / "NEW_SUBJECT",  # 未参与训练的完整同模态影像
    output_file="/data/results/new_subject_course.tsv",  # 可选单被试course文件
    device="cuda:0",  # 推理设备
)
```
### 输入数据格式

```text
cohort_images/
├── SUBJECT_A/{vbm,fa,md}.nii.gz
└── SUBJECT_B/{vbm,fa,md}.nii.gz
masks/{vbm,fa,md}.nii.gz
```

- 每幅输入为有限实数 3D NIfTI `[X,Y,Z]`，已经配准到选定标准空间；输入不会在该功能中再配准。
- 每个模态有自己的3D非零mask。同一模态所有被试与其mask须有相同shape、affine、orientation；不同模态可有不同网格。
- `modalities` 的 `image` 是相对于每个被试目录的文件名；`mask` 是绝对路径。模态插入顺序写入模型，推理沿用该顺序。
- `subjects` 是字符串ID序列，保留前导零；每个ID必须具有所有模态。省略时按源码规则搜索目录，实际名单写入模型。
- VBM、FA、MD只是示例模态名。强度和单位由用户的前处理决定，标准化后course没有物理单位。
- 原始影像、ID表、个体预测及用户目标列放在有授权的私有存储；公开报告只引用匿名标量。
配置JSON可直接对应Python映射：

```json
{"modalities": {
  "vbm": {"image": "vbm.nii.gz", "mask": "/data/masks/vbm.nii.gz"},
  "fa": {"image": "fa.nii.gz", "mask": "/data/masks/fa.nii.gz"},
  "md": {"image": "md.nii.gz", "mask": "/data/masks/md.nii.gz"}
}}
```

压缩模式要求C< D且C< R-1；直接体素模式要求C< N-1。示例C3/R10/D40是接口示意，不能自动保证用户数据或C20有效。

CUDA标准化/mMIGP缓存为float32，字典学习为float64；mMIGP采用严格float32协方差分解，关闭该阶段TF32。其他阶段保留各自源码精度。CPU压缩缓存与mMIGP主要为float64。

`fit_mmigp`是内存矩阵辅助入口，返回联合被试基U和各模态投影；`fit_dicl`兼容名见独立字典学习页。完整影像模型必须通过 `run_bigflica` 创建。

**`run_bigflica` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `subjects_root` | 是 | `str / Path` | `—` | 每名被试一个目录的多模态影像根目录。 |
| `modalities` | 是 | `Mapping[str, Mapping[str, str]]` | `—` | 有序模态名→{image:被试目录相对路径, mask:绝对掩膜路径}。 |
| `output_dir` | 是 | `str / Path` | `—` | 本次结果目录；路径按当前工作目录解析。 |
| `n_components` | 是 | `int` | `—` | 共享成分数C，必须与输入被试数、压缩维度及数据有效秩相容。 |
| `migp_dim` | 否 | `int / None` | `None` | 联合mMIGP维度；压缩模式要求，不得超过被试数。 |
| `dicl_dim` | 否 | `int / None` | `None` | 字典原子数≥2，不能超过各模态样本数。 |
| `subjects` | 否 | `Sequence[str] / None` | `None` | 被试目录名序列；None自动识别；顺序会写入模型。 |
| `device` | 否 | `str` | `'auto'` | PyTorch设备；auto自动选择CUDA，否则CPU，可显式指定cpu或cuda:N。 |
| `dicl_max_iter` | 否 | `int` | `1000` | 字典训练最大epoch数；遵循独立字典学习的停止条件。 |
| `flica_max_iter` | 否 | `int` | `1000` | FLICA变分更新次数，必须为正整数。 |
| `top_voxels` | 否 | `int` | `1000` | 每张成分图保留绝对z值最大的体素数。 |
| `random_state` | 否 | `int` | `0` | 初始化和数据划分随机种子。 |
| `max_gpu_gb` | 否 | `float` | `19.0` | 预算单位GiB，默认19.0约20.40十进制GB；实际峰值另记。 与严格20 GB目标有差别；需要严格预算时应留出CUDA上下文余量。 |
| `feature_block` | 否 | `int` | `2048` | 分块大小；DicL为样本行，影像流程为体素列。 |
| `dicl_batch_size` | 否 | `int` | `32` | GPU字典编码每批体素数；改动可能改变在线轨迹。 |
| `dicl_sparse_iterations` | 否 | `int` | `1000` | ADMM及兼容LARS稀疏求解预算。 |
| `use_mmigp_dicl` | 否 | `bool` | `True` | True启用mMIGP+DicL；False直接拟合体素，要求CUDA。 |
| `flica_lambda_dims` | 否 | `str` | `'o'` | o每模态一个噪声精度；R每个原始被试/压缩坐标一个。 |

**`apply_model` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `model_dir` | 是 | `str / Path` | `—` | 训练保存的模型目录。 |
| `subject_dir` | 是 | `str / Path` | `—` | 一名被试的输入目录，须包含上述全部模态。 |
| `ridge` | 否 | `float` | `1e-06` | 新被试冻结载荷岭投影的非负正则系数。 |
| `output_file` | 否 | `str / Path / None` | `None` | 可选返回值保存路径；None只返回内存对象。 |
| `device` | 否 | `str` | `'auto'` | PyTorch设备；auto自动选择CUDA，否则CPU，可显式指定cpu或cuda:N。 |
| `feature_block` | 否 | `int` | `32768` | 分块大小；DicL为样本行，影像流程为体素列。 |

**`fit_mmigp` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `matrices` | 是 | `Mapping[str, np.ndarray]` | `—` | 模态名→二维被试×体素NumPy矩阵，模态共享同一被试顺序。 |
| `migp_dim` | 是 | `int` | `—` | 联合mMIGP维度；压缩模式要求，不得超过被试数。 |
| `device` | 否 | `str` | `'auto'` | PyTorch设备；auto自动选择CUDA，否则CPU，可显式指定cpu或cuda:N。 |

### 输出

```text
bigflica/
├── normalized_f32/{vbm,fa,md}.h5
├── mmigp_R/U.npy
├── mmigp_R/{vbm,fa,md}_projected.h5
├── dicl_<参数和算法版本>/
└── components_C/
    ├── model.json
    ├── flica_reconstruction.json
    ├── subj_course.npy / subj_course.tsv
    ├── modality_contribution.npy
    ├── <模态>_{zstat,loadings,mean,std}.npy
    ├── <模态>_mask.nii.gz
    └── maps/<模态>/component-001_{zstat,top-1000}.nii.gz
```

`flica_lambda_dims="R"`的模型目录为 `components_C_lambda_R/`。CPU缓存为normalized/，直接体素模式没有mMIGP/DicL目录；GPU压缩使用normalized_f32/。

| 输出 | shape、空间与单位 |
|---|---|
| normalized HDF5 | `[Nsubjects,Nmasked_voxels]`；跨被试标准化，均值/标准差保存float64。 |
| U与projected | U为`[Nsubjects,R]`；每模态projected为`[Nmasked_voxels,R]`。 |
| dictionary | `[D,R]`；归一化原子，未代表最终脑图。 |
| subj_course | `[Nsubjects,C]`；TSV附subject，course无物理单位。 |
| modality_contribution | `[Nmodalities,C]`；各模态成分贡献。 |
| zstat及top图 | `[X,Y,Z]` float32；各自mask原shape/affine/orientation/标准空间，mask外零；带符号正态z。 |
| loadings/mean/std/mask | 固定模型所需载荷及训练标准化；新被试不得重新拟合。 |
| flica_reconstruction.json | 请求C、有效秩、奇异值比例与逐模态重建范数比；失败也保存。 |

压缩course使用 `U@H` 回映，脑图在mMIGP PC空间回归，自由度R-C-1；直接体素course在原始被试坐标，自由度N-C-1。两者坐标与噪声模型不同。

有效秩以H奇异值比>1e-6定义；不足C或模态重建范数比<1e-6时报错，不删失败成分。检查通过也不能替代脑图和留出预测验收。

缓存按被试顺序、影像/mask路径、大小、mtime和阶段参数核对。原地改写却保留这些属性时应清理对应缓存；参数或输入不同用新目录。`apply_model`返回`[C]`数组，沿用冻结标准化/载荷，非原FLICA重推断。

### 选择运行路径与分步接口

压缩模式适合先查看低维结构；直接模式省去mMIGP和DicL，仍需全部真实影像输入及原mask。二者的优化输入不同，应分别验证有效秩、course和脑图，不能只看同名成分数。

```python
from fnit.bigflica import run_bigflica

voxel_model_directory = run_bigflica(
    subjects_root="/data/cohort_images",  # 用户真实影像根
    modalities=modality_specifications,  # 上例完整模态映射
    output_dir="/data/results/bigflica_voxel",  # 独立新目录
    n_components=3,  # 按真实数据有效秩选择
    subjects=training_subject_ids,  # 上例训练名单与顺序
    device="cuda:0",  # 直接体素模式要求CUDA
    use_mmigp_dicl=False,  # 不经过压缩和字典阶段
    max_gpu_gb=18.0,  # 保守预算GiB，非实测峰值
)
```

内存辅助 `fit_mmigp` 只产生U和投影，不保存全影像模型，也不携带mask几何：

```python
import numpy as np
from fnit.bigflica import fit_mmigp

masked_modality_matrices = {
    name: np.load(f"/data/matrices/{name}.npy", allow_pickle=False)
    for name in ("vbm", "fa", "md")
}  # 每项被试×该模态mask体素，所有模态行ID及顺序一致
subject_basis, projected_modality_matrices = fit_mmigp(
    matrices=masked_modality_matrices,  # 已检查的有限矩阵
    migp_dim=10,  # 不超过共同被试数
    device="cuda:0",  # CUDA或cpu/auto
)
```

返回U为 `[Nsubjects,R]`，每个投影为 `[Nmasked_voxels,R]`；它供字典训练使用，而不是原被试×体素方向。原始影像affine不能从该矩阵恢复，完整影像流程保留几何并写入各模态mask。

## 3. 命令行调用

```bash
fnit-bigflica fit --subjects-root /data/cohort_images --config /data/modalities.json    --subjects-file /data/training_subjects.txt --output-dir /data/results/bigflica    --n-components 3 --migp-dim 10 --dicl-dim 40 --device cuda:0 --max-gpu-gb 18
fnit-bigflica apply --model-dir /data/results/bigflica/components_3    --subject-dir /data/cohort_images/NEW_SUBJECT    --output-file /data/results/new_subject_course.tsv --device cuda:0
```

| CLI | Python | 含义 |
|---|---|---|
| --subjects-root / --output-dir | subjects_root / output_dir | 影像根与输出 |
| --config | modalities | JSON的modalities映射 |
| --subjects-file | subjects | 每行字符串ID |
| --n-components / --migp-dim / --dicl-dim | n_components / migp_dim / dicl_dim | C/R/D |
| --no-mmigp-dicl | use_mmigp_dicl=False | 直接体素模式，要求CUDA |
| --dicl-max-iter / --dicl-batch-size / --dicl-sparse-iterations | 同名下划线参数 | 字典训练配置 |
| --flica-max-iter / --flica-lambda-dims | flica_max_iter / flica_lambda_dims | FLICA配置 |
| --top-voxels / --random-state | top_voxels / random_state | 图显示体素数和种子 |
| --device / --max-gpu-gb / --feature-block | 同名下划线参数 | 设备与资源 |
| apply --model-dir / --subject-dir / --output-file / --ridge | 同名下划线参数 | 新被试投影 |

其余默认值与上表Python一致；apply CLI必需output-file，Python允许None。直接体素关闭压缩时省略migp/dicl维度，用新的结果目录。

## 4. 原软件调用

原BigFLICA没有等价单行神经影像CLI，接收已展开的被试×体素NumPy矩阵。在独立原作者环境运行：

```python
from BigFLICA.BigFLICA_cpu import BigFLICA

reference_data_files = ["/data/matrices/vbm.npy", "/data/matrices/fa.npy", "/data/matrices/md.npy"]
reference_model = BigFLICA(
    data_loc=reference_data_files,  # 同一被试顺序的模态矩阵
    nlat=3,  # 对应C
    output_dir="/data/reference/bigflica",  # 参考结果目录
    migp_dim=10,  # 对应R
    dicl_dim=40,  # 对应D
    ncore=4,  # 原CPU并行数
)
```

| FNIT | 原实现 |
|---|---|
| n_components / migp_dim / dicl_dim | nlat / migp_dim / dicl_dim |
| modalities影像与mask | data_loc预先展开矩阵 |
| CPU字典 | sklearn变体；原BigFLICA_cpu.py为SPAMS |
| z-stat脑图 | 原sKPCR_regression输出t但命名Z；FNIT完成t→z |
| apply_model | FNIT新增冻结载荷投影，不等同原FLICA后验推断 |

mMIGP特征向量和成分符号可翻转，字典非凸训练会放大上游微小扰动；需先匹配坐标/原子再判断流程误差。不能用相同C或同名文件声称脑图等价。

## 5. 最新精度和运行时间

当前完整压缩C20的验收结论仍为未通过。[匿名30,000人CPU/GPU报告](../../validation/bigflica/bigflica_real30000_cpu_gpu_20260930.json)冻结源码 `6f8dcea1…`，完整三模态mask、C20/R100/D200；独立从原始NIfTI建库，最后有效秩CPU17、GPU13，未生成完整C20脑图或可验收模型。本轮未按140c3739重跑。

| 至科学检查失败的阶段 | CPU | GPU |
|---|---:|---:|
| 读取/标准化 | 5478.73 s | 4358.81 s |
| mMIGP | 3541.38 s | 2176.20 s |
| DicL | 79.85 s | 145.19 s |
| FLICA | 23.60 s | 46.58 s |
| 至失败的总墙钟 | 9182.28 s | 6753.04 s |

8CPU线程、共享H100顺序运行；CPUfloat64标准化/mMIGP，GPUfloat32且该阶段TF32关闭，DicLfloat64。GPU peak allocated/reserved3.66/3.85GiB，CPU峰RSS8.48GiB。它不是成功端到端模型的加速比。

新初始化/逐模态先验与真实1000人控制另见[修复报告](../../validation/bigflica/flica_initialization_prior_fix_real1000_20261001.md)：原始体素保留20成分，压缩同字典仍未保留请求C20，不能据此改写上面的完整流程结果。独立DicL优化没有补测整条FLICA链。

最近公开API的小mask18人C3及独立新被试检查见[接口报告](../../validation/bigflica/three_structural_public_R_smoke_real18.json)与[投影报告](../../validation/bigflica/three_structural_public_R_heldout_apply_real1.json)，属于接口核验。最新成功C20的reference/FNIT/difference脑图未提供；历史小样本图与诊断保留在验证目录。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-01 | fc76fb8f | 字典学习独立并复用成熟函数 | AST/缓存兼容，不代替FLICA全链 |
| 2026-10-01 | 571fc9ea | 同投影字典优化 | 1000人阶段检查，OMP尚有差异 |
| 2026-10-01 | 53ab978b | 修正PCA尺度、模态W先验及秩检查 | 原始体素20，压缩C20仍未通过 |
| 2026-09-30 | 6f8dcea1冻结源码 | 30,000人独立CPU/GPU压缩控制 | CPU17/GPU13；科学验收未通过 |

更早的debug、profiling和长表保留在[旧README归档](../../validation/bigflica/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="流程策略"></a>
<a id="调用独立字典学习模块"></a>
<a id="安装与输入"></a>
<a id="输出与参数"></a>

## 7. 参考文献、原软件和资源

- Gong、Beckmann与Smith，2021，[Phenotype Discovery from Population Brain Imaging](https://www.sciencedirect.com/science/article/pii/S1361841521000967)，Medical Image Analysis。
- 原代码：[BigFLICA](https://github.com/weikanggong/BigFLICA)，`BigFLICA_cpu.py`及FLICA模块；字典参考sklearn见[独立功能页](../dictionary_learning/README.md)。
- FNIT：[pipeline.py](../../src/fnit/bigflica/pipeline.py)、[flica_torch.py](../../src/fnit/bigflica/flica_torch.py)、[完整验证索引](../../validation/bigflica/README.md)。

### 外部资源

本功能不需要预训练权重、图谱或模板，也不自动下载真实输入数据。用户输入与参考软件的许可由各自来源决定。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 无额外模型资源 | — | — | 不适用 | 不适用 | 不适用 |
