# BigFLICA：多模态标准空间成分

`run_bigflica` 读取“每名被试一个目录”的 3D NIfTI。每个模态指定相对于被试目录的影像路径和自己的 3D 掩膜。输出被试成分 course、每模态每成分的原网格 z-stat NIfTI、绝对 z 值最高的若干体素的阈值图与 PNG，以及可投影新被试的固定模型。模态间可有不同网格；同一模态的影像必须与其掩膜形状和仿射一致。输入必须已经在所需标准空间；函数不做配准。

**字典学习已独立：**压缩流程调用 `fnit.dictionary_learning`，CPU/GPU 共用独立模块中的对应入口。单独拟合字典的输入、参数、求解方式和真实数据报告见[字典学习功能页](../dictionary_learning/README.md)；此页保留完整 BigFLICA 流程的接口和验收范围。

**验收范围：**字典阶段已有保存的同一 float64 R500 投影对照；公开入口的 float32 投影、默认逐体素标准化和从原始 NIfTI 开始的冷启动全链尚未验收。压缩流程的有效 C20、最终成分脑图和新被试模型也尚未通过。原始体素试验曾保留20个有效成分，旧版30,000人压缩对照仅保留 CPU17/GPU13个；这些不同设置的历史结果与剩余工作见[验证索引](../../validation/bigflica/README.md)。

默认 `use_mmigp_dicl=True`：逐体素跨被试标准化 → 联合 mMIGP → 每模态 Lasso-LARS 字典学习 → FLICA。CUDA 路径用 PyTorch 执行协方差、特征分解、稀疏编码、字典更新、FLICA、空间回归和 t→z 转换；nibabel/HDF5 负责 CPU 文件读写和分块传输。为复现原 FLICA 的自由度拟合，GPU 特征分解后每模态将少量特征值送给 SciPy 做一次标量优化；这一步属于 CPU 计算。`use_mmigp_dicl=False` 时，标准化后直接把体素送入 FLICA，不建立 mMIGP 或 DicL 模型。此模式保留体素信息，但每轮须读取全部模态矩阵，适合较小训练集；大样本建议开启预处理。`device="cpu"` 保留原 notebook 的 sklearn DicL 对照路径。

**FLICA 已修复项：**已修正自由能、精度和迭代记录、PCA 的 MATLAB SVD 尺度、逐模态 W 先验、零影像判定及归一化统计保存精度，并在脑图回归前检查含截距的设计矩阵。主要修复与真实数据控制见[初始化和先验报告](../../validation/bigflica/flica_initialization_prior_fix_real1000_20261001.md)。

CUDA 压缩路径先逐被试读取，把 float32 标准化矩阵作为 HDF5 分块存盘；后续阶段不在内存中装入完整的“被试 × 体素”模态矩阵。默认 `max_gpu_gb=19`，按 20 GiB 显存目标预留空间。float32 协方差本身需 `N × N × 4` 字节，计算还要为临时数组留空间；超过配置预算时报错。磁盘需求不受内存预算限制，例如 37,182 人 × 100 万掩膜体素的 float32 标准化缓存约 138.5 GiB/**每模态**。大于 2,048 人时，mMIGP 根据目标秩和显存预算选择完整 GPU 特征分解或随机子空间；随机路径与直接体素 FLICA 的自由度近似仍需单独核验，不能当作逐点一致。

## 流程策略

```mermaid
flowchart TD
    IN["每名被试的多模态标准空间 NIfTI"] --> CHECK["核对被试、模态掩膜与各模态网格"]
    MASK["每模态独立的 3D 掩膜"] --> CHECK
    CHECK --> CACHE["逐被试读取与逐体素标准化；分块写 HDF5"]
    CACHE --> MODE{"启用 mMIGP 与 DicL？"}
    MODE -- 是 --> MIGP["联合 mMIGP：低维被试子空间"] --> DICL["每模态 DicL 稀疏字典"] --> FLICA["多模态 FLICA 成分拟合"]
    MODE -- 否 --> RAW["直接使用标准化体素矩阵"] --> FLICA
    FLICA --> COURSE["被试成分 course 与模态贡献"]
    FLICA --> MAP["空间回归与 t→z 转换"]
    MAP --> OUT["各模态 z-stat NIfTI、top-voxel 图与 PNG"]
    FLICA --> MODEL["保存固定模型及标准化参数"]
    MODEL --> APPLY["可选 apply_model：投影新的单名被试"]
    NEW["未参与训练的新被试同模态影像"] --> APPLY
    APPLY --> NEWCOURSE["新被试成分 course"]
    classDef default fill:#ffffff,stroke:#000000,color:#000000;
```

## 调用独立字典学习模块

开启 `use_mmigp_dicl` 时，mMIGP 输出每模态 `[掩膜体素, migp_dim]` 矩阵。CPU 流程调用 `fnit.dictionary_learning.fit_dicl`，CUDA 流程调用 `fnit.dictionary_learning.fit_dicl_gpu_streaming`；两者返回 `[dicl_dim, migp_dim]` 字典，再交给 FLICA。这里保留 `dicl_dim`、`dicl_max_iter`、`dicl_batch_size`、`dicl_sparse_iterations` 等完整流程参数，内部使用同一份成熟字典实现。

GPU 字典缓存版本仍为 `rsvd5rowgraph`。本次提取模块未改变拟合数学和参数，已有通过输入签名检查的标准化、mMIGP 和字典缓存可复用。CLI 和 Python 的 `dicl_sparse_iterations` 默认 1000；显式设置 120 的调用仍按 120 运行。

独立字典阶段的精度、内存、CPU/GPU 运算分工和历史耗时见[功能页](../dictionary_learning/README.md)与[独立验证索引](../../validation/dictionary_learning/README.md)。BigFLICA 的验收继续包括 mMIGP 输入、FLICA 有效成分、最终脑图和新被试投影；字典阶段通过某项指标不能替代这些检查。

## 安装与输入

在仓库根目录运行 `conda env create -f environment.yml`，再运行 `conda activate fnit`。环境包含 PyTorch、nibabel、h5py、scikit-learn、SciPy、matplotlib；运行时不调用 FSL、FreeSurfer、SPM、MRtrix3 或 AFNI。

```text
ukb_multimodal_new/
  SUBJECT_A/
    VBM_2mm.nii.gz
    dti_FA_2mm_mmorf.nii.gz
    dti_MD_2mm_mmorf.nii.gz
    zstat1s.nii.gz
  SUBJECT_B/
    ...
```

此目录中的任务图为 `zstat1s.nii.gz`，不是 tstat。`modalities.json` 的四个掩膜均由使用者准备；路径须为绝对路径：

```json
{
  "modalities": {
    "vbm": {"image": "VBM_2mm.nii.gz", "mask": "/absolute/masks/vbm.nii.gz"},
    "fa": {"image": "dti_FA_2mm_mmorf.nii.gz", "mask": "/absolute/masks/fa.nii.gz"},
    "md": {"image": "dti_MD_2mm_mmorf.nii.gz", "mask": "/absolute/masks/md.nii.gz"},
    "zstat1": {"image": "zstat1s.nii.gz", "mask": "/absolute/masks/zstat1.nii.gz"}
  }
}
```

CUDA 压缩模式和新被试投影。以下 `C=3/R=10/D=40` 参数来自 [18 名真实被试的小掩膜核验](../../validation/bigflica/README.md)；示例 `subjects.txt` 应列出该规模的训练被试。更换被试或掩膜后须检查 `flica_reconstruction.json`，重新选择能保留所需成分的维度。

```bash
# subjects.txt 每行一个训练被试目录名；本示例使用 18 名训练被试。
fnit-bigflica fit \
  --subjects-root /absolute/path/ukb_multimodal_new \
  --config /absolute/path/modalities.json \
  --subjects-file /absolute/path/subjects.txt \
  --output-dir /absolute/path/bigflica_output \
  --n-components 3 --migp-dim 10 --dicl-dim 40 \
  --dicl-max-iter 20 --dicl-batch-size 32 \
  --dicl-sparse-iterations 1000 --flica-max-iter 100 \
  --top-voxels 300 --random-state 0 \
  --device cuda:0 --max-gpu-gb 19 --feature-block 2048

# 使用训练时冻结的均值、标准差和空间载荷，不重新拟合。
fnit-bigflica apply \
  --model-dir /absolute/path/bigflica_output/components_3 \
  --subject-dir /absolute/path/ukb_multimodal_new/NEW_SUBJECT \
  --output-file /absolute/path/new_subject_course.tsv \
  --ridge 1e-6 --device cuda:0 --feature-block 32768
```

直接体素模式省略 mMIGP/DicL 维度，另设输出目录以保留两套模型。支持标量 `o` 和逐被试 `R` 噪声精度，输出前执行与压缩流程相同的有效秩和模态重建检查。默认预处理为逐体素标准化，初始化已采用修正后的 MATLAB SVD 尺度；历史整体 RMS 数据控制使用另一种预处理，须与公开默认设置分别验收：

```bash
fnit-bigflica fit \
  --subjects-root /absolute/path/ukb_multimodal_new \
  --config /absolute/path/modalities.json \
  --subjects-file /absolute/path/subjects.txt \
  --output-dir /absolute/path/bigflica_raw_output \
  --n-components 3 --no-mmigp-dicl \
  --flica-max-iter 1000 --device cuda:0 --max-gpu-gb 19
```

Python API 的变量名与文件用途对应：

```python
from pathlib import Path
from fnit.bigflica import apply_model, run_bigflica

subjects_root = Path("/absolute/path/ukb_multimodal_new")
mask_root = Path("/absolute/masks")  # 四个事先准备好的标准空间掩膜
modalities = {
    "vbm": {"image": "VBM_2mm.nii.gz", "mask": str(mask_root / "vbm.nii.gz")},
    "fa": {"image": "dti_FA_2mm_mmorf.nii.gz", "mask": str(mask_root / "fa.nii.gz")},
    "md": {"image": "dti_MD_2mm_mmorf.nii.gz", "mask": str(mask_root / "md.nii.gz")},
    "zstat1": {"image": "zstat1s.nii.gz", "mask": str(mask_root / "zstat1.nii.gz")},
}
training_subject_ids = [
    subject_id.strip()
    for subject_id in Path("/absolute/path/subjects.txt").read_text().splitlines()
    if subject_id.strip()
]  # 与掩膜网格一致、模态齐全的训练被试；顺序会写入模型
model_dir = run_bigflica(
    subjects_root=subjects_root,
    modalities=modalities,
    output_dir=Path("/absolute/path/bigflica_output"),
    n_components=3,
    migp_dim=10,
    dicl_dim=40,
    subjects=training_subject_ids,
    use_mmigp_dicl=True,  # 改为 False 时省略 migp_dim、dicl_dim
    device="cuda:0",
    max_gpu_gb=19,
    feature_block=2048,
    dicl_batch_size=32,
    dicl_sparse_iterations=1000,
    dicl_max_iter=20,
    flica_max_iter=100,
    top_voxels=300,
    random_state=0,
)
new_subject_dir = subjects_root / "NEW_SUBJECT"  # 不得在训练列表中
new_subject_course = apply_model(
    model_dir, new_subject_dir, ridge=1e-6, device="cuda:0", feature_block=32768
)
```

只用结构模态时，应另设输出目录以免复用四模态模型。以下 `18` 人、三个小核验掩膜、`C=3/R=10/D=40` 示例用于说明 API 与 CLI；完整掩膜的性能和精度测试已扩展到30,000人。此前2,050人公开调用的C3输出已核对，C20/R100/D200在两个样本规模均未通过有效秩验收，见[三模态基准](../../validation/bigflica/README.md)。

```python
structural_modalities = {name: modalities[name] for name in ("vbm", "fa", "md")}
structural_model_dir = run_bigflica(
    subjects_root=subjects_root,
    modalities=structural_modalities,
    output_dir=Path("/absolute/path/bigflica_structural_output"),  # 与四模态输出分开
    n_components=3,
    migp_dim=10,
    dicl_dim=40,
    subjects=training_subject_ids,  # 此核验设置使用 18 名真实被试
    device="cuda:0",
    max_gpu_gb=19,
    dicl_max_iter=20,
    flica_max_iter=100,
    flica_lambda_dims="R",  # 每个 mMIGP 维度单独估计噪声精度
    top_voxels=100,
    random_state=0,
)
```

## 输出与参数

```text
bigflica_output/
  normalized_f32/vbm.h5, fa.h5, ...   # CUDA 压缩模式：[被试, 掩膜体素]；均值/标准差为 float64
  mmigp_10/U.npy                      # [被试, mMIGP 维度]
  mmigp_10/vbm_projected.h5, ...      # [掩膜体素, mMIGP 维度]
  mmigp_10/eigen_diagnostics.json     # 阶段耗时、收敛次数、整体/逐特征对残差
  dicl_10_40_20_0_32_1000_rsvd5rowgraph_cuda/ # 每模态 dictionary.npy 与 manifest.json
  components_3/
    model.json                         # 参数、被试顺序、输入签名、阶段耗时
    flica_reconstruction.json          # 重建比、有效秩、H 奇异值比例及成分范数；失败时也保留
    subj_course.npy, subj_course.tsv  # [被试, 成分]；TSV 首列 subject
    modality_contribution.npy         # [模态, 成分]
    vbm_zstat.npy, fa_zstat.npy, ...   # [掩膜体素, 成分]
    vbm_loadings.npy, ...             # 新被试的固定空间载荷
    vbm_mask.nii.gz, vbm_mean.npy, vbm_std.npy, ...
    maps/vbm/component-001_zstat.nii.gz
    maps/vbm/component-001_top-300.nii.gz
    maps/vbm/component-001_top-300.png
```

直接体素模式只有 `normalized/` 和 `components_*/`，不生成 mMIGP/DicL 目录。CPU 压缩模式使用 `normalized/` float64 缓存；被试数不超过 2,048 时保留 SciPy 精确 mMIGP，超过 2,048 时逐被试写入 HDF5、分块计算协方差和投影。大样本随机子空间最多迭代 120 次，float64 的整体及每对特征残差须低于 `1e-8`；float32 须低于 `1e-6` 且至少迭代 90 次。失败时不写成功缓存清单。CPU DicL 用 sklearn `.fit`，每次只读取一个压缩后的 `P × migp_dim` 模态；单模态投影超过 4 GiB 时会报错。原始 `N × P` 模态始终保存在 HDF5。CPU 大样本 mMIGP 缓存版本为 `cpu-stream-v2-adaptive-float64`；GPU 随机子空间为 `cuda-stream-v6-adaptive-float32`，高目标秩完整特征分解为 `cuda-stream-v7-highrank-exact-float32`，旧版本缓存不会复用。

压缩及直接体素模式的 `flica_reconstruction.json` 记录各模态与整体重建范数比、H 的奇异值相对最大奇异值的比例、逐成分范数、有效秩及请求成分数。有效秩以奇异值比例大于 `1e-6` 的个数定义。若出现非有限值、有效秩不足，或某模态重建范数比低于 `1e-6`，会保留诊断并停止输出该模型。该检查不替代收敛、逐模态残差、脑图及留出被试验收。逐被试噪声模式的输出目录为 `components_C_lambda_R/`，标量模式仍为 `components_C/`；已有模型的输入签名或模态不一致时会拒绝覆盖，应使用新输出目录。缓存以被试顺序、影像和掩膜的路径/大小/修改时间及阶段参数核验；若原地改写文件却保留这些属性，须删除相应缓存后重跑。

`model.json` 新增 `flica_algorithm_version`、`normalization_version`、`course_coordinates`、`brainmap_space` 和 `brainmap_df`。当前版本为 `matlab-pca-per-modality-W-v2`、`voxel-zscore-v3-stats64-any-nonzero`；旧算法模型不会被新拟合覆盖，已有固定模型仍可用 `apply_model`。标准化均值和标准差保存为 float64，供训练输出和新被试投影共用。脑图回归要求成分与截距合并后满列秩，否则停止生成统计图。

压缩模式保留原 BigFLICA 的 `U@H` course 回映和 PC 空间回归：`course_coordinates=legacy_spectral_PC_zscore`，`brainmap_space=mMIGP_PC`，自由度为 `migp_dim-C-1`。直接体素模式使用原始被试坐标，自由度为 `N-C-1`。两种模型的坐标、噪声和统计自由度不同，不能把成分数相同视为脑图等价。

| 参数 | 含义 |
|---|---|
| `subjects_root` / `--subjects-root` | 每名被试一个目录的父路径。 |
| `modalities` / `--config` | 有序模态映射；各项为相对影像路径 `image` 和绝对掩膜路径 `mask`。 |
| `output_dir` / `--output-dir` | 模型、成分图和可复用 HDF5 缓存目录。 |
| `subjects` / `--subjects-file` | 训练被试目录名及顺序；省略时自动搜索。 |
| `n_components` / `--n-components` | 成分数；压缩模式须小于 `dicl_dim` 和 `migp_dim - 1`，直接模式须小于被试数减一。 |
| `use_mmigp_dicl` / `--no-mmigp-dicl` | 默认开启两阶段压缩；CLI 标志将其关闭并直接拟合体素。 |
| `migp_dim` / `--migp-dim` | 压缩模式的联合 mMIGP 维度，不得超过被试数；直接模式可省略。 |
| `dicl_dim` / `--dicl-dim` | 压缩模式每模态的字典原子数；直接模式可省略。 |
| `dicl_max_iter` / `--dicl-max-iter` | DicL 最大 epoch 数，默认 1000；沿用 sklearn 提前停止规则。 |
| `dicl_batch_size` / `--dicl-batch-size` | GPU LARS 每批体素数，默认 32，与指定 notebook 一致。改动后结果可能变化。 |
| `dicl_sparse_iterations` / `--dicl-sparse-iterations` | GPU 稀疏求解预算，默认1000；用于ADMM迭代和回退LARS的路径事件数。预算耗尽且回退仍未求解时会报错，不返回截断编码。 |
| `flica_max_iter` / `--flica-max-iter` | FLICA 变分更新次数，默认 1000，必须为正整数。CPU/GPU 均执行指定次数；旧版多更新一次的问题已修复。 |
| `flica_lambda_dims` / `--flica-lambda-dims` | 噪声精度维度：默认 `o` 为每模态一个值，与指定 notebook 一致。`R` 在直接体素模式中为每模态、每个原始被试一个值，在压缩模式中为每模态、每个 mMIGP 坐标一个值。更改此项会改变模型，须分别验收成分稳定性和重建。 |
| `top_voxels` / `--top-voxels` | 各成分按绝对 z 值保留最高的体素数，默认 1000。 |
| `random_state` / `--random-state` | DicL 随机种子，默认 0。 |
| `device` / `--device` | `auto` 优先 CUDA；可显式指定 `cuda:0` 或 `cpu`。直接体素模式要求 CUDA。 |
| `max_gpu_gb` / `--max-gpu-gb` | mMIGP/直接 FLICA 的显存预算，默认 19 GiB；被试数超过 2,048 的 CPU mMIGP 将同一数值用作协方差的内存预算。 |
| `feature_block` / `--feature-block` | 逐批处理的体素列数，拟合默认 2048，投影默认 32768。 |
| `model_dir` / `--model-dir` | 已拟合的 `components_*` 目录。 |
| `subject_dir` / `--subject-dir` | 不在训练列表的新被试目录；各影像须与冻结掩膜同网格。 |
| `ridge` / `--ridge` | 新被试多模态岭回归的非负正则化系数，默认 `1e-6`。 |
| `output_file` / `--output-file` | 新被试 course TSV；Python 中可省略，此时只返回数组。 |

原软件接受已展开的 NumPy `N × P` 矩阵。相应调用是：

```python
from BigFLICA.BigFLICA_cpu import BigFLICA

BigFLICA(
    data_loc=["/absolute/path/vbm.npy", "/absolute/path/fa.npy", "/absolute/path/md.npy", "/absolute/path/zstat1.npy"],
    nlat=20,
    output_dir="/absolute/path/original_output",
    migp_dim=1000,
    dicl_dim=500,
    ncore=4,
)
```

[上游 `BigFLICA_cpu.py`](https://github.com/weikanggong/BigFLICA/blob/master/BigFLICA_cpu.py) 用 SPAMS 字典学习；本功能的压缩模式对照用户 notebook 的 sklearn DicL 变体。原 `sKPCR_regression` 实际计算 t 值却命名为 Z；FNIT 按相同回归和自由度将双侧 t 转为带符号正态 z。mMIGP 特征向量本身有任意正负号；CUDA 实现固定最大绝对载荷为正以便复现，但它不保证与 SciPy 参考的符号一致。字典学习是非凸问题，符号改变会改变固定种子下的拟合轨迹，因此比较压缩模式时必须记录并处理这一差异。`apply_model` 是 FNIT 新增的冻结载荷投影，不等同于原 FLICA 对新被试重新推断后验。

完整流程的原软件数值核对、缓存输出与留出投影检查，见[必要对照证据](../../validation/bigflica/README.md#当前实现所需的补充证据)；字典的同投影控制已移至[独立验证目录](../../validation/dictionary_learning/README.md)。小样本及 C3 检查用于定位和接口核验；30,000 人 C20 结论、阶段时间与剩余工作见 BigFLICA 验证索引。

参考：Gong W, Beckmann CF, Smith SM. [Phenotype Discovery from Population Brain Imaging](https://www.sciencedirect.com/science/article/pii/S1361841521000967). *Medical Image Analysis*, 2021；[BigFLICA 原仓库](https://github.com/weikanggong/BigFLICA)。
