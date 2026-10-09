# SuperBigFLICA：监督多模态成分与预测

| 项目 | 内容 |
|---|---|
| 输入 | 多模态标准空间影像/mask及用户目标CSV |
| 输出 | 共享成分、留出预测、空间图与固定模型 |
| 对应原软件 | 原SuperBigFLICA连续模型；分类为FNIT扩展 |
| Python / CLI | run_superbigflica / fnit-superbigflica fit、apply、plot |
| CPU / GPU | CPU/CUDA；float32训练，统计float64 |

## 1. 功能简介

`run_superbigflica` 同时学习多模态共享影像成分和用户指定的连续或分类目标。按被试ID匹配影像与CSV，再划分训练、验证、测试；只用训练集拟合标准化，验证集选模，测试集评估。保存模型可对新被试输出course和预测。

FNIT依据原方法独立实现线性自编码、共享成分融合、影像重建、空间稀疏与监督损失。PyTorch训练和推理，nibabel/HDF5读写，不运行原软件或外部神经影像pipeline。训练/缓存float32，CUDA允许TF32，不用FP16/BF16；空间回归与t→z统计float64。原软件许可不明确，不复制其源码。

```mermaid
flowchart LR
    A[多模态影像与用户目标CSV] --> B[ID匹配和固定数据划分]
    B --> C[仅训练集标准化]
    C --> D[共享成分与监督训练]
    D --> E[验证集选模]
    E --> F[冻结模型与测试集评估]
    F --> G[预测、course、脑图与固定模型]
    G --> H[可选新被试推理]
```

## 2. Python 调用

```python
import json
from pathlib import Path
from fnit.superbigflica import run_superbigflica, apply_model, plot_superbigflica

configuration = json.loads(Path("/data/superbigflica_config.json").read_text())  # 用户配置
model_directory = run_superbigflica(
    subjects_root="/data/cohort_images",  # 被试影像根目录
    modalities=configuration["modalities"],  # 每模态image和mask映射
    phenotypes_csv="/data/targets.csv",  # 独立用户目标CSV
    targets=configuration["targets"],  # 目标列名与类型
    output_dir="/data/results/superbigflica",  # 新的模型目录
    n_components=20,  # 共享成分数，须满足训练人数和有效秩
    split_column="split",  # CSV固定train/validation/test划分
    device="cuda:0",  # 训练设备
    max_gpu_gb=18.0,  # 保守18GiB预算，不代表实测峰值
)
new_subject_prediction = apply_model(
    model_dir=model_directory,  # 冻结模型及训练标准化
    subject_dir="/data/cohort_images/NEW_SUBJECT",  # 未参与选模的新被试
    device="cuda:0",  # 推理设备
    output_file="/data/results/new_prediction.json",  # 可选JSON保存
)
figure_directory = plot_superbigflica(
    model_dir=model_directory,  # 已完成模型，不重新训练
    output_dir="/data/results/superbigflica_figures",  # 图保存目录
    labels={"trait_01": "Trait 01 (user unit)", "trait_02": "Trait 02"},  # 通用显示名
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
### 用户目标和配置格式

```json
{"modalities": {
  "vbm": {"image": "vbm.nii.gz", "mask": "/data/masks/vbm.nii.gz"},
  "fa": {"image": "fa.nii.gz", "mask": "/data/masks/fa.nii.gz"},
  "md": {"image": "md.nii.gz", "mask": "/data/masks/md.nii.gz"}
}, "targets": {"trait_01": "continuous", "trait_02": "binary"}}
```

```csv
subject_id,trait_01,trait_02,split
SUBJECT_A,1.2,0,train
SUBJECT_B,2.0,1,validation
SUBJECT_C,,0,test
```

以上三行只展示存储格式，不能作为可训练数据规模或benchmark。trait_01/trait_02是通用占位列名，替换成用户CSV列名即可。

- ID按字符串精确匹配并保留前导零；重复ID报错。subjects列表包含训练、验证和测试候选。
- continuous需要有限数值；binary训练集恰好两类，multiclass至少三类，categorical自动按训练类别数识别。文字或离散编码均按标签处理。
- 空值、NA、N/A、NaN、null屏蔽该目标损失，不填补标签；被试仍可参与其他目标和影像重建。
- 显式split使用train/validation/test。省略时默认60%/20%/20%，按第一个分类目标分层；家系/重复扫描需由用户同组划分。
- 训练人数必须大于C+1，验证和测试各至少2；连续/分类的有效标签和类别数也需满足检查，不能仅看总人数。
- 预测连续值还原到用户目标原单位；分类输出标签与概率。外推新类别和不匹配网格会报错。

**`run_superbigflica` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `subjects_root` | 是 | `str / Path` | `—` | 每名被试一个目录的多模态影像根目录。 |
| `modalities` | 是 | `Mapping[str, Mapping[str, str]]` | `—` | 有序模态名→{image:被试目录相对路径, mask:绝对掩膜路径}。 |
| `phenotypes_csv` | 是 | `str / Path` | `—` | 被试ID与用户自有目标列的CSV；不要求UKB列名。 |
| `targets` | 是 | `Mapping[str, str]` | `—` | CSV列名→continuous/binary/multiclass/categorical。 |
| `output_dir` | 是 | `str / Path` | `—` | 本次结果目录；路径按当前工作目录解析。 |
| `n_components` | 是 | `int` | `—` | 共享成分数C，必须与输入被试数、压缩维度及数据有效秩相容。 |
| `id_column` | 否 | `str` | `'subject_id'` | CSV被试ID列，按字符串精确匹配目录名，保留前导零。 |
| `split_column` | 否 | `str / None` | `None` | 划分列；值为train/validation/test；None使用自动划分。 |
| `subjects` | 否 | `Sequence[str] / None` | `None` | 被试目录名序列；None自动识别；顺序会写入模型。 |
| `validation_fraction` | 否 | `float` | `0.2` | 自动划分验证比例；显式split时不用。 |
| `test_fraction` | 否 | `float` | `0.2` | 自动划分测试比例；显式split时不用。 |
| `max_epochs` | 否 | `int` | `50` | 恰好训练的epoch数，之后保留验证损失最低的模型。 |
| `batch_size` | 否 | `int` | `64` | 每批并行样本数；具体约束见各入口。 |
| `learning_rate` | 否 | `float` | `0.001` | RMSprop模型及Adam自动损失权重的初始学习率。 |
| `dropout` | 否 | `float` | `0.2` | 训练时影像和共享成分dropout概率；推理关闭。 |
| `relative_weight` | 否 | `float` | `0.5` | 影像重建/稀疏项相对权重，范围(0,1)。 |
| `class_weight` | 否 | `str` | `'balanced'` | balanced按训练类别频数冻结权重；none使用普通交叉熵。 |
| `random_state` | 否 | `int` | `0` | 初始化和数据划分随机种子。 |
| `device` | 否 | `str` | `'auto'` | PyTorch设备；auto自动选择CUDA，否则CPU，可显式指定cpu或cuda:N。 |
| `max_gpu_gb` | 否 | `float` | `19.0` | 预算单位GiB，默认19.0约20.40十进制GB；实际峰值另记。 与严格20 GB目标有差别；需要严格预算时应留出CUDA上下文余量。 |
| `feature_block` | 否 | `int` | `2048` | 分块大小；DicL为样本行，影像流程为体素列。 |
| `top_voxels` | 否 | `int` | `1000` | 每张成分图保留绝对z值最大的体素数。 |
| `make_plots` | 否 | `bool` | `True` | 是否在拟合后自动输出权重、训练Top3与测试散点/ROC。 |

**`apply_model` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `model_dir` | 是 | `str / Path` | `—` | 训练保存的模型目录。 |
| `subject_dir` | 是 | `str / Path` | `—` | 一名被试的输入目录，须包含上述全部模态。 |
| `device` | 否 | `str` | `'auto'` | PyTorch设备；auto自动选择CUDA，否则CPU，可显式指定cpu或cuda:N。 |
| `output_file` | 否 | `str / Path / None` | `None` | 可选返回值保存路径；None只返回内存对象。 |

**`plot_superbigflica` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `model_dir` | 是 | `str / Path` | `—` | 训练保存的模型目录。 |
| `output_dir` | 否 | `str / Path / None` | `None` | 本次结果目录；路径按当前工作目录解析。 |
| `labels` | 否 | `Mapping[str, str] / None` | `None` | 目标列名→中文显示名/单位的映射；None沿用用户列名。 |

### 输出

```text
superbigflica/
├── model.json / model.pt
├── input_store/<模态>.h5
├── subj_course.npy / subj_course.tsv
├── predictions.csv / metrics.json
├── history.csv
├── prediction_weights.npy
├── <模态>_{zstat,mean,std}.npy
├── maps/<模态>/component-001_{zstat,top-1000}.nii.gz
└── plots/{latent_weights,target-001_top-components,...}.{png,svg,pdf}
```

具体文件名以[保存代码](../../src/fnit/superbigflica/pipeline.py)为准。`run_superbigflica`返回模型目录Path，已有非空输出目录拒绝覆盖。

| 输出 | shape、空间与含义 |
|---|---|
| course | `[Nsubjects,C]` float32；TSV记录ID与split，成分无物理单位。 |
| predictions.csv | 每人一行，连续原单位值，分类标签和各类概率；含ID。 |
| metrics/loss/model | 训练/验证/测试汇总、选模epoch、目标编码与冻结标准化。 |
| 空间z图 | 每模态`[X,Y,Z]` float32，沿用mask affine/orientation/标准空间、mask外零，正态z。 |
| top图 | 按绝对z保留最多top_voxels；不是显著性校正图。 |
| 新被试JSON/返回dict | subject_id、`[C]`components、按用户目标名组织的predictions。 |
| plots | PNG和可编辑SVG/PDF；排名只用训练数据，散点/ROC使用测试数据。 |

图中的course方向和z符号按固定模型保留。测试不参与Top3选择；连续按训练\|r\|，二分类按max(AUC,1-AUC)，多分类按η²排名。没有可计算ROC的类别记null。

模型、CSV、绘图摘要可能包含被试ID和用户目标名，需在私有环境使用。模型缓存按预算自动选择GPU预载或HDF5批读，实际batch/feature_block记录在model.json；预算失败时回到HDF5。

### 冻结模型后绘图

绘图读取已保存模型、course、预测和训练名单，不重新训练。用通用用户标签写明单位，图和绘图摘要仍放在私有输出目录。

```python
from fnit.superbigflica import plot_superbigflica

plot_directory = plot_superbigflica(
    model_dir="/data/results/superbigflica",  # 上例训练保存的模型
    output_dir="/data/results/superbigflica_figures",  # 单独图示目录
    labels={"trait_01": "目标一（用户单位）", "trait_02": "目标二"},  # 显示名映射
)
```

目标显示名只改变注释，不改变编码、预测或训练Top3排序。连续散点坐标沿用原单位；二分类ROC需要测试集含两个类别，缺失时明确记录不可计算。

## 3. 命令行调用

```bash
fnit-superbigflica fit --subjects-root /data/cohort_images    --config /data/superbigflica_config.json --phenotypes-csv /data/targets.csv    --split-column split --output-dir /data/results/superbigflica    --n-components 20 --device cuda:0 --max-gpu-gb 18
fnit-superbigflica apply --model-dir /data/results/superbigflica    --subject-dir /data/cohort_images/NEW_SUBJECT    --output-file /data/results/new_prediction.json --device cuda:0
fnit-superbigflica plot --model-dir /data/results/superbigflica    --output-dir /data/results/superbigflica_figures --labels /data/display_labels.json
```

| CLI | Python | 含义 |
|---|---|---|
| --subjects-root / --output-dir / --n-components | 同名下划线参数 | 数据根、模型目录和C |
| --config | modalities、targets | JSON两个映射 |
| --phenotypes-csv / --id-column / --split-column | 同名下划线参数 | CSV及匹配/划分列 |
| --subjects-file | subjects | 每行ID，包含全部数据划分 |
| --validation-fraction / --test-fraction | 同名下划线参数 | 自动划分比例 |
| --max-epochs / --batch-size / --learning-rate | 同名下划线参数 | 训练配置 |
| --dropout / --relative-weight / --class-weight / --random-state | 同名下划线参数 | 损失/随机设置 |
| --device / --max-gpu-gb / --feature-block / --top-voxels | 同名下划线参数 | 设备、资源与图输出 |
| --no-plots | make_plots=False | 跳过自动绘图 |
| apply --model-dir / --subject-dir / --output-file | 同名下划线参数 | 冻结推理，CLI必须输出文件 |
| plot --labels | labels | JSON列名→显示名，可标用户单位 |

`fit`默认自动绘图；`plot`不训练。Python apply允许output_file=None；CLI要求文件。未指定subjects-file则使用影像与CSV匹配交集。

## 4. 原软件调用

原SuperBigFLICA没有独立CLI，接收已经准备的影像矩阵列表及连续目标矩阵。在独立原作者目录：

```python
import numpy as np
from SupBigFLICA_cpu import SupervisedFLICA

training_imaging_matrices = [np.load(f"/data/reference_inputs/{name}_train.npy", allow_pickle=False)
                             for name in ("vbm", "fa", "md")]  # 按训练均值/标准差准备的模态矩阵
validation_imaging_matrices = [np.load(f"/data/reference_inputs/{name}_validation.npy", allow_pickle=False)
                               for name in ("vbm", "fa", "md")]  # 用相同训练标准化参数
training_target_matrix = np.load("/data/reference_inputs/targets_train.npy", allow_pickle=False)  # 标准化连续目标
validation_target_matrix = np.load("/data/reference_inputs/targets_validation.npy", allow_pickle=False)  # 同目标顺序

relative_weight = 0.5  # 原影像/监督权重
reference_outputs = SupervisedFLICA(
    x_train=training_imaging_matrices,  # [被试,特征]模态列表
    y_train=training_target_matrix,  # 连续目标训练矩阵
    x_test=validation_imaging_matrices,  # 原参数名test，选模实际传validation
    y_test=validation_target_matrix,  # 验证目标
    dropout=0.2, device="cpu",  # 原设备和dropout
    auto_weight=[1, 1, 1, 1],  # 原四损失自动权重
    lambdas=[relative_weight, relative_weight, 1-relative_weight, 1-relative_weight],
    nlat=20, lr=0.001, random_seed=0, maxiter=50, batch_size=64,
    init_method="random",  # 与FNIT随机初始化对应
)
```

上例矩阵需先按同训练标准化准备，原版的 `get_model_param`读取冻结best_model进行投影。

| FNIT | 原连续模型 |
|---|---|
| n_components / learning_rate / random_state | nlat / lr / random_seed |
| max_epochs=50 | 原maxiter=50执行51轮；不能按同名计作相同训练 |
| 平均验证监督损失选模 | 连续目标Pearson相关选模 |
| categorical、缺失标签、ID匹配 | FNIT扩展，无对应原分类benchmark |
| 各模态z图 | 原提取循环重复第一模态且把t命名z；FNIT独立回归并t→z |

原源码冻结 `6695b638…`，未明确授权FNIT再分发。连续前向/损失/梯度控制和新增分类预测应分开评价。

## 5. 最新精度和运行时间

最新正式真实运行见[匿名效率报告](../../validation/superbigflica/efficiency_real5000.json)，绑定实现 `aef3b4f8…`及verified_core_sha256。5000真实被试、完整三模态mask、C20，训练/验证/测试3000/1000/1000，50epoch、batch64、随机初始化。目标名称和字段不在本页复制；本轮未按140c3739重跑。

| 完整公开API，含读写/缓存/绘图 | 观测值 |
|---|---:|
| 总墙钟 | 660.04 s |
| 缓存前载 | 38.86 s |
| 训练、选模和预测 | 101.37 s |
| 空间统计和脑图 | 18.30 s |
| GPU peak allocated / RSS | 12.25 / 1.39 GiB |
| 匿名连续目标测试r / R² | 0.1731 / 0.0254 |
| 匿名二分类测试ROC AUC / balanced accuracy | 0.7226 / 0.6716 |

共享H100、8CPU线程，float32训练/TF32允许、float64空间统计，GPU预载11.22GiB。120张float32 NIfTI均有限且mask外零，训练course秩20、加截距秩21。单次共享系统观测不代表稳定加速。

同一队列旧HDF5版本1517.16 s，新模型/损失/course/预测逐值相同；空间图9个float32末位差、最大4.77e-7、阈值支持相同。原版连续模型只覆盖8人完整特征的前向/损失/梯度控制：重建最大差1.19e-7、梯度最大差1.49e-8；没有原软件完整5000人端到端配对。

当前仓库图含具体用户目标展示文本，本页不重新嵌入。适合公开的匿名真实reference/FNIT/difference脑图尚未提供；原始图/指标保留不改，公开入口用户可自行用通用labels重画自己的模型。


<!-- FNIT-UNIFIED-BENCHMARK-20261008 -->
### 本轮统一 benchmark 摘要（2026-10-08）

5000 人、20 成分、3000/1000/1000 划分、50 epoch 的公开真实结果是 GPU-only：H100 完整 API **660.04 s**，训练/选模/预测 **101.37 s**，空间统计/脑图 **18.30 s**，allocation/RSS **12.25/1.39 GiB**；连续目标测试 `r/R²=0.1731/0.0254`，二分类 `AUC/balanced accuracy=0.7226/0.6716`。本轮没有同输入、同线程 CPU 全链对照；新旧 HDF5 输入的模型、course 和预测逐值相同，空间图最大差 `4.77e-7`。见 [统一 benchmark 索引](../BENCHMARK_INDEX.md)。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-01 | aef3b4f8 | GPU完整输入缓存及各模态空间统计修复 | 5000人API结果与旧模型一致；原模型控制有限 |
| 2026-09-30 | 82c7a1ef | 5000人数据划分、类别平衡和绘图 | 真实留出指标；不公开目标字段 |
| 2026-09-30 | 4a0e0376 | 初始监督多模态接口 | ID匹配、CSV与冻结推理 |

更早的debug、profiling和长表保留在[旧README归档](../../validation/superbigflica/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="流程"></a>
<a id="安装与输入"></a>
<a id="python-示例"></a>
<a id="命令行示例"></a>
<a id="参数"></a>
<a id="训练与输出"></a>
<a id="图像与成分排序"></a>
<a id="与原实现的对应"></a>
<a id="真实数据验证"></a>
<a id="同队列前后对照"></a>
<a id="本次真实输出图"></a>
<a id="成分预测权重"></a>
<a id="连续目标训练-top3-成分脑图"></a>
<a id="分类目标训练-top3-成分脑图"></a>
<a id="连续目标独立测试散点"></a>
<a id="分类目标独立测试-roc"></a>
<a id="参考文献与原实现"></a>

## 7. 参考文献、原软件和资源

- 原仓库：[SuperBigFLICA](https://github.com/weikanggong/SuperBigFLICA)、[核对版本6695b638](https://github.com/weikanggong/SuperBigFLICA/tree/6695b638802aab50f43f889e97af223f8191018e)，`SupBigFLICA_cpu.py`的SupervisedFLICA/get_model_param。
- 多模态基础方法：Gong等，2021，[Phenotype Discovery from Population Brain Imaging](https://www.sciencedirect.com/science/article/pii/S1361841521000967)。具体监督方法说明以原仓库所附文章为准，不编造出版信息。
- FNIT：[pipeline.py](../../src/fnit/superbigflica/pipeline.py)、[model.py](../../src/fnit/superbigflica/model.py)、[plotting.py](../../src/fnit/superbigflica/plotting.py)。
- 原代码未获明确再分发许可；FNIT提供独立实现，不随包复制原仓库源码。

### 外部资源

本功能不需要预训练权重、图谱或模板，也不自动下载真实输入数据。用户输入与参考软件的许可由各自来源决定。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 无额外模型资源 | — | — | 不适用 | 不适用 | 不适用 |
