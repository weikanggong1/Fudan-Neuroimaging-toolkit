# 字典学习：CPU与GPU公开接口

| 项目 | 内容 |
|---|---|
| 输入 | 模态名→样本×特征矩阵；GPU用HDF5 data |
| 输出 | 每模态原子×特征归一化字典 |
| 对应原软件 | sklearn MiniBatchDictionaryLearning；原BigFLICA用SPAMS |
| Python / CLI | fit_dictionary_learning / fit_dictionary_learning_streaming；无CLI |
| CPU / GPU | CPU用sklearn；GPU用PyTorch float64 |

## 1. 功能简介

`fnit.dictionary_learning` 对一个或多个样本×特征矩阵分别拟合稀疏字典。CPU入口读取内存NumPy数组，GPU入口读取分块HDF5，返回每个模态的归一化字典。函数不做影像配准、mMIGP或FLICA；[BigFLICA](../bigflica/README.md)压缩阶段复用同一模块。

CPU使用scikit-learn MiniBatchDictionaryLearning，GPU使用项目PyTorch求解与顺序原子更新；Triton不可用时更新回到PyTorch。GPU统计/训练采用float64，TF32对该精度不提速，不自动用FP16/BF16。所有依赖已纳入主页Conda环境，不需要权重或模板。

## 2. Python 调用

```python
import numpy as np
from fnit.dictionary_learning import fit_dictionary_learning

modality_matrices = {
    "modality_01": np.load("/data/features/modality_01.npy", allow_pickle=False),
}  # 每个真实输入是有限值二维样本×特征矩阵
modality_dictionaries = fit_dictionary_learning(
    projected=modality_matrices,  # CPU内存输入
    dicl_dim=200,  # 字典原子数，不能超过样本数
    max_iter=1000,  # 最大epoch数，支持提前停止
    random_state=0,  # 固定初始化和shuffle种子
)
np.save("/data/results/modality_01_dictionary.npy", modality_dictionaries["modality_01"])
```

GPU使用相同矩阵结构的HDF5文件，不要求输入一定来自mMIGP：

```python
import numpy as np
from fnit.dictionary_learning import fit_dictionary_learning_streaming

modality_names = ["modality_01"]  # 读取顺序，与文件basename一致
gpu_dictionaries = fit_dictionary_learning_streaming(
    projected_dir="/data/projected",  # 含modality_01_projected.h5的目录
    modality_names=modality_names,  # 文件名不带_projected.h5后缀
    dicl_dim=200,  # 原子数
    device="cuda:0",  # 此入口必须使用CUDA
    max_iter=1000,  # 最大epoch数
    batch_size=32,  # 稀疏编码批大小
    sparse_iterations=1000,  # 兼容LARS事件预算
    alpha=1.0,  # L1惩罚
    random_state=0,  # 随机种子
    feature_block=4096,  # 流式统计和初始化的样本行块
)
np.save("/data/results/modality_01_gpu_dictionary.npy", gpu_dictionaries["modality_01"])
```

### 输入数据格式

- CPU：`Mapping[str, np.ndarray]`，每个数组`[Nsamples,Nfeatures]`、float32或float64、有限值；不同模态的行列数可不同。
- GPU：`<modality>_projected.h5`中二维dataset `data`，float32或float64；特征数至少2。sample对应行、feature对应列。
- 字典原子数至少2，不能超过该模态样本数；允许过完备字典，因此可超过特征数。
- 这些是矩阵接口，没有affine、orientation或MRI单位。BigFLICA投影中行是mask体素、列是mMIGP坐标，不能与被试×体素混淆。
- 逐特征跨样本标准化；零标准差替换成0.1。CPU产生整模态标准化数组，需相应RAM；GPU流式统计和预加载。

可从真实NumPy内存映射分块创建HDF5：

```python
import h5py
import numpy as np

sample_feature_matrix = np.load("/data/features/modality_01.npy", mmap_mode="r", allow_pickle=False)
sample_block_size = 4096  # 只按样本行分块，不改变数值
with h5py.File("/data/projected/modality_01_projected.h5", "w") as projection_file:
    projection_dataset = projection_file.create_dataset(
        "data", shape=sample_feature_matrix.shape, dtype=sample_feature_matrix.dtype
    )  # 保留输入dtype和二维结构
    for sample_start in range(0, sample_feature_matrix.shape[0], sample_block_size):
        sample_stop = min(sample_start + sample_block_size, sample_feature_matrix.shape[0])
        projection_dataset[sample_start:sample_stop] = sample_feature_matrix[sample_start:sample_stop]
```

先建立输出父目录。转换只改变存储结构，不代替输入的科学前处理。

**`fit_dictionary_learning` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `projected` | 是 | `Mapping[str, np.ndarray]` | `—` | 模态名→有限二维NumPy样本×特征矩阵。 |
| `dicl_dim` | 是 | `int` | `—` | 字典原子数≥2，不能超过各模态样本数。 |
| `max_iter` | 否 | `int` | `1000` | 最大迭代/epoch数；正整数，可能按收敛规则提前结束。 |
| `random_state` | 否 | `int` | `0` | 初始化和数据划分随机种子。 |

**`fit_dictionary_learning_streaming` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `projected_dir` | 是 | `str / Path` | `—` | 含各 <名称>_projected.h5 的目录，二维dataset名为data。 |
| `modality_names` | 是 | `Sequence[str]` | `—` | HDF5模态名称及读取顺序，不含文件后缀。 |
| `dicl_dim` | 是 | `int` | `—` | 字典原子数≥2，不能超过各模态样本数。 |
| `device` | 否 | `str` | `'cuda:0'` | 指定PyTorch设备；默认cuda:0，CPU支持按该入口说明选择。 |
| `max_iter` | 否 | `int` | `1000` | 最大迭代/epoch数；正整数，可能按收敛规则提前结束。 |
| `batch_size` | 否 | `int` | `32` | 每批并行样本数；具体约束见各入口。 |
| `sparse_iterations` | 否 | `int` | `1000` | 兼容LARS路径事件预算；耗尽时明确报错。 |
| `alpha` | 否 | `float` | `1.0` | 非负L1稀疏惩罚系数。 |
| `random_state` | 否 | `int` | `0` | 初始化和数据划分随机种子。 |
| `feature_block` | 否 | `int` | `4096` | 分块大小；DicL为样本行，影像流程为体素列。 |

### 输出

接口只返回 `dict[str,np.ndarray]`，不自动创建结果目录。前述用户保存示例产生：

```text
results/
├── modality_01_dictionary.npy
└── modality_01_gpu_dictionary.npy
```

每个字典为`[dicl_dim,Nfeatures]`。训练后每个原子跨feature去均值，再将整个字典除以整体RMS；与sklearn裸`components_`尺度不同。GPU返回float64；CPU保留其NumPy/sklearn训练精度。

返回值不包含训练编码、标准化参数、estimator或新样本transform，不等同于可投影新被试的完整BigFLICA模型。兼容名字`fit_dicl`和`fit_dicl_gpu_streaming`指向同一接口。

GPU没有独立硬性显存上限参数。小于4GiB且小于空闲显存四分之一的标准化投影可能整块预载；其他情况按训练批次读取。初始化、稀疏求解工作区、并发调用仍需纳入预算。减小batch/原子数会改变在线训练轨迹，不能只按显存判断科学等价。

## 3. 命令行调用

没有独立 `fnit-dictionary-learning` 或子命令；使用上述公开Python函数。BigFLICA CLI中的 `--dicl-*` 仅在完整影像流程中配置字典阶段，处理范围不同。

| 配置目的 | Python入口 |
|---|---|
| NumPy数组CPU训练 | `fit_dictionary_learning` |
| HDF5流式CUDA训练 | `fit_dictionary_learning_streaming` |
| 字典训练及FLICA/脑图全流程 | `fnit-bigflica fit`，见BigFLICA页 |

独立导入可在主页环境检查：

```bash
python -c "from fnit.dictionary_learning import fit_dictionary_learning, fit_dictionary_learning_streaming"
```

## 4. 原软件调用

对应sklearn Python API，没有等价神经影像CLI。在独立参考环境处理同一矩阵和标准化：

```python
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

sample_feature_matrix = np.load("/data/features/modality_01.npy", allow_pickle=False)
feature_mean = sample_feature_matrix.mean(axis=0)  # 跨样本均值
feature_std = sample_feature_matrix.std(axis=0)  # 总体标准差
feature_std[feature_std == 0] = 0.1  # 与FNIT同规则
standardized_samples = (sample_feature_matrix - feature_mean) / feature_std
number_of_atoms = 200  # 与FNIT dicl_dim对应
reference_learner = MiniBatchDictionaryLearning(
    n_components=number_of_atoms,  # 原子数
    max_iter=1000,  # 最大epoch
    batch_size=32,  # CPU入口固定batch
    transform_n_nonzero_coefs=max(1, int(number_of_atoms * 0.15)),  # 原配置
    random_state=0,  # 相同种子
)
raw_dictionary = reference_learner.fit(standardized_samples).components_
centered_dictionary = raw_dictionary - raw_dictionary.mean(axis=1, keepdims=True)
reference_dictionary = centered_dictionary / np.sqrt(np.mean(centered_dictionary ** 2))
```

| FNIT | sklearn |
|---|---|
| dicl_dim / max_iter / random_state | n_components / max_iter / random_state |
| CPU固定32 / alpha默认1 | batch_size=32 / alpha=1 |
| GPU alpha/batch_size | 对应稀疏惩罚和在线批次，需分别配对 |

`transform_n_nonzero_coefs`只配置后续transform，当前训练不调用它。原BigFLICA使用SPAMS；本CPU参考是用户notebook的sklearn变体，不能把两种原实现混写。

## 5. 最新精度和运行时间

最新真实阶段记录为[2026-10-01同投影效果检查](../../validation/dictionary_learning/dicl_match_real1000_20261001.md)和[优化报告](../../validation/dictionary_learning/dicl_speed_optimization_real1000_20261001.md)：1000人、完整三模态mask、同一float64 R500投影、D200。源码角色与SHA分别绑定训练和评估，不能把评估器导入SHA当训练实现。

| 同投影字典阶段 | 保存CPU参考 | 历史GPU | 优化GPU |
|---|---:|---:|---:|
| 总观测时间 | 103.90 s | 141.29 s | 63.65 s |
| peak allocated | 不适用 | 见原报告 | 2.31 GiB |

H100共享GPU，CPU参考sklearn1.7.1，GPUfloat64；线程及分阶段观测见原报告，未提供同时间窗的干净端到端加速实验。这里只测字典阶段，不包含mMIGP、FLICA或最终脑图。

字典与LASSO门槛通过；FA/MD的OMP30重建差2.44%/5.33%仍超0.1%原门槛。OMP使用同CPU编码器，本接口没有GPU OMP transform。公开float32输入与CPUfloat32标准化/训练的匹配尚未验收。

[独立模块提取检查](../../validation/dictionary_learning/module_split_20261001.json)表明原训练函数AST未变、缓存版本仍rsvd5rowgraph；未重新做完整real1000科学benchmark。本轮只审核140c3739文档/API，不重标为当前源码实测。

这是矩阵阶段，没有独立原软件/FNIT脑图配对；脑图全流程证据见BigFLICA验证页。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-01 | fc76fb8f | 提取独立CPU/GPU模块并保留兼容别名 | 原函数AST未变；未重跑完整1000人 |
| 2026-10-01 | 571fc9ea | 优化兼容稀疏求解和顺序字典更新 | 1000人同投影；OMP门槛仍未通过 |
| 2026-10-01 | 945242c7 | 修正LARS停止及初始化内存 | 真实投影控制与定向回归 |

更早的debug、profiling和长表保留在[旧README归档](../../validation/dictionary_learning/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="安装"></a>
<a id="输入与输出"></a>
<a id="python-调用"></a>
<a id="cpu已在内存中的真实矩阵"></a>
<a id="gpu复用已有-hdf5-缓存"></a>
<a id="从现有-numpy-文件分块建立-hdf5"></a>
<a id="参数"></a>
<a id="求解精度与内存"></a>
<a id="真实数据报告"></a>
<a id="原实现与参考"></a>

## 7. 参考文献、原软件和资源

- sklearn：[MiniBatchDictionaryLearning文档](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html)、[1.7.1字典源码](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/decomposition/_dict_learning.py)、[LARS源码](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/linear_model/_least_angle.py)。
- Mairal等，2010，[Online Learning for Matrix Factorization and Sparse Coding](https://jmlr.org/papers/v11/mairal10a.html)，JMLR11:19–60。
- FNIT：[cpu.py](../../src/fnit/dictionary_learning/cpu.py)、[torch_backend.py](../../src/fnit/dictionary_learning/torch_backend.py)、[验证索引](../../validation/dictionary_learning/README.md)。

### 外部资源

本功能不需要预训练权重、图谱或模板，也不自动下载真实输入数据。用户输入与参考软件的许可由各自来源决定。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 无额外模型资源 | — | — | 不适用 | 不适用 | 不适用 |
