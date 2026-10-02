# 字典学习：独立 CPU 与 GPU 接口

`fnit.dictionary_learning` 对一个或多个“样本 × 特征”矩阵分别拟合稀疏字典。CPU 使用 scikit-learn `MiniBatchDictionaryLearning`；GPU 使用仓库的 PyTorch 稀疏求解与顺序字典更新。输入可来自 mMIGP，也可来自其他已经准备好的特征矩阵。函数内部不运行 mMIGP、FLICA、影像配准或脑图回归。

BigFLICA 的压缩流程调用这个模块。字典学习的代码、接口和报告分别放在 [`src/fnit/dictionary_learning/`](../../src/fnit/dictionary_learning/)、本页和 [`validation/dictionary_learning/`](../../validation/dictionary_learning/README.md)。BigFLICA 的完整影像流程见[单独说明](../bigflica/README.md)。

## 安装

在仓库根目录创建主页的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate fnit
python -c "from fnit.dictionary_learning import fit_dictionary_learning, fit_dictionary_learning_streaming"
```

NumPy、scikit-learn、PyTorch、h5py 和 Triton 已在 `environment.yml` 中配置；本功能不需要权重、模板或其他神经影像软件。GPU 接口要求 CUDA；Triton 不可用时字典更新回退到原 PyTorch 算子。

## 输入与输出

CPU 入口为 `fit_dictionary_learning`，兼容名称为 `fit_dicl`。输入 `projected` 是按名称组织的 NumPy 数组，例如 `{"vbm": matrix, "fa": matrix}`；每个数组为 `[样本数, 特征数]` 的有限实数二维矩阵。模态间的样本数和特征数可以不同。CPU 会生成完整标准化矩阵，因此需要能容纳单个模态及其临时数组的主机内存。

GPU 入口为 `fit_dictionary_learning_streaming`，兼容名称为 `fit_dicl_gpu_streaming`。输入目录按以下结构组织；HDF5 的 `data` 为 `[样本数, 特征数]`、float32 或 float64 的有限矩阵，特征数至少为2：

```text
dictionary_input/
  vbm_projected.h5   # data: [样本数, 特征数]
  fa_projected.h5    # data: [样本数, 特征数]
  md_projected.h5    # data: [样本数, 特征数]
```

`_projected.h5` 是约定的文件名，不要求矩阵由 mMIGP 生成。BigFLICA 中，样本对应掩膜体素，特征对应 mMIGP 坐标；一般矩阵也使用同一行列约定。NIfTI、掩膜和被试目录应先由调用者整理成这些矩阵。

两个接口都返回 `dict[str, np.ndarray]`：每个名称对应 `[dicl_dim, 特征数]` 字典。函数先逐特征跨样本标准化输入，零标准差替换为 `0.1`；训练结束后，每个原子跨特征去均值，再将整个字典除以整体 RMS。这沿用 BigFLICA 的字典输出约定，返回值与 sklearn 原始 `components_` 的尺度不同。输出只包含归一化字典，不包含训练编码、标准化参数或 sklearn estimator；本接口也不提供 `.transform()` 或新样本投影。

## Python 调用

### CPU：已在内存中的真实矩阵

```python
from pathlib import Path
import numpy as np
from fnit.dictionary_learning import fit_dictionary_learning

input_directory = Path("/absolute/path/real_feature_matrices")
output_directory = Path("/absolute/path/cpu_dictionaries")
output_directory.mkdir(parents=True, exist_ok=True)

# 每个 .npy 为真实数据的“样本 × 特征”矩阵；不使用 object 数组。
modality_matrices = {
    modality_name: np.load(input_directory / f"{modality_name}.npy", allow_pickle=False)
    for modality_name in ("vbm", "fa", "md")
}
modality_dictionaries = fit_dictionary_learning(
    projected=modality_matrices,
    dicl_dim=200,       # 每个模态拟合 200 个原子；样本数须至少为 200
    max_iter=1000,     # 最大 epoch 数，允许提前停止
    random_state=0,    # 固定 SVD、shuffle 与死原子重采样的随机种子
)

for modality_name, dictionary in modality_dictionaries.items():
    np.save(output_directory / f"{modality_name}_dictionary.npy", dictionary)
```

### GPU：复用已有 HDF5 缓存

```python
from pathlib import Path
import numpy as np
from fnit.dictionary_learning import fit_dictionary_learning_streaming

projection_directory = Path("/absolute/path/dictionary_input")
output_directory = Path("/absolute/path/gpu_dictionaries")
output_directory.mkdir(parents=True, exist_ok=True)
modality_names = ["vbm", "fa", "md"]  # 按此顺序读取对应的 *_projected.h5

modality_dictionaries = fit_dictionary_learning_streaming(
    projected_dir=projection_directory,
    modality_names=modality_names,
    dicl_dim=200,
    device="cuda:0",
    max_iter=1000,
    batch_size=32,
    sparse_iterations=1000,
    alpha=1.0,
    random_state=0,
    feature_block=4096,  # 文件流式统计、预加载与初始化使用的样本行块大小
)

for modality_name, dictionary in modality_dictionaries.items():
    np.save(output_directory / f"{modality_name}_dictionary.npy", dictionary)
```

上述返回文件可以单独保存和复用。它们不能替代含预处理参数的完整 BigFLICA 模型；完整模型的新被试调用仍使用 `fnit.bigflica.apply_model`。

### 从现有 NumPy 文件分块建立 HDF5

如果输入为大型 `.npy`，可通过内存映射分块写入 HDF5，避免一次装入整模态。以下只转换文件结构，不改变数值：

```python
from pathlib import Path
import h5py
import numpy as np

input_directory = Path("/absolute/path/real_feature_matrices")
projection_directory = Path("/absolute/path/dictionary_input")
projection_directory.mkdir(parents=True, exist_ok=True)
modality_names = ["vbm", "fa", "md"]
sample_block_size = 4096

for modality_name in modality_names:
    sample_feature_matrix = np.load(
        input_directory / f"{modality_name}.npy", mmap_mode="r", allow_pickle=False
    )  # [样本数, 特征数]，原始 dtype 为 float32 或 float64
    number_of_samples, number_of_features = sample_feature_matrix.shape
    with h5py.File(projection_directory / f"{modality_name}_projected.h5", "w") as output_file:
        sample_dataset = output_file.create_dataset(
            "data",
            shape=sample_feature_matrix.shape,
            dtype=sample_feature_matrix.dtype,
            chunks=(min(sample_block_size, number_of_samples), number_of_features),
        )
        for sample_start in range(0, number_of_samples, sample_block_size):
            sample_stop = min(sample_start + sample_block_size, number_of_samples)
            sample_dataset[sample_start:sample_stop] = sample_feature_matrix[sample_start:sample_stop]
```

## 参数

| 参数 | 入口与含义 |
|---|---|
| `projected` | CPU：模态名称到二维 NumPy 矩阵的映射；每行一个样本，每列一个特征。 |
| `projected_dir` | GPU：包含 `<名称>_projected.h5` 的目录，各文件必须有二维 `data` 数据集。 |
| `modality_names` | GPU：读取的名称及顺序，不包含 `_projected.h5` 后缀。 |
| `dicl_dim` | 两端：字典原子数，至少为 2，不得超过对应模态的样本数；可大于特征数。 |
| `max_iter` | 两端：最大 epoch 数，默认 1000，至少为 1；达到字典变化或连续无改善的停止条件时提前结束。 |
| `random_state` | 两端：整数随机种子，默认 0；每模态分别初始化随机状态。 |
| `device` | GPU：CUDA 设备字符串，默认 `"cuda:0"`；此入口不接受 CPU 计算设备。 |
| `batch_size` | GPU：每次稀疏编码的样本数，默认 32，至少为 1；CPU 入口固定为 32。改动批量会改变在线更新轨迹。 |
| `sparse_iterations` | GPU：兼容 LARS 的路径事件上限，默认 1000，至少为 1；达到上限仍未求解时明确报错。ADMM 每20步检查一次，最多20轮检查后回退 LARS。 |
| `alpha` | GPU：L1 惩罚系数，默认 1.0；CPU 入口使用 sklearn 的默认 `alpha=1.0`。 |
| `feature_block` | GPU：HDF5 流式统计、预加载及随机 SVD 的行块大小，默认 4096；虽然参数名保留为 `feature_block`，这里切分的是样本行。 |

## 求解、精度与内存

GPU 以 ADMM 寻找非零系数，再检查活动集解与最优性条件。靠近 sklearn 停止节点的敏感行使用兼容 LARS；前四批、活动集检查失败或预算用尽时整批回退。兼容 LARS 保留节点停止、插值和原子退出规则，不向活动集添加 ridge。批量不超过 32、原子数不超过 256 时，LU 求解前后分别重放 CUDA Graph；LU 仍使用原 PyTorch 求解。图工作区每线程最多缓存四项，并保护跨 CUDA stream 的复用。

字典原子仍按原顺序更新。Triton 只融合逐元素运算，矩阵向量乘法、范数归约、死原子重采样和随机数顺序沿用 PyTorch 路径。此前图执行在 float32、任意 `alpha` 下存在停止阈值二次舍入问题，已改成与 eager 路径相同的一次舍入；对应回归随独立模块维护。

CPU 保留输入数组的 NumPy 标准化精度和 sklearn 训练精度；GPU 内部统计及训练使用 float64，返回也为 float64。没有自动使用 float16/bfloat16，TF32 对这些 float64 运算不提供提速。float32 输入可运行，但它与 CPU float32 标准化及训练的效果匹配尚待验收。

GPU 从 HDF5 逐块读取，不在主机内存中装入完整模态。标准化投影小于 4 GiB 且小于调用时空闲显存的四分之一时，会逐块预加载到 GPU；其他情况按训练批次读取。均值、方差和兼容随机数生成在 CPU，SVD、稀疏编码、字典更新在 GPU。显存还包括初始化和求解临时量；本函数没有独立的硬性显存上限参数，使用者应按样本数、特征数、原子数和并发数设置资源预算。

## 真实数据报告

提取为独立模块前，已有真实 1000 人、完整 VBM/FA/MD 掩膜、R500/D200 的同投影验收。保存的 CPU 参考、历史 GPU 和优化 GPU 总耗时分别为 103.90、141.29 和 63.65 秒；优化版分配显存峰值为 2.31 GiB。不同时间窗的共享 GPU 负载及计时边界不同，这些是运行观测，不能作为受控加速比。

三模态字典和 LASSO 指标通过原容差；FA/MD 的 OMP30 重建差仍为 2.44%/5.33%，超过原 0.1% 门槛。OMP 检查使用同一 CPU 编码器，本接口没有增加 GPU OMP transform。该试验没有重跑 mMIGP、FLICA 或最终脑图，不能据此推断 BigFLICA 全流程等价。完整指标、运行环境、哈希和复现命令见[效果匹配报告](../../validation/dictionary_learning/dicl_match_real1000_20261001.md)与[速度优化报告](../../validation/dictionary_learning/dicl_speed_optimization_real1000_20261001.md)。独立模块提取后的回归证据见[验证索引](../../validation/dictionary_learning/README.md)。

## 原实现与参考

原软件没有对应的神经影像命令行。本功能对应 sklearn 的 Python 调用，CPU 参考训练可写为：

```python
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

sample_feature_matrix = np.load("/absolute/path/real_feature_matrix.npy", allow_pickle=False)
feature_mean = sample_feature_matrix.mean(axis=0)
feature_std = sample_feature_matrix.std(axis=0)
feature_std[feature_std == 0] = 0.1
standardized_samples = (sample_feature_matrix - feature_mean) / feature_std
number_of_atoms = 200

dictionary_learner = MiniBatchDictionaryLearning(
    n_components=number_of_atoms,
    max_iter=1000,
    batch_size=32,
    transform_n_nonzero_coefs=max(1, int(number_of_atoms * 0.15)),
    random_state=0,
)
raw_dictionary = dictionary_learner.fit(standardized_samples).components_
centered_dictionary = raw_dictionary - raw_dictionary.mean(axis=1, keepdims=True)
dictionary = centered_dictionary / np.sqrt(np.mean(centered_dictionary ** 2))
```

`transform_n_nonzero_coefs` 只配置 estimator 的后续 transform；此处训练和 FNIT 接口均未调用 transform。默认 CPU 配置沿用用户指定 notebook 的 sklearn 变体；[原 BigFLICA](https://github.com/weikanggong/BigFLICA/blob/master/BigFLICA_cpu.py) 使用 SPAMS，须区分两种实现。

来源与引用：[sklearn 1.7 系列接口文档](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html)、[sklearn 1.7.1 字典学习源码](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/decomposition/_dict_learning.py)、[LARS 源码](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/linear_model/_least_angle.py)；Mairal J, Bach F, Ponce J, Sapiro G. [Online Learning for Matrix Factorization and Sparse Coding](https://jmlr.org/papers/v11/mairal10a.html). *JMLR*, 2010, 11:19–60。ADMM 方程参考 [SPORCO BPDN](https://sporco.readthedocs.io/en/latest/modules/sporco.admm.bpdn.html)，运行时不安装或调用 SPORCO。
