# mri_em_register 的纯 PyTorch 候选后端

`fnit.recon_all.mri_em_register_torch` 提供候选仿射变换的分块采样和
GPU 评分。它把候选矩阵、体数据、掩膜和评分保留在同一个 PyTorch 设备，
仅在候选选择结束时把一个索引同步到 Python，因此可用于剖析候选搜索中
逐候选 CPU/GPU 往返造成的开销。

## 当前范围

该模块不是 `mri_em_register` 的完整替代。GCA 分类器参数估计、EM 更新、
刚体/仿射优化和 LTA 写出仍由现有实现负责；当前模块只实现固定候选的
`grid_sample` 与 NCC/SSD 排序。仓库已有的 `GCASearchScorer` 则可通过
`register_t1(..., search_backend="torch")` 复用到翻译和线性候选搜索，EM
仍在 CPU 上运行。没有完成同输入 LTA、`norm`、`brain` 和下游表面输出回归
前，不应将新的评分实现称为完整 FreeSurfer 等价实现。NCC/SSD候选接口没有接入
生产 GCA 链；生产 `native_optimizations='auto'` 在 CUDA 上复用的是
`GCASearchScorer` 的 GCA likelihood，并保留 CPU EM，不使用 NCC/SSD 替代目标函数。

## Python 调用

```python
import torch

from fnit.recon_all.mri_em_register_torch import (
    TorchAffineScoreConfig,
    evaluate_affine_candidates,
)

source_volume = torch.from_numpy(source_array).to("cuda:0", dtype=torch.float32)
target_volume = torch.from_numpy(target_array).to("cuda:0", dtype=torch.float32)
candidate_matrices = torch.from_numpy(candidate_lta_matrices).to(
    "cuda:0", dtype=torch.float32
)
sampled, scores, best_index = evaluate_affine_candidates(
    source=source_volume,  # [D,H,W]，源图像体素顺序
    target=target_volume,  # [D,H,W]，目标图像体素顺序
    source_to_target=candidate_matrices,  # [K,4,4]，目标体素到源体素的采样变换
    metric="ncc",  # 候选评分："ncc" 或 "ssd"
    config=TorchAffineScoreConfig(
        candidate_chunk=16,  # 每次送入 GPU 的候选数，控制峰值显存
        padding_mode="border",  # 边界采样语义
        align_corners=True,  # 与参考 grid_sample 约定保持一致
        use_tf32=True,  # 保持项目默认 TF32 策略，不启用 FP16/BF16
    ),
)
```

返回的 `sampled` 形状为 `[K,D,H,W]`，`scores` 为 `[K]`，`best_index`
是唯一在搜索结束时转换为 Python 整数的索引。输入 affine 可以通过
`source_affine` 和 `target_affine` 传入；坐标索引顺序为 `x,y,z`，张量
存储顺序为 `z,y,x`。

## 完整 Python 注册的显式实验选项

```python
from fnit.recon_all.mri_em_register_python import register_t1

report = register_t1(
    nu_path="/work/sub-07/mri/nu.mgz",  # uint8/float32 的 FNIT nu.mgz
    atlas_path="/assets/average.gca",  # 固定版本 GCA 图谱
    mask_path="/work/sub-07/mri/brainmask.mgz",  # 与 nu 相同网格的掩膜
    output_path="/work/sub-07/mri/talairach.lta",  # 输出 voxel LTA
    device="cuda:0",  # GPU 候选评分设备
    search_backend="torch",  # 只将候选评分迁移到 Torch
    candidate_chunk=64,  # GPU 候选分块，控制显存
    sample_chunk=8192,  # GCA 样本分块，控制显存
    reduce_on_device=True,  # 在 GPU 上累加 likelihood，减少同步和 D2H
    inverse_backend="cpu",  # 保留已有逐候选余子式公式；torch为新的显式批量求逆实验
)
```

该返回值包含 LTA 路径、矩阵、EM cost、各阶段耗时、搜索后端、分块设置和
`reduce_on_device` 和 `inverse_backend` 状态。设备端归约使用 float64 累加，未启用 FP16/BF16。
输入坐标保持 GCA/voxel-LTA 约定，体积张量仍按 FNIT 的 `(x,y,z)` 数组读取；
候选评分阶段不启用 FP16/BF16。`search_backend="torch"` 只在显式指定
CUDA 时可用，默认 `cpu` 不变。

## 命令行和原软件对应关系

NCC/SSD接口没有独立官方命令。完整 Python GCA 注册提供以下实验 CLI：

```bash
python -m fnit.recon_all.mri_em_register_python \
  /data/nu.mgz /assets/average/RB_all_2020-01-02.gca /data/talairach.lta \
  --mask /data/brainmask.mgz --device cuda:0 --search-backend torch \
  --candidate-chunk 256 --sample-chunk 8192 --reduce-on-device --inverse-backend torch
```

原固定 T1 阶段命令为：

```bash
mri_em_register -uns 3 -mask brainmask.mgz nu.mgz RB_all_2020-01-02.gca talairach.lta
```

FNIT 生产调度为 Python/Conda 混合流程；`auto` 的 CUDA GCA 使用已存在的评分路径，
新的 `inverse_backend='torch'` 尚未接入默认调度。它只改变固定余子式的批量执行位置，
必须另做真实全搜索、LTA 和下游回归；模拟 identity 测试不能作为等价性证明。

## 当前验证

`tests/recon_all/test_mri_em_register_torch.py` 验证 identity 采样、不同
候选分块得到相同结果，以及 GPU/CPU 均在最终候选选择时只产生一个索引。
真实 T1 的速度、显存和 LTA 一致性尚未建立，因此文档不会把该模块标成
默认或等价实现。

下表是此前保存的阶段记录，不是2026-10-09的新版本测试。在 gpucw1 的共享 H100
环境，对 ds000114 sub-07 的同一 `nu.mgz`、GCA 和
brainmask 做完整 Python 注册配对：

| 配置 | 总耗时 | 翻译 | 线性搜索 | EM | LTA 矩阵 |
|---|---:|---:|---:|---:|---|
| CPU source-order | 420.51 s | 17.47 s | 379.13 s | 8.62 s | 基线 |
| Torch GPU，逐块回传 | 481.06 s | 19.24 s | 435.21 s | 8.81 s | 逐元素相同 |
| Torch GPU，`reduce_on_device=True` | 335.08 s | 13.45 s | 296.15 s | 8.48 s | 逐元素相同 |

Torch 设备端归约相对 CPU 观测加速 **20.3%**。GPU 当时同时运行其他作业，
该数字是共享负载观测，不能当作独占设备吞吐。设备端 float64 reduction
改变了极端越界候选的求和树，单个 score 在测试中最大相差 `0.0078125`；
真实 sub-07 的最终候选和 LTA 矩阵逐元素相同。下一步需在空闲 GPU 上复测，
并将同一 LTA 送入 `norm/brain/wm/filled` 连续链。它不证明下面的新批量求逆已通过。

## 2026-10-09：批量余子式求逆与较大评分块

本次原始T1控制运行中，GCA阶段实测444.146秒，其中线性搜索400.532秒、
translation21.225秒、EM8.934秒。完整整例未结束，不能将该单阶段记录当作整例结果。
现有路径每64候选逐个调用CPU求逆，再执行多次小CUDA算子。

新增 `vnl_affine_inverse_tensor(matrices=...)` 接收同设备 FP32(B,4,4) 仿射矩阵，
输出同形状、同设备逆矩阵。逐项保留原余子式和FP32乘加顺序；倒数按既有标量
FP64除法后转FP32，不使用 `torch.linalg.inv` 或 TF32矩阵乘法。
非法dtype/shape抛ValueError；仿射末行、有限/奇异检查由调用者或scorer负责。
这是原命令内部步骤，没有独立官方CLI。

`GCASearchScorer(..., inverse_backend='cpu')` 默认保持旧路径，显式 `'torch'`
才批量迁移求逆。`register_t1(..., inverse_backend='torch')` 要求
`search_backend='torch'`，其他组合报错。原始nu与mask同conform网格；source数组
顺序(x,y,z)，候选/LTA保持源体素到GCA体素的变换方向，不改变搜索网格或候选顺序。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `nu_path/atlas_path/mask_path/output_path` | 必填 | 校正图、固定GCA、同网格脑掩膜、输出voxel LTA路径 |
| `device` | `cpu` | 明确设备；Torch评分必须CUDA |
| `search_backend` | `cpu` | cpu为既有Numba搜索评分，torch为驻留GCA likelihood |
| `candidate_chunk` | `64` | 候选数分块；不删除候选，较大块需要更多显存 |
| `sample_chunk` | `8192` | 每块样本数；保持完整样本 |
| `reduce_on_device` | `False` | True在CUDA按FP64归约，再回传每块候选分数 |
| `inverse_backend` | `cpu` | 新torch选项复用相同余子式批量求逆 |

返回包括4×4 LTA矩阵、输出路径、EM cost/trials、linear_iterations、实际后端和
prepare/translation/linear/EM/write/total秒数。失败直接抛异常，部分文件不代表成功。
PyTorch CPU 两项契约回归通过：128个普通/病态矩阵与既有公式逐位一致，非法接口拒绝。
真实GPU评分、完整注册与显存尚未验证：gpucw1入口不可达，默认尚未改变。

`tools/benchmark_recon_gca_inverse.py` 接收具名 `--nu/--mask/--atlas/--output`
和 `--code-version`，`--device=cuda:0`、`--threads=4`；输出完整首轮矩阵/分数npy及
带哈希的JSON、GPU同期进程树采样。固定真实强度/掩膜，原搜索代码生成全部首轮候选；
收集时返回零只用于取得候选网格，**不是真实优化轨迹或最终LTA**。
配对CPU inverse64与Torch inverse64/256/1024：求逆要求逐位一致，分数逐位差异单列
严格复现诊断；固定候选优化回归沿用仓库原有1e-2分数容差，并要求首次argmax不变。
该标准在首次GPU评分前写入脚本；固定网格通过仍须完整注册和LTA回归，
不以小分数尾差或局部速度代替最终指标检查。

本次没有新增依赖；PyTorch/NumPy/Numba/nibabel均在主页Conda环境声明。
原算法与资料：[固定mri_em_register源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mri_em_register)，
[原余子式公式对应工具库](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/matrix.cpp)。
Fischl B. FreeSurfer. *NeuroImage* 62:774–781, 2012。
