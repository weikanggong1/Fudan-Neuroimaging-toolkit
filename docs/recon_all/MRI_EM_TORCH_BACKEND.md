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
前，生产 profile 不得切换到该后端。这样可以避免把部分 GPU 候选评分误报为
FreeSurfer 等价实现。

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
)
```

该返回值包含 LTA 路径、矩阵、EM cost、各阶段耗时、搜索后端和分块设置。
输入坐标保持 GCA/voxel-LTA 约定，体积张量仍按 FNIT 的 `(x,y,z)` 数组读取；
候选评分阶段不启用 FP16/BF16。`search_backend="torch"` 只在显式指定
CUDA 时可用，默认 `cpu` 不变。

## 命令行和原软件对应关系

当前没有独立命令行入口。原 FreeSurfer 命令为：

```bash
mri_em_register --fstarg template.gca --targ atlas.mgz input.mgz output.lta
```

FNIT 生产调度默认仍使用现有 Python/Conda 混合阶段；`search_backend="torch"`
只用于候选评分实验。要替换原生阶段，必须补充完整 GCA/EM、LTA 及下游连续
真实 T1 回归；模拟 identity 测试不能作为等价性证明。

## 当前验证

`tests/recon_all/test_mri_em_register_torch.py` 验证 identity 采样、不同
候选分块得到相同结果，以及 GPU/CPU 均在最终候选选择时只产生一个索引。
真实 T1 的速度、显存和 LTA 一致性尚未建立，因此文档不会把该模块标成
默认或等价实现。
