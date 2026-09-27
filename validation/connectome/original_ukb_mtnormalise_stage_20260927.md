# 原 UKB 流程：脑掩膜形态学和三组织归一化

本阶段固定同一真实 UKB 的平均 b0 官方 BET 掩膜、原版与 PyTorch 已核对的原始 WM/GM/CSF FOD，以及相同 DWI 网格。原脚本从 BET 掩膜分别生成**膨胀两次**的 FOD 拟合掩膜和**侵蚀两次**的 `mtnormalise` 掩膜；不能把两者互换。原版 MRtrix 固定在 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`。完整输入哈希、原图和主体标识只留在服务器；公开[聚合 JSON](original_ukb_mtnormalise.public.json)可核对数值。

## 形态学函数

`maskfilter_six_connected(mask, operation, passes)` 接受与 DWI 网格相同的 `torch.bool` 三维张量 `mask`；`operation` 为 `"dilate"` 或 `"erode"`，`passes` 为迭代次数。每次使用中心及六个面相邻体素，图像外视作零。输出仍是同形状、同设备的布尔张量。原版等价命令：

```bash
maskfilter bet_mask.mif dilate fod_mask.mif -npass 2
maskfilter bet_mask.mif erode norm_mask.mif -npass 2
```

```python
from fnit.connectome.masks import maskfilter_six_connected

fod_mask = maskfilter_six_connected(
    mask=bet_mask,       # 输入；平均 b0 的官方 BET 二值掩膜，torch.bool [X,Y,Z]
    operation="dilate", # 膨胀；供多组织 CSD 拟合
    passes=2,           # 原脚本连续两轮
)
norm_mask = maskfilter_six_connected(
    mask=bet_mask,      # 输入；与上面完全相同的原始 BET 掩膜
    operation="erode", # 侵蚀；供 mtnormalise 拟合
    passes=2,          # 原脚本连续两轮
)
```

同一 `104×104×72` 真实脑掩膜有 182,616 个阳性体素；原版／FNIT 膨胀结果均为 212,831 个，XOR **0**；侵蚀结果均为 154,441 个，XOR **0**。PyTorch CPU 已载入张量核心分别为 0.0395／0.0926 s；MRtrix CPU 完整命令为 0.05／0.06 s。时间范围不同。复跑入口为 [`tools/benchmark_connectome_maskfilter.py`](../../tools/benchmark_connectome_maskfilter.py)，其 `--brain-mask` 是原始 BET 掩膜，`--reference-dilated`、`--reference-eroded` 是独立原版输出，`--output` 写入包含私有哈希的完整报告。

## 三组织归一化函数

`normalise_mrtrix_three_tissue` 的输入是同一 DWI 网格上的原始 `wm_sh` float32 `[X,Y,Z,45]`、`gm` 和 `csf` float32 `[X,Y,Z]`、两次侵蚀的布尔 `mask` `[X,Y,Z]`、体素中心到 RAS 世界毫米的 `affine` `[4,4]`，全部在同一 CPU/CUDA 设备。返回 `MTNormaliseResult`：`wm`、`gm`、`csf` 是归一化 float32 图，形状不变；`field` 为 float32 三维偏置场；`accepted_mask` 为离群值剔除后的三维布尔图；`balance_factors` 为 float64 长度 3 的组织平衡诊断。GPU 默认允许 TF32。

```python
from fnit.connectome.mtnormalise import normalise_mrtrix_three_tissue

result = normalise_mrtrix_three_tissue(
    wm_sh=wm_raw,       # 输入；未归一化白质球谐图 [X,Y,Z,45]
    gm=gm_raw,          # 输入；未归一化灰质体积分数图 [X,Y,Z]
    csf=csf_raw,        # 输入；未归一化脑脊液体积分数图 [X,Y,Z]
    mask=norm_mask,     # 输入；原始 BET 掩膜两次侵蚀后的 bool 图
    affine=dwi_affine,  # 输入；DWI 体素中心到 RAS 世界毫米的 4×4 矩阵
)
wm_norm = result.wm    # 输出；归一化白质球谐图
```

原版等价命令：

```bash
mtnormalise wm_raw.mif wm_norm.mif gm_raw.mif gm_norm.mif \
  csf_raw.mif csf_norm.mif -mask norm_mask.mif -nthreads 8
```

| 归一化输出，全图 | 比较值数 | MAE | 最大绝对误差 | Pearson |
|---|---:|---:|---:|---:|
| WM 45 个系数 | 35,043,840 | 8.63×10⁻¹¹ | 1.19×10⁻⁷ | >0.999999999999995 |
| GM | 778,752 | 2.79×10⁻¹⁰ | 5.96×10⁻⁸ | >0.999999999999997 |
| CSF | 778,752 | 1.18×10⁻¹⁰ | 2.98×10⁻⁸ | >0.999999999999963 |

比较覆盖**全图**，不只看侵蚀掩膜内。全部差值小于 `2e-7`，但字节并非完全一致。FNIT 的 `field`、`accepted_mask`、`balance_factors` 是额外诊断，本次没有对应的官方输出逐项比较，不把它们称作已验证等价。

独立 MRtrix CPU 8 线程完整命令用时 **2.11 s**、峰值 RSS **336,364 KiB**。FNIT H100 已载入张量求解用时 **0.650 s**、PyTorch 峰值已分配 **0.461 GiB**；含 NIfTI 读取、参考比较和 JSON 的完整进程用时 **11.09 s**、RSS 峰值 **2,832,476 KiB**。CPU 候选核心另测 **2.497 s**。详细输入及参考参数见 [`tools/benchmark_connectome_mtnormalise.py`](../../tools/benchmark_connectome_mtnormalise.py)；其 `--wm`／`--gm`／`--csf` 为原始 FOD，`--wm-ref`／`--gm-ref`／`--csf-ref` 为官方归一化输出，`--mask` 为官方侵蚀掩膜，`--device` 为计算设备，`--output` 为**私有**完整报告路径。

下图是 [OpenNeuro ds004666](https://openneuro.org/datasets/ds004666) 的公开示例，展示相同原始 FOD 输入下的 WM 输出与差图。上表数值只对应私有 UKB 输入。

![公开数据归一化 FOD 示例](ds004666/mtnormalise_example.png)
