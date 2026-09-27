# 原 UKB 默认 Dhollander 响应：逐步对照

该阶段与[默认 DWI 掩膜对照](default_dwi_mask_stage_20260927.md)使用同一例真实校正 DWI 和同一份梯度。原脚本未给 `dwi2response dhollander` 提供 `-mask`，因此先调用 `dwi2mask legacy`；FNIT 在 `response_mask=None` 时执行同一顺序。参考 MRtrix 固定在 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`，旧计算节点的编译仅改变动态库符号绑定。原版程序只在独立基准中运行。

## 输入、输出与调用

`dwi2mask_legacy` 输入为 float32 DWI `[X,Y,Z,N]`、同序 b 值 `[N]` 和 shell 均值 `[S]`，返回 DWI 网格的 bool 掩膜 `[X,Y,Z]`。`estimate_mrtrix_dhollander` 随后接收同一 DWI、完整 MRtrix 梯度 `[N,4]`、响应文件的 shell 标签 `[S]` 与该掩膜；返回 shell 标签、float64 WM `[S,6]`、GM `[S,1]`、CSF `[S,1]` 响应，以及字典中的组织选择掩膜和 FA/SDM 中间图。字典的 11 个掩膜均为 DWI 网格 bool `[X,Y,Z]`，数值中间图为 float32 `[X,Y,Z]`，张量留在输入设备。

```python
from fnit.connectome import dwi2mask_legacy, estimate_mrtrix_dhollander, mrtrix_shell_centres

# signal: 同一已校正 float32 DWI，[X,Y,Z,N]，位于 CPU 或 CUDA。
# gradient: MRtrix 格式梯度，[N,4]，前3列是方向，第4列是 b 值。
# raw_shell_means: MRtrix 对 gradient 的 b 值聚类所得 shell 均值，[S]。
# response_shell_labels: 原版响应头使用的整数 shell 标签，[S]。
raw_shell_means, _, response_shell_labels, _ = mrtrix_shell_centres(
    grad_mrtrix=gradient,              # 每卷的方向和 b 值
)
brain_mask = dwi2mask_legacy(
    signal=signal,                     # 校正后 DWI
    bvalues=gradient[:, 3],           # 与 DWI 卷顺序一致的 b 值
    shell_bvalues=raw_shell_means,     # 用于分 shell 求均值
)
shells, wm, gm, csf, maps = estimate_mrtrix_dhollander(
    signal=signal,                     # 同一校正 DWI
    grad_mrtrix=gradient,             # 同一 MRtrix 梯度表
    shell_bvals=response_shell_labels, # 响应输出的 shell 标签
    brain_mask=brain_mask,            # 上一步生成的三维 bool 掩膜
)
# shells: [S]；wm/gm/csf: 各组织响应系数；maps: 组织掩膜和 FA/SDM 图。
```

原版独立参考命令：

```bash
mrconvert corrected_dwi.nii.gz corrected.mif -fslgrad rotated.bvec dwi.bval
dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt \
  -voxels voxels.mif -nthreads 8 -nocleanup
```

`wm.txt`、`gm.txt`、`csf.txt` 为带 shell 头的响应系数表；`voxels.mif` 与保留的 scratch 中间图用于逐步验证。`-nocleanup` 只为保存基准中间结果，原 UKB 算法和默认参数未变。FNIT 的[基准脚本](../../tools/benchmark_connectome_response_fod_dhollander.py)接收 DWI NIfTI、梯度文本、官方掩膜与响应/中间图路径，输出精度、时间和显存 JSON；其独立对照环节读取 MRtrix 产物，FNIT 正式调用不读取它们。

## 同输入实测

输入为 `104×104×72×105` 的真实 UKB 校正 DWI，三个 shell 分别有 5、50、50 卷。默认脑掩膜两边均为 156,270 个前景体素，XOR 为 0。固定原版中间输出后，FNIT 的 **11 个组织选择掩膜全部 XOR 0**。

| 输出 | 与原版的 MAE | 最大绝对误差 |
|---|---:|---:|
| WM 响应系数 | 7.04e−12 | 6.28e−11 |
| GM 响应系数 | 1.33e−10 | 2.55e−10 |
| CSF 响应系数 | 3.99e−11 | 8.73e−11 |
| 安全 SDM 图 | 5.68e−8 | 9.54e−7 |
| 张量 FA 图 | 2.13e−11 | 5.96e−8 |
| 两级单纤维指标图 | 4.00e−8 / 1.19e−7 | 2.38e−7 / 3.58e−7 |

MRtrix 完整命令（8 线程，保留中间图）耗时 27.04 s、最大 RSS 0.614 GiB；当前版 FNIT 在 nodecw10 的 CPU 8 线程已载入张量后的核心计算耗时 3.197 s；此前 H100 时间和进程 RSS 属于旧版求解器，已撤下。官方完整命令与 FNIT 核心计时边界不同，不据此宣称整链加速。聚合数据见[JSON](original_ukb_dhollander.public.json)；公开配对 DWI 的脑图及另一份完全一致的默认掩膜比较见[ds004666 示例](default_dwi_mask_stage_20260927.md)。受试者编号、私有路径、输入哈希、原始图像和逐体素数据仅保留在授权服务器。

该阶段只证明掩膜和 Dhollander 响应；全脑 FOD、mtnormalise、解剖、追踪、SIFT2 与最终连接矩阵需分别通过同输入检查。
