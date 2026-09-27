# PyTorch 结构连接矩阵

[完整 API、CLI 和验证](../../../docs/connectome/README.md) · [ds004666 阶段报告](../../../validation/connectome/ds004666/README.md)

`UKBConnectome` 从**已校正** DWI、bval、eddy 旋转后的 bvec、配对 skull-stripped T1、官方 FreeSurfer `recon-all` 的 `aparc+aseg.mgz`、可选 DWI BET 掩膜和固定整数 atlas，计算四张 region × region 矩阵。FreeSurfer 分割和 DWI 的 TOPUP/eddy 等前处理在此调用之外完成；LAS DWI 可在调用内运行 PyTorch BET；不会自动运行 SynthSeg 或生成 atlas。

```python
from fnit.connectome import UKBConnectome

result = UKBConnectome(device="cuda:0")(
    dwi="corrected_dwi.nii.gz",       # 输入：已校正 4D DWI
    bvals="corrected_dwi.bval",      # 输入：逐卷 b 值
    bvecs="eddy_rotated.bvec",       # 输入：eddy 旋转后的方向
    t1_brain="t1_brain.nii.gz",      # 输入：同次 T1 去脑图
    t1_segmentation="aparc+aseg.mgz",# 输入：官方 recon-all 分割
    atlas_dwi="atlas_dwi.nii.gz",    # 输入：整数 atlas
    brain_mask=None,                 # 输入：LAS DWI 自动 PyTorch BET
    n_seeds=10_000,                  # 输入：播种尝试数
    shell_bvals=None,                # 输入：自动聚类 shell
    response_mask=None,              # 输入：默认 DWI 响应掩膜
    fod_mask=None,                   # 输入：BET 掩膜膨胀两次
    normalise_mask=None,             # 输入：BET 掩膜侵蚀两次
    fa_map=None,                     # 输入：从 DWI 拟合 FA
    dwi_to_t1_world=None,            # 输入：自动 TorchFLIRT
    seed=0,                          # 输入：PyTorch 随机种子
)
count = result.matrices["count"]
```

必选 `n_seeds` 是尝试次数。可选 `dwi_to_t1_world` 为 DWI→T1 RAS-mm `4×4` 矩阵；省略时运行 TorchFLIRT 6-DOF/normmi。可用 `shell_bvals`、`response_mask`、`fod_mask`、`normalise_mask`、`fa_map` 固定参考条件。`ConnectomeResult.matrices` 包含 `count`、`sift2_fbc`、`mean_length`（mm）和 `mean_fa`；结果还保存 5TT/GMWMI、归一化 WM FOD、FA、世界毫米流线、BET 二值脑掩膜、逐流线 SIFT2 权重和几何变换。

响应、原始 FOD、mtnormalise、官方 FreeSurfer 输入的 5TT/GMWMI、固定轨迹 SIFT2 子阶段、精确 FA 与矩阵赋值已有各自配对验证。独立 iFOD2 风格追踪与 MRtrix 仍有分布差异；当前整链固定输入 seed 0 的数值、时间、显存和矩阵 CSV 见[当前验证](../../../validation/connectome/ds004666/README.md)。
