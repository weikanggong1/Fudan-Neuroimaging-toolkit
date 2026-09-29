# PyTorch 结构连接矩阵

[完整 API、CLI 和验证](../../../docs/connectome/README.md) · [ds004666 阶段报告](../../../validation/connectome/ds004666/README.md)

`UKBConnectome` 从已校正 DWI、bval、eddy 旋转后的 bvec 和已完成的 `recon-all` subject 目录开始，自动读取 `brain.mgz`、`aparc+aseg.mgz`，构建 84 节点 `fs-aparc` atlas，并计算四张 region × region 矩阵。TOPUP/eddy 与官方 `recon-all` 由调用方提前完成；包内不运行它们。独立 iFOD2/ACT 追踪仍处于与 MRtrix 的分布对照阶段。

```python
from fnit.connectome import UKBConnectome

result = UKBConnectome(device="cuda:0")(
    dwi="corrected_dwi.nii.gz",                 # 输入：已校正四维 DWI
    bvals="corrected_dwi.bval",                # 输入：逐卷 b 值
    bvecs="eddy_rotated.bvec",                 # 输入：eddy 旋转后的方向
    freesurfer_subject_dir="subjects/sub-01",  # 输入：已完成的 recon-all subject 目录
    atlas="fs-aparc",                     # 输入：84 节点 Desikan atlas
    brain_mask=None,                           # 输入：LAS DWI 自动 PyTorch BET
    n_seeds=10_000,                            # 输入：播种尝试数
    shell_bvals=None,                          # 输入：自动聚类 shell
    response_mask=None,                        # 输入：默认 DWI 响应掩膜
    fod_mask=None,                             # 输入：BET 掩膜膨胀两次
    normalise_mask=None,                       # 输入：BET 掩膜侵蚀两次
    fa_map=None,                               # 输入：从 DWI 拟合 FA
    dwi_to_t1_world=None,                      # 输入：自动 TorchFLIRT
    seed=0,                                    # 输入：PyTorch 随机种子
)
count = result.matrices["count"]              # 输出：84×84 流线计数
nodes = result.nodes                          # 输出：84 个矩阵行列定义
```

必选 `n_seeds` 是尝试次数。可选 `dwi_to_t1_world` 为 DWI→T1 RAS-mm `4×4` 矩阵；省略时运行 TorchFLIRT 6-DOF/normmi。可用 `shell_bvals`、`response_mask`、`fod_mask`、`normalise_mask`、`fa_map` 固定参考条件。`ConnectomeResult.nodes` 提供 `index`、`original_label`、`hemisphere`、`name`，矩阵行列严格按 index 排列。`ConnectomeResult.matrices` 包含 `count`、`sift2_fbc`、`mean_length`（mm）和 `mean_fa`；结果还保存 5TT/GMWMI、归一化 WM FOD、FA、世界毫米流线、BET 二值脑掩膜、逐流线 SIFT2 权重和几何变换。

响应、原始 FOD、mtnormalise、官方 FreeSurfer 输入的 5TT/GMWMI、固定轨迹 SIFT2 子阶段、精确 FA 与矩阵赋值已有各自配对验证。追踪球谐函数查表的[固定单弧基准](../../../validation/connectome/ds004666/ifod2_single_arc_20260929.md)及连续初始方向的[五次真实 FOD 基准](../../../validation/connectome/ds004666/ifod2_initial_direction_20260929.md)分别记录数值、时间与图。传播和 ACT 仍有差异，最终矩阵未进入已测官方重复范围。旧版整链固定输入 seed 0 的矩阵 CSV 见[历史验证](../../../validation/connectome/ds004666/README.md)。

[真实 b0/T1 无 Surfa 自动配准核对](../../../validation/connectome_registration_no_surfa_20260928/README.md)只覆盖配准矩阵；完整连接组未因该迁移重新验收。
