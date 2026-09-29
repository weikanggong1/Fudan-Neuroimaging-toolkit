# PyTorch 结构连接矩阵

[完整 API、CLI 和验证](../../../docs/connectome/README.md) · [ds004666 阶段报告](../../../validation/connectome/ds004666/README.md)

`UKBConnectome` 从已校正 DWI、bval、eddy 旋转后的 bvec 和已完成的 `recon-all` subject 目录开始，自动读取 `brain.mgz`、`aparc+aseg.mgz`，构建 84 节点 `fs-aparc` atlas，并计算四张 region × region 矩阵。TOPUP/eddy 与官方 `recon-all` 由调用方提前完成；包内不运行它们。独立 iFOD2/ACT 追踪仍处于与 MRtrix 的分布对照阶段。

`--atlas` 另支持原 UKB 七套皮层+Tian 组合。Glasser 的 32k fsLR→164k fsaverage 标签重采样使用主页 conda 环境中的 Connectome Workbench；后续原生表面与 T1 ribbon 投影由 PyTorch 完成。[真实 Glasser 逐体素比较、参数和脑图](../../../validation/connectome/ds004666/atlas_glasser_20260929.md)记录原版 360 节点皮层标签与 FNIT 完全一致，Tian 的 SynthMorph 路线则与原 UKB FNIRT 路线分别解释。

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
    compile_arc=False,                         # 输入：True 时编译 CUDA iFOD2 圆弧概率核
)
count = result.matrices["count"]              # 输出：84×84 流线计数
nodes = result.nodes                          # 输出：84 个矩阵行列定义
```

必选 `n_seeds` 是尝试次数。可选 `dwi_to_t1_world` 为 DWI→T1 RAS-mm `4×4` 矩阵；省略时运行 TorchFLIRT 6-DOF/normmi。可用 `shell_bvals`、`response_mask`、`fod_mask`、`normalise_mask`、`fa_map` 固定参考条件。`ConnectomeResult.nodes` 提供 `index`、`original_label`、`hemisphere`、`name`，矩阵行列严格按 index 排列。`ConnectomeResult.matrices` 包含 `count`、`sift2_fbc`、`mean_length`（mm）和 `mean_fa`；结果还保存 5TT/GMWMI、归一化 WM FOD、FA、世界毫米流线、BET 二值脑掩膜、逐流线 SIFT2 权重和几何变换。

响应、原始 FOD、mtnormalise、官方 FreeSurfer 输入的 5TT/GMWMI、固定轨迹 SIFT2 子阶段、精确 FA 与矩阵赋值已有各自配对验证。追踪球谐函数查表的[固定单弧基准](../../../validation/connectome/ds004666/ifod2_single_arc_20260929.md)、连续初始方向的[真实 FOD 基准](../../../validation/connectome/ds004666/ifod2_initial_direction_20260929.md)和[ACT 种子判定基准](../../../validation/connectome/ds004666/ifod2_act_seed_20260929.md)分别记录数值、时间与图。传播中的独立流线群体仍有差异，最终矩阵未全面进入已测官方重复范围。旧版整链固定输入 seed 0 的矩阵 CSV 见[历史验证](../../../validation/connectome/ds004666/README.md)。

[新 100k 真实输入验证](../../../validation/connectome/ds004666/tracking_100k_matrices_20260929.md)已比较三次官方与一次 FNIT 的轨迹、SIFT2 和四矩阵：count 相对 L1 有 2/3 组跨软件配对进入官方自身范围，长度、端点和 TDI 的跨软件差异仍略超范围。当前该规模全链 Torch 峰值 2.473 GiB，追踪速度仍明显慢于独立 MRtrix CPU 参考。

可选的[CUDA 圆弧编译核实测](../../../validation/connectome/ds004666/tracking_compile_20260929.md)在相同真实输入的另一次 100k 运行中追踪 782.49 s、全链 Torch 峰值 2.468 GiB。首次编译计入；共享 GPU 两次负载不同。输出结构不变，但独立轨迹的长度、端点和 TDI 仍未进入官方三次运行的随机范围。

[七套原 UKB atlas 的 100k 矩阵验证](../../../validation/connectome/ds004666/seven_atlas_100k_20260929.md)复用同一 TCK 或 FNIT 独立 TCK 生成 84–1054 节点的四矩阵。同一官方 TCK 和逐轨数值时七套 count 矩阵均逐值一致；独立追踪后仍有随机范围外的指标。报告提供每套行列节点表、矩阵压缩文件、计时和连接图。

[真实 b0/T1 无 Surfa 自动配准核对](../../../validation/connectome_registration_no_surfa_20260928/README.md)只覆盖配准矩阵；完整连接组未因该迁移重新验收。
