# 单被试结构连接矩阵

`UKBConnectome` 读取已校正的四维 DWI、逐体积 b 值、eddy 旋转后的 b 向量和配对 T1w，输出 `count`、`sift2_fbc`、`mean_length`、`mean_fa` 四张脑区矩阵。FOD 估计、概率追踪及 SIFT2 权重是 PyTorch 实现；输入不包括 TOPUP/EDDY 等前处理。自动 DWI→T1 配准使用包内 6 DOF / normmi TorchFLIRT 和 NiBabel 图像载入，不导入 Surfa。

```python
from fnit.connectome import UKBConnectome

model = UKBConnectome(
    device="cuda:0",                         # PyTorch 设备
    synthseg_weights=None,                   # SynthSeg 权重目录或文件；None 用默认配置
)
result = model(
    dwi="sub-01_desc-preproc_dwi.nii.gz",     # 已校正四维 DWI
    bvals="sub-01_dwi.bval",                  # 每个 DWI 体积的 b 值
    bvecs="sub-01_desc-eddyRotated_dwi.bvec",  # 旋转后的方向
    t1="sub-01_T1w.nii.gz",                  # 配对 T1w
    atlas_dwi="sub-01_space-dwi_atlas.nii.gz",  # DWI 空间脑区标签
    t1_segmentation=None,                    # None 时自动运行 SynthSeg
    segmentation_source="synthseg",          # 标签来源
    dwi_to_t1_world=None,                    # None 时自动估计 DWI→T1 变换
    n_seeds=10_000,                          # 播种尝试次数
    seed=0,                                  # 随机种子
)
count = result.matrices["count"]           # 脑区间纤维计数，K×K
```

`result` 还包含 `region_labels`、`atlas`、`tissues`、`wm_sh`、`fa`、`tractogram`、`sift2_weights`、`dwi_affine`、`atlas_affine` 和 `dwi_to_t1_world`。输出文件结构、CLI 每个输入含义、官方软件命令及各阶段真实数据基准见[中文功能说明](../../../docs/connectome/README.md)。当前[无 Surfa 自动配准核对](../../../validation/connectome_registration_no_surfa_20260928/README.md)只验证配准矩阵；完整连接组没有因此重新验收。
