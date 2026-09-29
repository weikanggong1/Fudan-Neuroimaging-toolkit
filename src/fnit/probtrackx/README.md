# ProbtrackX 源码目录

`TorchProbtrackX` 使用 BEDPOSTX 方向后验进行 diffusion 网格上的体积概率追踪。MNI 网格掩膜先通过本包 `TorchConvertWarp`、`TorchInvWarp` 和 `TorchApplyWarp` 自动映射到 diffusion 网格，再输出 seed→voxel 密度、voxel→voxel 稀疏矩阵、voxel→ROI 计数图和有向 ROI×ROI 矩阵。CPU 路径使用 PyTorch，CUDA 步进使用 Triton；无额外路径约束的 CUDA 结果由 Numba 汇总计数。运行时不调用 FSL。

## Python 单被试示例

```python
from fnit import TorchProbtrackX

tracker = TorchProbtrackX(
    device="cuda:0",  # 计算设备；可改为 "cpu"
    nsamples=5000,  # 每个种子体素的轨迹数
    nsteps=2000,  # 双向总步数；必须为偶数
    steplength=0.5,  # 每步长度，单位 mm
    cthr=0.2,  # 方向点积阈值
    fibthresh=0.01,  # 纤维分数阈值
    batch_size=16384,  # 每批并行轨迹数
    seed=12345,  # 随机数种子
)
result = tracker.run(
    samples_dir="/absolute/path/subject.bedpostX",  # 输入：BEDPOSTX 后验目录
    output_dir="/absolute/path/probtrackx_result",  # 输出：单被试结果目录
    seed="/absolute/path/seed.nii.gz",  # 输入：同网格非空 3D seed mask
    regions=None,  # 单 seed 模式必须为 None
    mask=None,  # 可选追踪 mask；None 使用 nodif_brain_mask.nii.gz
    mni_reference=None,  # MNI 掩膜的参考网格；diffusion seed 可省略
    diff2struct_mat=None,  # 仅外部 diffusion→T1 + T1→MNI 组合模式需要
    struct2mni_warp=None,  # 仅外部两段 FSL 配准模式需要
    overwrite=False,  # 是否覆盖已有输出
)
```

`result.mni_to_diffusion_dir` 指向自动生成的组合场、逆场与 diffusion 掩膜目录；没有 MNI 输入时为 `None`。`result.paths` 指向 `fdt_paths.nii.gz`，`result.waytotal` 指向有效轨迹计数；启用相应选项后还会返回 matrix1/2/3、seed-to-target 和 network 文件路径。追踪图和映射后的掩膜使用 BEDPOSTX diffusion 网格；组合前向场使用 MNI 参考网格。

输入 MNI 掩膜时，优先传 `dmri_pipeline_dir="/data/subject_tbss"` 或 `"/data/subject_mmorf"`，即本包 dMRI pipeline 单被试输出根。`registration_backend="auto"` 根据 `registration/` 内的 warp 选择 TBSS 或 MMORF；TBSS 系数已含 affine，MMORF 使用 `mmorf_warp.nii.gz` 加 `dti_FA_to_MNI_affine.mat`。也可传已含 affine 的 `diff2mni_warp`，或成对传 `diff2struct_mat`、`struct2mni_warp`。完整带参数名示例、官方三步命令和真实 DWI 对照见[功能说明](../../../docs/probtrackx/README.md)。

内部 `accumulate_paths` 接收前、后半轨迹的 `[轨迹, 步数]` 一维体素索引数组（`-1` 表示终止）、ROI 查找表、步长和最短距离阈值，并就地更新路径密度、长度、网络矩阵和有效轨迹计数数组；没有单独返回文件。它用于无额外路径约束的 CUDA `--opd`、`--pd --ompl`、`--network` 模式，对应原版 `probtrackx2_gpu` 的同名选项。当前真实数据速度和精度见[性能记录](../../../validation/probtrackx/report.performance.public.json)及[配对报告](../../../validation/probtrackx/report.current.latest.public.json)。

## 命令行与原软件对应

```bash
fnit probtrackx --samples-dir /absolute/path/subject.bedpostX \
  --seed /absolute/path/seed.nii.gz \
  --output-dir /absolute/path/fnit_result \
  --device cuda:0 --nsamples 5000 --nsteps 2000

"$FSLDIR/bin/probtrackx2" \
  -s /absolute/path/subject.bedpostX/merged \
  -m /absolute/path/subject.bedpostX/nodif_brain_mask.nii.gz \
  -x /absolute/path/seed.nii.gz \
  --dir=/absolute/path/fsl_result --forcedir --opd \
  -P 5000 -S 2000 --steplength=0.5
```

FNIT 的 `samples_dir` 同时定位 `merged_th/ph/f` 后验和默认 mask；`seed` 对应 FSL `-x`；`output_dir` 对应 `--dir`；`nsamples`、`nsteps`、`steplength` 对应 `-P`、`-S` 和 `--steplength`。

真实 DWI 对照已覆盖默认密度、长度加权、matrix1/2/3、seed→ROI 和五区网络，但双方随机数流不同，不能要求逐轨迹一致。公开仓库保存汇总 JSON 和一张[去标识五区长度加权连接矩阵图](../../../docs/probtrackx/figures/probtrackx_real_ukb_network_pd_ompl.png)；原始 DWI、后验、seed、逐体素结果和完整文本矩阵仍在授权服务器。当前证据只有一例稀疏五区网络，不能替代多病例或全脑分区验证。

完整输入、全部参数、输出树、官方命令、差异表和逐项 benchmark 见[功能说明](../../../docs/probtrackx/README.md)。
