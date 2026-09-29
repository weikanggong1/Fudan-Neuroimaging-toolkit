# MNI152 非线性变换和检查图

[返回 recon-all](README.md) · [真实 T1 配对结果](../../validation/recon_all/python_gpu_port/mni_nonlinear_real_20260929.json)

标准单 T1 流程先以 PyTorch SynthMorph affine 生成个体 `aff.lta`，再用 PyTorch 的 deform 模型优化同一裁剪 T1 到 MNI152 的非线性变换。Conda 中从固定 FreeSurfer 源码编译的 `mri_warp_convert`、`mri_ca_register` 和 `mri_convert` 将位移转为 FreeSurfer warp、求逆，并写出最近邻重采样检查图。它们不从系统 FreeSurfer 安装目录调用。

`run_mni_nonlinear_chain(...)` 的参数如下。`subject_dir` 指向已含 `mri/orig.mgz`、`mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz` 与 `aff.lta` 的被试目录；前者是 1 mm conform 网格，裁剪 NIfTI 保留其 scanner RAS。`weights_dir` 必须有经大小和 SHA-256 校验的 `synthmorph.deform.3.h5`；`assets_dir` 必须有同一固定 MNI152 版本的裁剪及完整 1 mm 模板。`warp_convert`、`ca_register`、`mri_convert` 分别是当前 Conda 环境中这三个源码构建程序的绝对路径。`device` 指定 PyTorch 设备，默认 `cpu`；`threads` 指定 CPU 线程数，默认 4。需在运行前设置本人有效的 `FS_LICENSE` 路径。

```python
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain

nonlinear_outputs = run_mni_nonlinear_chain(
    subject_dir="/data/subjects/sub01",  # 已生成 orig、裁剪 T1 和 affine LTA 的被试目录
    weights_dir="/data/fnit-weights",  # 已校验的 SynthMorph deform 权重目录
    assets_dir="/data/fnit-assets",  # 已校验的裁剪及完整 MNI152 模板目录
    warp_convert="/data/conda/envs/fnit/bin/mri_warp_convert",  # 位移到 FS warp
    ca_register="/data/conda/envs/fnit/bin/mri_ca_register",  # 求逆变换
    mri_convert="/data/conda/envs/fnit/bin/mri_convert",  # 生成最近邻检查图
    device="cpu",  # 非线性模型设备；默认 CPU 可避免挤占重建 GPU 显存
    threads=4,  # CPU 算子线程数
)
print(nonlinear_outputs["forward"])  # 目标 MNI152 网格上的前向变换绝对路径
```

返回字典还含 `inverse`、`check`、`model` 和 `device`。三个正式输出分别位于被试的 `mri/transforms/synthmorph.1.0mm.1.0mm/`：前向 `warp.to.mni152.1.0mm.1.0mm.nii.gz` 的结构为 `(193,229,193,1,3)`，逆向同名加 `.inv` 的结构为 `(256,256,256,1,3)`；两者为 NIfTI 位移向量，intent 1006。`test.nii.gz` 是完整 MNI152 的 `(193,229,193)` 最近邻检查图。`tmp/deform.mgz` 及两个 LTA 是中间文件。输入缺失、程序失败或网格不符时抛出异常，主流程在运行 JSON 中记录失败阶段。

对应官方 8.2 链的主要命令依次是 `mri_synthmorph -m deform -i aff.lta -t deform.mgz invol.crop.nii.gz mni152.1.0mm.cropped.nii.gz`、`mri_warp_convert --inras deform.mgz --insrcgeom invol.crop.nii.gz --outfswarp warp.to.mni152.1.0mm.1.0mm.nii.gz --vg-thresh 1e-4 --lta1-inv reg.crop-to-invol.lta --lta2 reg.1.0mm.cropped.to.1.0mm.lta`、`mri_ca_register -invert-and-save 前向变换 逆向变换`、`mri_convert -rt nearest orig.mgz -at 前向变换 test.nii.gz`。FNIT 用 nibabel 几何生成两个等价的 LTA，避免让这一阶段依赖系统安装的模板目录。[固定版本的 mri_synthmorph](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_synthmorph/mri_synthmorph)、[mri_warp_convert](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_warp_convert/mri_warp_convert.cpp)可查阅。

去标识真实 `sub-01` 的阶段原型在 CPU 上完成 PyTorch deform 推理用时 91.50 s；相同官方存档与 FNIT 输出的前向位移相关性超过 0.9999999999999、平均绝对差 0.0000084 mm。逆向位移相关性超过 0.9999999999、最大局部差 0.233 mm；检查图非零区域相关性 0.9999973，有 127 个不同体素。详细值见[机器记录](../../validation/recon_all/python_gpu_port/mni_nonlinear_real_20260929.json)。该阶段结果是同一真实 T1 的阶段比较，不能替代全流程 138 项验收。

参考文献：Hoffmann M, et al. SynthMorph: learning contrast-invariant registration without acquired images. *IEEE Trans Med Imaging*. 2022;41:543–558. [doi:10.1109/TMI.2021.3116879](https://doi.org/10.1109/TMI.2021.3116879)。
