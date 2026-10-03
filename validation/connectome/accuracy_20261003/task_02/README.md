# DTI、response、CSD、mtnormalise：2026-10-03 精度修复

## 1. 功能与本轮策略

从已校正 DWI、梯度和同网格掩膜开始，计算 FA/主方向、Dhollander 三组织响应、MSMT-CSD 和三组织归一化。本轮修复两个成熟子函数的梯度解释及一个四分位数下标问题，保留现有 GPU 算法、Float64 求解、TF32 默认、两次 IWLS 更新与全部体素。

```mermaid
flowchart LR
    A[corrected DWI + FSL bval/bvec] --> B[Double FSL 到 RAS 梯度转换]
    B --> C[MRtrix Auto b缩放与方向单位化]
    A --> D[brain / response / FOD / normalise masks]
    C --> E[IWLS tensor 与 FA]
    D --> E
    C --> F[Dhollander response]
    D --> F
    F --> G[MSMT-CSD]
    G --> H[mtnormalise 正半数舍入四分位数]
    E --> I[后续追踪与矩阵]
    H --> I
```

### 成熟实现中特别修复的问题

- `fit_mrtrix_dhollander_tensor()`：原先直接使用 MRtrix 十位有效数字导出的方向。官方重新读入时会 Double 单位化。新增私有解释函数，不修改调用者输入。
- `pipeline._gradients()`：原先 Float32 读入，并对包含体素尺寸的 affine 直接做极分解。现在 Double 读入，先单位化 affine 每列，再 Double 极分解；FSL 右手图像的 x 翻转保留。非单位方向按官方默认 Auto 缩放 b 值；最大 `abs(log(norm²)) > 0.01` 时才触发，正常 eddy 输出的十例 b 值不变。
- `normalise_mrtrix_three_tissue()`：Python `round()` 在 `.5` 处取偶数，官方 `std::round()` 对正半数向上。修复下标计算，仍用原 15 次主更新和最多 7 次组织平衡更新。

当前 `b<50` 向量置零及零 diffusion 方向报错规则保留。官方可保留任意 b0 方向，并可仅警告非零 b 的零向量，这一更广义输入差异单列保留。FNIT 病态法方程的现有 QR fallback 也保留；官方 Eigen LLT 未检查失败状态，不能把其一次失败后的偶然数值当成稳定目标。

## 2. Python 调用、输入与输出

```python
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.connectome.response import fit_mrtrix_dhollander_tensor

device = torch.device('cuda:0')
corrected_dwi_path = Path('/data/sub-CON03/corrected_dwi.nii.gz')
gradient_mrtrix_path = Path('/data/sub-CON03/gradient_mrtrix.txt')
brain_mask_path = Path('/data/sub-CON03/brain_mask.nii.gz')
corrected_dwi_image = nib.load(corrected_dwi_path)
# 读写用 nibabel；DWI 保持 Float32，梯度与回归运算用 Float64。
corrected_dwi = torch.as_tensor(
    corrected_dwi_image.get_fdata(dtype=np.float32), device=device,
)
gradient_mrtrix = torch.as_tensor(
    np.loadtxt(gradient_mrtrix_path), dtype=torch.float64, device=device,
)
brain_mask = torch.as_tensor(
    np.asarray(nib.load(brain_mask_path).dataobj) > 0, device=device,
)
fractional_anisotropy, principal_direction = fit_mrtrix_dhollander_tensor(
    signal=corrected_dwi,
    grad_mrtrix=gradient_mrtrix,
    safe_mask=brain_mask,
    batch_size=4096,  # 一批最多求解的体素数；不裁减掩膜。
)
```

|名称|格式与含义|
|---|---|
|`signal`|Float32 `[X,Y,Z,N]`，topup/eddy 校正后的完整 DWI，CPU/CUDA|
|`grad_mrtrix`|`[N,4]`，RAS 方向 xyz 和 b 值；内部转换 Float64，Auto 解释|
|`safe_mask`|bool `[X,Y,Z]`，与 DWI 轴、体素中心和 affine 对齐；不重新采样|
|`batch_size`|正整数，默认 4096；改变分批不改变估计器或迭代次数|
|FA 输出|Float32 `[X,Y,Z]`，掩膜外 0；没有正信号的体素沿用 NaN 行为|
|主方向输出|Float32 `[X,Y,Z,3]`，RAS 单位向量；正负方向等价，掩膜外 0|
|response 输出|shell 中心及 WM/GM/CSF 响应系数、实际组织选择掩膜|
|CSD 输出|WM SH `[X,Y,Z,45]`，GM/CSF `[X,Y,Z]`，默认 lmax 8|
|normalise 输出|上述归一化图、Float32 field、bool accepted mask、Double 三个平衡因子|

响应、CSD 与归一化的完整参数说明沿用 [`README_official_chain.md`](../../tenraw_20261002/task_02/README_official_chain.md) 和相应函数 docstring。本轮未更改这些函数的公共参数。

## 3. 命令行与复现实验

正式入口仍为 `fnit UKBConnectome_pipeline`；无需新增参数即可采用修复。

```bash
fnit UKBConnectome_pipeline \
  --dwi /data/sub-CON03/corrected_dwi.nii.gz \
  --bvals /data/sub-CON03/dwi.bval \
  --bvecs /data/sub-CON03/eddy_rotated.bvec \
  --freesurfer-subject-dir /data/freesurfer/sub-CON03 \
  --atlas fs-aparc --n-seeds 100000 --device cuda:0 \
  --output-dir /data/connectome/sub-CON03
```

基准脚本均为验证代码，不属于生产运行时。`--output` 必须为新目录；原始数据和旧结果只读。

|脚本|参数与用途|
|---|---|
|`benchmark_tensor_gradient_fix.py`|`--case-map` 实际十例 completed consumer/producer 映射；`--baseline`/`--candidate` 冻结 response 文件；`--cases` 病例列表；`--paired-cases` 默认 CON01/03；`--abba-rounds` 默认 4；`--output` 新输出目录。完整 tensor 捕获在独立调用，ABBA 使用未插桩原函数|
|`prepare_formal_tensor_reference.py`|`--old-root` 已完成正式输出根目录；`--pipeline` 必须为冻结旧版本文件；`--mrtrix-bin` CPU 官方 bin；`--output` 新 reference。验证实际报告及准备输入 SHA 后执行官方 CPU 命令|
|`diagnose_formal_gradient_fix.py`|`--reference-root` 上述成功 reference；`--baseline-response`/`--candidate-response`、`--baseline-pipeline`/`--candidate-pipeline` 为冻结文件；`--device` 默认 CPU；`--output` 新目录。实际 GPU table 若与 CPU oracle 不同，不借用该 oracle|
|`diagnose_normalise_quartiles.py`|`--source` 必须为冻结旧 mtnormalise；`--case-map` 实际 consumer 映射；`--device` 默认 CPU；`--output` 新目录。只比较四例完整官方 CSD 输入|
|`diagnose_tensor_arithmetic.py`|算术次序候选诊断，`--contract` 实际 completed consumer；`--source` 冻结旧 response；`--device`/`--batch-size`/`--threads`；`--modes` 候选；`--output` 新目录。basis/RHS 候选已拒绝，未进入产品|
|`make_precision_images.py`|`--reference-root`/`--diagnostic-root` 为同例实际输出；`--output` 新 PNG；统一误差色标，不以模拟数据替代|

GPU 评测遵守共享 flock，固定已核实 GPU UUID，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。监测 Torch allocated/reserved 和 NVML 本进程；tensor 运算不启动 CUDA 子进程。共享负载、采样间隔和失败采样保留在报告。

## 4. 对应官方命令

```bash
# 同一真实 corrected DWI、旋转 bvec、bval 和掩膜；本轮 reference 在 nodecw10 CPU8。
mrinfo corrected_dwi.nii.gz -fslgrad eddy_rotated.bvec dwi.bval \
  -export_grad_mrtrix gradient_mrtrix.txt -nthreads 8
dwi2tensor corrected_dwi.nii.gz tensor.nii.gz \
  -fslgrad eddy_rotated.bvec dwi.bval -mask brain_mask.nii.gz -nthreads 8
tensor2metric tensor.nii.gz -fa fa.nii.gz -vector direction.nii.gz \
  -modulate none -mask brain_mask.nii.gz -nthreads 8
dwi2response dhollander corrected_dwi.nii.gz wm.txt gm.txt csf.txt \
  -fslgrad eddy_rotated.bvec dwi.bval -mask response_mask.nii.gz -nthreads 8
dwi2fod msmt_csd corrected_dwi.nii.gz wm.txt wm_raw.nii.gz \
  gm.txt gm_raw.nii.gz csf.txt csf_raw.nii.gz \
  -fslgrad eddy_rotated.bvec dwi.bval -mask fod_mask.nii.gz -nthreads 8
mtnormalise wm_raw.nii.gz wm_norm.nii.gz gm_raw.nii.gz gm_norm.nii.gz \
  csf_raw.nii.gz csf_norm.nii.gz -mask normalise_mask.nii.gz -nthreads 8
```

归档中十位梯度导出的命令显式用了 `-bvalue_scaling no`；后续 `dwi2tensor -grad` 仍按 Auto 再解释。正式路径本轮 reference 的 `dwi2tensor -fslgrad` 使用默认 Auto。这是两个具体 reference 的实际参数，不用于推断旧正式 FNIT 私有入口的科学定义。

## 5. 真实精度、时间与脑图

### 正式调用输入，CPU 同输入 tensor 对照

两例均读取已完成正式 FNIT corrected DWI、真实旋转梯度和其 brain mask；SHA 与正式 `raw_bids_wall.json`/`gpu_report.json` 绑定。官方 CPU reference 独立执行，保存 bin SHA/version、argv、所有输入输出 SHA。主表包含全部掩膜，不删除负信号体素。

|实际正式输入|原 FNIT 对官方 FA max|修复后 FA max|原/新 `abs(error)>1e-5` 体素|修复后 P99|非有限值错配|
|---|---:|---:|---:|---:|---:|
|CON03，98,625 体素|0.002352715|0，逐值相同|172 / 0|0|0|
|CON10，115,626 体素|1.015095688|0.0000120401|183 / 1|0|0|

仅冻结同一 Float32 caller table、单独修 response，CON03 FA 逐值相同，CON10 最大差 `1.90735e-6`。这将 tensor 解释和梯度导入两个差异分别验证。原 Float32 world 方向对官方单位化导出表最大差 `1.17e-7 / 1.53e-7`；Double 导入为 `5.89e-11 / 5.23e-11`，受官方十位文本导出精度限制。

![CON10 相同实际正式 corrected 输入的 FA 脑图](formal_CON10_FA_precision.png)

### 官方 own-corrected 输入，真实 GPU 算子诊断

|病例|原 FA max|仅单位化修复 FA max|结论|
|---|---:|---:|---|
|CON03|9.059906e-6|0|本次输入没有复现旧历史 0.046；实际当次 official tensor 严格绑定|
|CON10|0.2456678|0.0003737211|只剩 1 个大于 1e-5 的病态尾部体素；未填零、加正则或删数据|

basis 和 RHS 乘法次序候选在 CON10 GPU 使最大误差变为约 `0.00077`，未采用。此前 CON07 历史方向 41.63° 未在新 trace 重现，不用其他输入的 trace 解释旧结果。

### mtnormalise，四例完整固定官方 CSD 输入

|病例|官方零起点 quartile 下标|WM 对官方最大差|accepted mask XOR|修复前/后 CPU s，单轮诊断|
|---|---|---:|---:|---:|
|CON03|20106 / 60317|5.960464e-8|0|0.248 / 0.307|
|CON04|20708 / 62123|5.960464e-8|0|0.233 / 0.254|
|CON07|22431 / 67292|1.192093e-7|0|0.231 / 0.273|
|CON11|21521 / 64562|1.192093e-7|0|0.230 / 0.229|

四例 WM/GM/CSF/field/accepted-mask 修复前后逐位相同；CON04/07 平衡因子仅约 `2e-15 / 1.92e-13` 变化。此修复保证 C++ 下标语义，不声称该四例最终图精度提高或单轮速度提高。官方平衡因子文本为有限有效位数，报告中的几 e-6 差不能解释为真实求解误差；全场 field 最大差保留在 JSON，不只截取掩膜。

本轮当前 CPU focused/既有 response、CSD、pipeline 回归 `27 passed, 1 CUDA skipped`。正式十例 GPU tensor、CON01/03 四轮 ABBA、最新 CUDA 回归仍排共享锁；本报告不将排队或上述短诊断当作已完成端到端性能验收。全链时间、精度和严格 `<20e9` 三类显存验收由总控制实测补齐。

数据为新下载公开 OpenNeuro ds001226 十例 CON01/03/04/05/06/07/08/09/10/11；许可证与原始下载出处见前一轮实际来源记录。完整数组保留新服务器目录，不将受试者体积复制进仓库；提交报告、脚本及公开脑图。旧 frozen config、旧 producer 和旧路径未补写。

## 6. 更新与 benchmark 记录

- 基线：`7af34e6d072e843fb2558c931bb2781f1d4b0be9`，保留其 CSD 有序 active-set 缓存与 batch4096。
- 2026-10-03：实际 bin `3.0.3-103-g026e850d` 与安装目录陈旧 Git HEAD 分开审计，按精确 commit 核对。
- 同日：拒绝 basis/RHS 数学相同但尾部变坏的算术候选；只采用已证实梯度解释修复。
- 同日：闭合 CON03/10 正式 caller CPU 输入，修复私有 Double FSL 导入；四例 mtnormalise 半数舍入语义修复并保存全场结果。
- 同日：GPU 十例和成对速度/显存仍待锁后实际完成，最终结果在后续证据提交追加，保留本轮状态。

## 7. 参考文献与原实现

- [MRtrix3 实际精确 commit 026e850d171ec2a12f09865d31b8332d23d7ecf6](https://github.com/MRtrix3/mrtrix3/tree/026e850d171ec2a12f09865d31b8332d23d7ecf6)。安装目录 Git HEAD 不作为已编译来源证据。
- [gradient.cpp：FSL 导入、Double 单位化、Auto b scaling](https://github.com/MRtrix3/mrtrix3/blob/026e850d/core/dwi/gradient.cpp)，[NIfTI affine 列单位化](https://github.com/MRtrix3/mrtrix3/blob/026e850d/core/file/nifti_utils.cpp)。
- [dwi2tensor.cpp](https://github.com/MRtrix3/mrtrix3/blob/026e850d/cmd/dwi2tensor.cpp)，[tensor.h：FA 与主方向](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tensor.h)。
- [Dhollander](https://github.com/MRtrix3/mrtrix3/blob/026e850d/lib/mrtrix3/dwi2response/dhollander.py)，[dwi2fod](https://github.com/MRtrix3/mrtrix3/blob/026e850d/cmd/dwi2fod.cpp)，[mtnormalise](https://github.com/MRtrix3/mrtrix3/blob/026e850d/cmd/mtnormalise.cpp)。
- Tournier JD et al. MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. *NeuroImage*, 2019, 202:116137.
- Jeurissen B et al. Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data. *NeuroImage*, 2014, 103:411–426.
- Dhollander T et al. Unsupervised 3-tissue response function estimation from single-shell or multi-shell diffusion MR data without a co-registered T1 image. ISMRM Workshop, 2016；single-fibre 改进 ISMRM 2019。
- Raffelt D et al. Bias field correction and intensity normalisation for quantitative analysis of apparent fibre density. ISMRM 2017；Dhollander T et al. Mtnormalise ISMRM 2021。

精确上游源码只存独立 oracle 目录，不随本项目复制发布。生产函数不调用 MRtrix/FSL/FreeSurfer 或其包装包。
