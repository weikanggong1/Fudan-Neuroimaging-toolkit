# 原 UKB 流程：b0→T1 刚性配准同输入核对

## 功能与来源

FNIT 的 `TorchFLIRT(device="cuda:0", angular_search=True, dof=6, cost="normmi")` 对去脑的平均 b0 和 T1 做 6 自由度刚性配准。它在 8 mm 角度搜索阶段使用 FSL 默认的相关比（`searchcost=corratio`），在 4/2/1 mm 精化阶段使用 `-cost normmi` 指定的归一化互信息。该区别来自本仓库保存的 FSL FLIRT 2111.2 源码：`globaloptions.h` 分别默认初始化两个代价函数，`globaloptions.cc` 的 `-cost` 只修改主代价函数，`flirt.cc` 在搜索结束后切回主代价函数。FNIT 运行时不启动 FSL。

原始 [UKB-connectomics 脚本](https://github.com/sina-mansour/UKB-connectomics/blob/ec73ab75c868060e3ecacd19735b7a347ad226a5/scripts/bash/probabilistic_tractography_native_space.sh) 先从 `data_ud.nii.gz` 与梯度提取并平均 b0，再运行 BET，随后运行 FLIRT。以下是同一过程的通用命令模板；此处的文件名仅表示输入输出角色：

```bash
# 输入：四维、已校正 DWI；bvecs/bvals 是对应梯度。输出：带梯度元数据的 dwi.mif。
mrconvert data_ud.nii.gz dwi.mif -fslgrad bvecs bvals -datatype float32 -strides 0,0,0,1
# 输入：dwi.mif 的 b=0 体积。输出：单帧平均 b0。
dwiextract dwi.mif - -bzero | mrmath - mean -axis 3 dwi_meanbzero.mif
mrconvert dwi_meanbzero.mif dwi_meanbzero.nii.gz
# 输入：平均 b0。输出：去脑 b0 和二值脑掩膜。
bet dwi_meanbzero.nii.gz dwi_meanbzero_brain.nii.gz -m -R -f 0.2 -g -0.05
# 输入：去脑 b0 与去脑 T1。输出：FSL scaled-mm 坐标系中 b0→T1 的 4×4 矩阵。
flirt -in dwi_meanbzero_brain.nii.gz -ref T1_brain.nii.gz -cost normmi -dof 6 -omat DWI_to_T1_FSL.mat
# 仅供独立比较：固定上一行的 FSL 矩阵，将 b0 重采样到 T1 网格。
flirt -in dwi_meanbzero_brain.nii.gz -ref T1_brain.nii.gz -applyxfm -init DWI_to_T1_FSL.mat -out b0_in_T1_FSL.nii.gz
```

输入是同一真实 UKB 配对 DWI/T1，平均 b0 严格用原脚本的 MRtrix 命令、脑图严格用原脚本的 BET 参数生成。归档中现成的 bedpostX `nodif_brain` 与本脚本的 BET 结果不同，本次没有用它代替配准输入。原始 MRtrix 固定在 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`，官方参考为 FSL 6.0.7.4。真实原始图像、矩阵及主体标识留在私有工作区；仓库仅提供[聚合数值](original_ukb_flirt.public.json)。

## FNIT 的输入与输出

```python
from fnit.flirt import run_flirt

result = run_flirt(
    input="dwi_meanbzero_brain.nii.gz",  # 必填；单帧 3D 去脑 b0，移动图像
    reference="T1_brain.nii.gz",          # 必填；单帧 3D 去脑 T1，定义输出网格
    output="b0_in_T1_FNIT.nii.gz",        # 输出；T1 网格上的 float32 重采样 b0
    omat="DWI_to_T1_FNIT.mat",            # 输出；b0→T1 的 FSL scaled-mm 4×4 矩阵
    init=None,                             # 输入；无初始矩阵时用默认初值
    dof=6,                                 # 刚性变换的 6 自由度
    cost="normmi",                         # 4/2/1 mm 精化代价；搜索仍用默认 corratio
    device="cuda:0",                      # PyTorch 计算设备；GPU 采用默认 TF32
    overwrite=False,                      # 已存在的输出文件不覆盖
)
```

`result` 是 `FLIRTResult`：`moved` 为带 T1 几何信息的 `nibabel.Nifti1Image`；`matrix`/`fsl_matrix` 是输入到参考的 FSL scaled-mm NumPy 4×4 矩阵；`moving_to_fixed_world` 是对应的 RAS 世界坐标正向 4×4 矩阵；`fixed_to_moving_world` 是逆向 4×4 矩阵；`qc` 是包含代价、求解器、评估次数和设备的字典。FSL `.mat` 不能直接当成世界坐标矩阵使用。`output` 或 `omat` 至少指定一项。完整命令行参数见[FLIRT 使用说明](../../docs/flirt/README.md)。

独立比较命令如下；每个参数显式给出，`--output` 为**私有**报告路径，因为原始报告含输入哈希和矩阵：

```bash
python tools/benchmark_connectome_registration.py \
  --b0 dwi_meanbzero_brain.nii.gz \
  --t1 T1_brain.nii.gz \
  --fsl-matrix DWI_to_T1_FSL.mat \
  --fsl-moved b0_in_T1_FSL.nii.gz \
  --device cuda:0 \
  --output private_registration_report.json
```

其中 `--b0`/`--t1` 是同输入图像，`--fsl-matrix` 是独立官方矩阵，`--fsl-moved` 是固定官方矩阵生成的 T1 网格参考图，`--device` 是 FNIT 设备，`--output` 是私有完整比较记录。固定矩阵重采样隔离测试用 `tools/benchmark_connectome_registration_resample.py`，它另外要求 `--scratch` 私有临时目录；如已提供 `--fsl-moved`，便复用该参考图，不再调用 FSL。

## 同一真实病例的修订后对照

参考矩阵取原始 UKB 命令的 FSL FLIRT 6.0.7.4 输出。双方使用同一去脑 b0（104×104×72）和去脑 T1（162×215×180）。把两份 input→reference 矩阵转成 world-RAS 后，在 b0 视野内取 13×13×13 个点，测量同一点的配准位置差；影像指标取双方非零体素交集。完整聚合值与当前源码 SHA-256 见[数值报告](original_ukb_flirt.public.json)。

| 指标 | 修订前 FNIT GPU | 修订后 FNIT CPU | 修订后 FNIT H100 TF32 |
|---|---:|---:|---:|
| 世界坐标位移 RMS，相对 FSL | 0.195330 mm | 0.009932 mm | 0.009983 mm |
| 世界坐标位移 95% 位 | 0.285483 mm | 0.015217 mm | 0.015355 mm |
| 重采样图前景 Pearson | 0.999512 | 0.9999993 | 0.9999724 |
| 重采样图前景 Dice | 0.997655 | 0.999892 | 0.999557 |
| 重采样图前景 MAE，原始强度 | 75.967 | 2.903 | 17.411 |

修复来自 FSL 2111.2 的 `optimise_strategy3()`：它先允许 12 自由度粗角搜索，然后按 `min(用户请求的自由度, 7)` 自由优化每个候选。旧版 FNIT 在此处固定用 7 自由度，因此 `-dof 6` 候选仍带额外缩放。两例 8 mm 入选候选与 FSL 最近矩阵的 Frobenius 距离从 0.790、3.883 降至 0.00068、0.00139；后续最终位移随之缩小。粗网格平移/缩放及细网格最小点原本已接近 FSL，误差主要从候选自由优化开始。12-DOF 路径仍用 7 自由度候选优化，符合原 schedule。

| 本例时间 | 范围 | 实测 |
|---|---|---:|
| FSL CPU | 完整 `flirt` 命令，含读写 | 10.02 秒 |
| FNIT CPU | 已载入影像后的求解和重采样调用，不含写盘 | 45.26 秒 |
| FNIT H100 | 与 CPU 相同调用范围；当时 GPU 被其他任务持续占满 | 337.75 秒 |

GPU 本次时间不能用于计算加速比。FSL 与 FNIT 仍未逐矩阵或逐体素一致；本表只验证一例真实跨模态配对。FNIT 的 output 网格、矩阵方向与 FSL 相同。原始 UKB 图像及被试标识不进入公开仓库；公开 T1w 对照图见[FLIRT 功能页](../../docs/flirt/README.md)。

## 参考文献与原实现

- Jenkinson et al., *Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images*, NeuroImage 2002，[doi:10.1006/nimg.2002.1132](https://doi.org/10.1006/nimg.2002.1132)。
- [FSL FLIRT 原实现代码库](https://git.fmrib.ox.ac.uk/fsl/flirt)；[UKB-connectomics 原始配准脚本](https://github.com/sina-mansour/UKB-connectomics/blob/ec73ab75c868060e3ecacd19735b7a347ad226a5/scripts/bash/probabilistic_tractography_native_space.sh)。
