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

`result` 是 `FLIRTResult`：`moved` 为带 T1 几何信息的 `surfa.Volume`；`matrix`/`fsl_matrix` 是输入到参考的 FSL scaled-mm NumPy 4×4 矩阵；`moving_to_fixed_world` 是对应的 RAS 世界坐标正向 4×4 矩阵；`fixed_to_moving_world` 是逆向 4×4 矩阵；`qc` 是包含代价、求解器、评估次数、设备和验证范围的字典。FSL `.mat` 不能直接当成世界坐标矩阵使用。`output` 或 `omat` 至少指定一项。完整命令行参数见[FLIRT 使用说明](../../docs/flirt/README.md)。

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

## 实数据对照

矩阵比较先把双方 FSL 矩阵分别换成 b0→T1 世界坐标矩阵，再在 b0 视野内均匀取 `13×13×13` 个点，统计变换后两点的欧氏距离。图像比较用每种方法各自估计的矩阵，将**同一** b0 重采样到同一 T1 网格；相关与前景 MAE 限定在双方非零交集，Dice 对非零前景计数。MAE 单位是未归一化 b0 强度。

| 比较项 | 实测结果 |
|---|---:|
| 世界坐标位移：平均 / p95 / 最大 / RMS | 0.186655 / 0.285483 / 0.367350 / 0.195330 mm |
| 重采样图：T1 网格 | 162×215×180 |
| 各自估计矩阵：前景 Dice / 非零 Pearson | 0.997655 / 0.999512 |
| 各自估计矩阵：交集 MAE / 全图 MAE | 75.967 / 19.364 原始强度单位 |
| 官方 FSL CPU `flirt` 整条命令墙钟 | 10.02 s |
| FNIT H100 GPU 求解及重采样调用 | 19.155 s |
| FNIT PyTorch CUDA 峰值已分配 | 0.746 GiB |

为了定位误差，固定**同一个 FSL 矩阵**，分别由 FSL `-applyxfm` 和 FNIT 纯 PyTorch 重采样：前景 Dice `0.999995`、非零 Pearson `0.999999995`、交集 MAE `0.1490`、全图 MAE `0.03804`。FNIT 重采样调用 `0.512 s`、CUDA 已分配峰值 `0.693 GiB`；FSL 独立 `-applyxfm` 命令 `0.73 s`。因此本例主要剩余差异来自配准求解过程；当前矩阵和图像均**未达到逐值一致**。FSL 计时包括命令启动与读写，FNIT 计时从已载入图像的模型调用开始；CPU 与 GPU 的计时仅描述本次运行，不能解释为等硬件加速比。

UKB 图像不在公开仓库展示。公开数据的[示例脑图](../../docs/connectome/figures/ds004666_default_dwi_mask_comparison.png)可用于查看 b0 掩膜外观，但它不是本次配准对照的图像；数值表只对应上面的真实配对 UKB 输入。

## 逐阶段定位记录（2026-09-28）

对同一输入及同一官方最终矩阵，分别调用官方 FLIRT 的 `measurecost` 和 FNIT 的代价函数；8 mm 用 `corratio`，4/2/1 mm 用 `normmi`。官方 schedule 的 `setrow` 要把 4×4 矩阵**按列**展开；测量结果重新组成矩阵后与输入的最大元素差小于 `6×10⁻⁶`，才计入下表。官方固定矩阵代价在 CPU 节点测量，FNIT 代价在同节点测量。

| 采样尺度 | 官方代价 | FNIT 代价 | 绝对差 |
|---|---:|---:|---:|
| 8 mm | 0.137430 | 0.1374304 | 0.0000004 |
| 4 mm | −1.219517 | −1.2194949 | 0.0000221 |
| 2 mm | −1.188031 | −1.1883177 | 0.0002867 |
| 1 mm | −1.188751 | −1.1890363 | 0.0002853 |

官方默认 schedule 另在生成原始参考矩阵的同一台主机上逐阶段保存候选。8 mm 搜索后，官方与 FNIT 各保留 2 个候选；两组候选按最近矩阵配对，4×4 矩阵的 Frobenius 距离为 `0.79` 和 `3.88`。这表明可观察到的首次分叉已在 8 mm 搜索阶段，而该尺度的固定矩阵代价基本一致。候选差异的具体来源尚未定位，不能把它归因于某一个搜索函数。这里的 Frobenius 距离只用于定位候选，**不是**毫米位移。

官方 `flirt` 对这例数据还显示 CPU 平台差异：相同输入、二进制、参数和默认 schedule，在原参考主机上重新求解所得矩阵与原记录逐元素一致；换到另一 CPU 节点，最终 4×4 矩阵的最大元素差为 `0.208398`。后续追踪求解轨迹时须固定官方运行主机；不能把跨主机的矩阵变化算作 FNIT 误差。这个矩阵元素差也不能直接解释为世界坐标毫米位移。

下一步应核对 8 mm 粗角搜索的质心初值、每个角位置的平移与共同缩放优化，再核对 11×11×11 细角代价格、阈值和局部极小值；找到首个不同的中间量后只修改相应函数，并重跑 8/4/2/1 mm 及公开数据叠图。目前没有完成该修复或公开配准叠图，最终 b0→T1 配准仍以本页上一节的误差为准。
