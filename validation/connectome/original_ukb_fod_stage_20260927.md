# 原 UKB 流程：全脑多组织 FOD 同输入对照

原脚本先从校正 DWI 估计 Dhollander 三组织响应，再从平均 b0 运行 `bet -m -R -f 0.2 -g -0.05`，把脑掩膜膨胀两次后求解多组织 CSD。本次两臂共用**同一**真实 UKB 校正 DWI、MRtrix 梯度表、已核对的 WM/GM/CSF 响应及该膨胀掩膜。原版 MRtrix 固定在 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`。FNIT 运行时只用 PyTorch，不启动 MRtrix；官方程序仅在独立参考臂运行。受试者标识、原图、逐文件哈希与完整报告留在私有服务器。公开[聚合 JSON](original_ukb_fod.public.json)不含这些信息。

## 输入、输出与用法

`fit_mrtrix_msmt_csd` 的输入是：`signal`，float32 四维 DWI `[X,Y,Z,N]`；`grad_mrtrix`，与 DWI 帧逐行对应的 `[N,4]` 梯度方向和 b 值；`shell_bvals`，MRtrix 响应文件头中的 `[S]` 壳中心；`wm_response`，`[S,≥5]` 的白质球谐响应；`gm_response`、`csf_response`，各 `[S]` 的各向同性响应；`mask`，与 DWI 网格一致的三维布尔掩膜；`batch_size`，每批拟合的掩膜体素数。输入张量可放在 CPU 或 CUDA，GPU 使用默认 TF32；47 参数约束求解使用 float64。返回设备上的 `wm_fod` float32 `[X,Y,Z,45]`、`gm_fod` float32 `[X,Y,Z]`、`csf_fod` float32 `[X,Y,Z]`，背景为零；球谐系数采用 MRtrix 顺序。本例为 `104×104×72×105` DWI、212,831 个掩膜体素，三个输出形状分别为 `104×104×72×45`、`104×104×72` 和 `104×104×72`。

```python
from fnit.connectome.fod import fit_mrtrix_msmt_csd

# 以下变量先按同一 DWI 的 NIfTI、梯度表、三个响应文件和膨胀掩膜读入。
# signal: torch.float32 [X,Y,Z,N]；mask: torch.bool [X,Y,Z]，两者在同一设备。
# grad_mrtrix: [N,4]；shell_bvals: [S]；wm_response: [S,≥5]。
# gm_response 和 csf_response: 各 [S]；响应文件必须对应同一批 DWI。
wm_fod, gm_fod, csf_fod = fit_mrtrix_msmt_csd(
    signal=signal,                 # 输入；校正后的四维 DWI
    grad_mrtrix=grad_mrtrix,     # 输入；每帧方向与 b 值
    shell_bvals=shell_bvals,     # 输入；响应文件声明的壳中心
    wm_response=wm_response,     # 输入；白质响应系数
    gm_response=gm_response,     # 输入；灰质响应强度
    csf_response=csf_response,   # 输入；脑脊液响应强度
    mask=mask,                   # 输入；膨胀两次的 b0 脑掩膜
    batch_size=4096,             # 每批最多拟合 4096 个掩膜体素
)
```

独立原版参考命令如下；`dwi.mif` 和三个响应来自同一受试者，[响应阶段](original_ukb_dhollander_stage_20260927.md)另有同输入验证：

```bash
bet dwi_meanbzero.nii.gz dwi_meanbzero_brain.nii.gz -m -R -f 0.2 -g -0.05
mrconvert dwi_meanbzero_brain_mask.nii.gz dwi_meanbzero_brain_mask.mif
maskfilter dwi_meanbzero_brain_mask.mif dilate fod_mask.mif -npass 2
dwi2fod msmt_csd dwi.mif wm.txt wm_raw.mif gm.txt gm_raw.mif \
  csf.txt csf_raw.mif -mask fod_mask.mif -nthreads 8
```

复跑比较入口是 [`tools/benchmark_connectome_response_fod.py`](../../tools/benchmark_connectome_response_fod.py)。参数 `--dwi` 指四维 NIfTI，`--grad` 指 MRtrix `[N,4]` 文本，`--wm-response`／`--gm-response`／`--csf-response` 指三个响应，`--mask` 指上述膨胀掩膜，`--ref-wm`／`--ref-gm`／`--ref-csf` 指独立原版输出，`--device` 是 GPU，`--batch-size` 是批体素数，`--max-voxels 0` 表示全掩膜，`--out` 是完整报告输出。**完整报告含私有输入哈希，必须写入私有目录。**

## 全脑实测

| 输出 | 比较值数 | 平均绝对误差 | 最大绝对误差 | Pearson |
|---|---:|---:|---:|---:|
| 白质 45 个球谐系数 | 9,577,395 | 4.39×10⁻¹³ | 2.98×10⁻⁸ | >0.999999999999999 |
| 灰质 | 212,831 | 4.14×10⁻¹⁷ | 2.84×10⁻¹⁴ | >0.999999999999999 |
| 脑脊液 | 212,831 | 4.45×10⁻¹⁵ | 9.31×10⁻¹⁰ | >0.999999999999998 |

掩膜内所有三个组织值的绝对误差均小于 `1e-7`；这是数值容差内的一致，**不代表 NIfTI 字节完全相同**。先抽样的 10,000 个真实掩膜体素也得到白质／灰质／脑脊液最大误差 `7.45e-9`／`5.53e-15`／`6.39e-16`，之后才完成全脑比较。

MRtrix 参考 CPU 8 线程完整 `dwi2fod` 命令用时 **107.04 s**、峰值 RSS **256,992 KiB**。FNIT H100 GPU 的已载入张量全脑求解用时 **269.457 s**，PyTorch CUDA 峰值已分配 **1.354 GiB**。当前实现未测完整进程墙钟和 RSS，不能将旧实现的进程数字移用至此。两种时间覆盖范围和硬件不同；当前 PyTorch 求解仍明显较慢。已把仅依赖设计矩阵的 Gram、Cholesky 与约束投影预计算并跨体素批次复用；当前全脑数字在该实现下复测。后续更新求解器时，本页及 JSON 必须随当前实现重新测量。

下图来自**公开** [OpenNeuro ds004666](https://openneuro.org/datasets/ds004666)，仅供查看白质 FOD 的外观；上表的数值和时间来自私有配对 UKB 输入，不把私有脑图纳入仓库。

![公开数据白质 FOD 示例](ds004666/response_fod_example.png)

此阶段只固定了 BET 和膨胀掩膜作同输入求解。BET 的纯 PyTorch 版本仍在逐步核对；原流程后续的 `mtnormalise`、ACT 追踪、SIFT2 和七套 atlas 的 28 张矩阵尚须分别通过原版输入输出验收。
