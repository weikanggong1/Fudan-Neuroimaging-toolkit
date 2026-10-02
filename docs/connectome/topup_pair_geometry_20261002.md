# TOPUP 原始 AP/PA 打包几何

## 1. 功能简介
修复 UKB helper 对非 UKB 原始 BIDS 数据过严的 affine 限制。`prepare_ukb_topup`、`run_ukb_topup` 新增 `pair_geometry`：默认 `strict` 保留同 affine 要求；`fslmerge-first` 保留 AP/PA 体素与顺序，采用 AP header 打包两帧，不重采样。连接组原始 BIDS 入口显式使用后者。矩阵大小和体素间距仍须一致。

## 2. Python、输入与输出
```python
from fnit.topup.ukb import prepare_ukb_topup
prepared_topup = prepare_ukb_topup(
    raw_dir="/data/AP_PA",          # AP/PA 的 nii.gz、bval 和 JSON
    output_dir="/data/topup_input", # 独立输出目录
    device="cuda:0",               # 原有最佳 b0 选择的设备
    overwrite=False,               # 默认拒绝覆盖已有输出
    pair_geometry="fslmerge-first",# 原始 BIDS 打包策略
)
```
原有输入输出不变：`imain` 为两帧 b0 NIfTI，`datain` 为两行相位编码/readout 参数，`ap_index/pa_index` 是选中原始帧号，`ap_scores/pa_scores` 是选择评分。额外 `pair_geometry.json` 保存策略、两幅原始 NIfTI header 二进制 hex、原 affine 以及 `resampled=false`。
`run_ukb_topup` 接受同参数并运行 FNIT TOPUP，返回 `(result, prepared)`。

## 3. 命令行
此 helper 没有独立 CLI；已有 standalone Python 调用保持 strict 默认。连接组 pipeline 接入此策略。

## 4. 原软件调用
隔离 reference：`fslroi AP.nii.gz AP_b0 0 1`、`fslroi PA.nii.gz PA_b0 0 1`，然后 `fslmerge -t merged AP_b0 PA_b0`。参考执行 FSL 6.0.7.4，程序 SHA 及完整命令保存在本轮 JSON。官方输出警告原方向不一致、采用 voxel-based orientation，需要检查；本补丁不将警告解释为几何验证或 TOPUP 精度验证。

## 5. 真实精度与耗时
新下载 ds001226 CON01：AP=96×96×60×102，PA=96×96×60×2。FSL 第一 b0 打包实测：AP 逐值相同、PA 逐值相同，输出 affine 与 AP 逐值相同。保留两幅 qform/sform/pixdim 记录。该证据验证打包行为，未验证 TOPUP/EDDY 官方数值精度，也不是完整 pipeline 耗时。完整 TOPUP/EDDY 组件正在运行；脑图与正式精度耗时需待实测完成。

## 6. 版本记录
2026-10-02：默认 strict 不变，新增显式 fslmerge-first；修复成本单列，不纳入无损优化收益。冻结基线与候选整例须共同使用此 raw 输入兼容补丁，不能把无法执行的旧入口当成较慢基线。相关报告：`validation/connectome/tenraw_20261002/task_01/fslmerge_CON01.json`。
同日 recon 恢复修复：调用前原子记录 T1 路径、大小、mtime 与内容 SHA；记录官方 executable SHA。state 不能单独判为完成，仍检查必需输出和 done。官方部分失败后允许同输入继续，变更 T1 则拒绝复用。

## 7. 参考与原代码
[TOPUP 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/index.html)、[FSL utilities](https://fsl.fmrib.ox.ac.uk/fsl/docs/utilities/fslutils.html)、[ds001226 源库](https://github.com/OpenNeuroDatasets/ds001226)。Andersson et al. 2003，susceptibility-induced distortion correction。
