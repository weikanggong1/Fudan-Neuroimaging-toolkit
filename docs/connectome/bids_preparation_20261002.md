# 原始 BIDS 连接组输入准备

## 1. 功能简介
`prepare_bids_connectome` 选择同一受试者、session 的原始 DWI 与 T1w，运行 FNIT PyTorch TOPUP/EDDY 和官方 FreeSurfer recon-all，返回连接组计算所需路径。反向相位编码缺失时只运行 EDDY。已有外部校正影像或解剖可显式提供。

## 2. Python 调用、输入与输出
```python
from fnit.connectome.bids import prepare_bids_connectome
prepared_inputs = prepare_bids_connectome(
    bids_root="/data/raw",         # 原始 BIDS 根目录
    output_dir="/data/new_output", # 本例输出目录
    subject="CON01",              # 受试者标签，可带 sub-
    session="preop",              # session 标签，可带 ses-
    acquisition="AP",             # 前向 DWI acquisition
    device="cuda:0",              # TOPUP、EDDY 和脑掩膜的目标设备
    eddy_gp_seed=12345,            # EDDY GP 选点种子；None 保留按时间初始化
)
```
原始 DWI 为四维 NIfTI，bval 每帧一个值，bvec 为 3×N；JSON 必须提供相位编码与 readout。T1w 为三维 NIfTI。`subject` 必填；`session/run/acquisition/direction` 默认 None，用于消除选择歧义。`t1` 可显式指定 T1；`freesurfer_subject_dir` 是完整外部 FreeSurfer subject，与 `t1` 互斥。`corrected_dwi` 和 `rotated_bvecs` 必须同时提供。`device` 默认 cuda:0，`overwrite` 默认 False，`eddy_gp_seed` 默认 None，显式种子取 1…2³²−1。
返回 `BIDSConnectomeInputs`：`dwi` 是校正 NIfTI 路径，`bvals` 是原始选定 bval，`bvecs` 是旋转后梯度路径，`freesurfer_subject_dir` 是解剖目录，`stages` 记录 completed/skipped/supplied/no_reverse_pe。
自动 recon 缓存按 T1 路径、大小、修改时间与内容 SHA-256 匹配。已存在 orig 而指纹不匹配时明确报错，使用新输出目录或显式传入外部解剖；不会删除旧产物。

## 3. 命令行调用
此准备函数没有独立 CLI；pipeline 命令入口由总任务统一接入固定 GP seed。Python 接口可直接调用。

## 4. 对应原软件调用
参考工具的 TOPUP/EDDY 输入由真实 JSON 和梯度生成，不能猜 readout。官方解剖命令为 `recon-all -sd SUBJECTS_DIR -s SUBJECT -i T1w.nii.gz -all`。本轮独立官方组件队列额外使用 `-parallel -openmp 8`，其耗时不能视为整条 pipeline 耗时。TOPUP/EDDY 官方命令仅用于隔离 reference。

## 5. 最新精度与运行时间比较
入口测试验证 seed 透传、seed 改变时缓存失效、设备透传、T1 变化防护与 TOPUP 已选 AP b0 复用。十例新下载 ds001226 raw 已核验：CON01、CON03–CON11，同 session DWI/T1w 配对。CON02 的 PA JSON 为 i-，与 AP j- 的 world 方向近乎正交，已排除并以新下载 CON11 补足。33 个 NIfTI（包括排除例）均与冻结 Git annex 的大小、MD5 内容匹配，并保存 SHA-256、S3 ETag/LastModified 及原始 JSON 来源。CON03 真实 TOPUP/EDDY 组件已完成，详见[几何修复和实测](topup_pair_geometry_20261002.md)。官方 recon 与正式完整 pipeline 仍运行，尚无完整端到端耗时。

## 6. 更新记录与 benchmark 记录
2026-10-02：修复成熟入口 AP-only 漏传 device；新增兼容可选 `eddy_gp_seed`，默认随机行为保持原值；同次 TOPUP/EDDY 复用最佳 b0 编号，省去重复选择；自动 recon 跳过增加 T1 指纹。无精度、迭代、分辨率和算法调整。基线 f436de588647a0de80735e4a98d53df5d88e502d；真实数据清单和运行记录位于本轮 task_01 服务器目录。

## 7. 参考文献与原代码
- Andersson et al. 2003, susceptibility-induced distortions correction；[FSL TOPUP](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/index.html)。
- Andersson & Sotiropoulos 2016, integrated correction for off-resonance effects and subject movement；[FSL EDDY](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/index.html)。
- Fischl 2012, FreeSurfer；[官方代码](https://github.com/freesurfer/freesurfer)。
- [ds001226 原始数据与 CC0 声明](https://github.com/OpenNeuroDatasets/ds001226)，DOI 10.18112/openneuro.ds001226.v5.0.0；具体下载绑定 Git commit 与 SHA-256。
