# MNI152 非线性变换和检查图

2026-10-11：独立生产函数默认且只支持 GPU 后处理；三个原版路径参数默认 None，仅保留旧签名兼容。原版命令由独立 benchmark 执行。模型、转换、求逆与检查图算法保持成熟实现，最新验证见[本轮记录](../../validation/recon_all/optimizations/20261011_metrics_install/README.md)。

[返回 recon-all](README.md) · [真实 T1 配对结果](../../validation/recon_all/python_gpu_port/mni_nonlinear_real_20260929.json)

## 1. 功能与流程

标准单 T1 流程先以 PyTorch SynthMorph affine 生成个体 `aff.lta`，再用 PyTorch 的 deform 模型优化同一裁剪 T1 到 MNI152 的非线性变换。这里只需模型产生的变换，已跳过没有后续消费者的 `moved` 和 `fixed_moved` 两张重采样图。当前 CUDA recon-all 和独立函数都调用 FNIT 已有完整 GPU 转换、求逆与检查图算子。`postprocess_backend="conda"` 或 CPU 设备在输入读取、目录创建前报错。历史原软件比较仍绑定其对应版本；最新参数见下文及 [GPU 后处理说明](MNI_WARP_GPU.md)。

```mermaid
flowchart LR
 A[自产 conform orig、crop 和 affine LTA] --> B[SynthMorph deform 两次反对称 FP32 前向]
 B --> C[自产 deform.mgz 与两条几何 LTA]
 C --> D[完整 MNI 前向 warp 转换]
 D --> E[完整原图网格 GCAM 求逆]
 D --> F[原图最近邻检查图]
 E --> G[逆向 FS NIfTI warp]
 F --> H[test.nii.gz]
```

## 2. Python 调用与输入输出

`run_mni_nonlinear_chain(...)` 的参数如下。`subject_dir` 指向已含 `mri/orig.mgz`、`mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz` 与 `aff.lta` 的被试目录；前者是 1 mm conform 网格，裁剪 NIfTI 保留其 scanner RAS。`weights_dir` 必须有经大小和 SHA-256 校验的 `synthmorph.deform.3.h5`；`assets_dir` 必须有同一固定 MNI152 版本的裁剪及完整 1 mm 模板。`warp_convert`、`ca_register`、`mri_convert` 默认 None，保留旧调用兼容但不读取或执行。`device` 默认 `cuda:0`，要求实际 CUDA 设备。模型沿用 FP32 例外，返回时恢复调用方 TF32 设置；`threads` 为正整数 CPU 线程数，默认 4。`postprocess_backend` 默认且只支持 `gpu`。`chunk_slices` 默认 16，为转换与检查图的 X 方向正整数分块大小，不改变求逆网格、迭代或停止规则。

```python
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain

nonlinear_outputs = run_mni_nonlinear_chain(
    subject_dir="/data/subjects/sub01",  # 已生成 orig、裁剪 T1 和 affine LTA 的被试目录
    weights_dir="/data/fnit-weights",  # 已校验的 SynthMorph deform 权重目录
    assets_dir="/data/fnit-assets",  # 已校验的裁剪及完整 MNI152 模板目录
    device="cuda:0",  # GPU 模型使用既有 FP32 例外
    threads=4,  # CPU 算子线程数
    postprocess_backend="gpu",  # 默认gpu；复用完整已有GPU算子
    chunk_slices=16,  # GPU 转换和检查图的 X 轴分块大小
)
print(nonlinear_outputs["forward"])  # 目标 MNI152 网格上的前向变换绝对路径
```

返回字典还含 `inverse`、`check`、`model`、`device`、`precision`（CUDA FP32 例外及未启用半精度）和四个子步骤的 `timings_seconds`，另有 `postprocess_backend` 及 GPU `postprocess` 分步报告。三个正式输出分别位于被试的 `mri/transforms/synthmorph.1.0mm.1.0mm/`：前向 `warp.to.mni152.1.0mm.1.0mm.nii.gz` 的结构为 `(193,229,193,1,3)`，逆向同名加 `.inv` 的结构为 `(256,256,256,1,3)`；两者为 NIfTI 位移向量，intent 1006，单位为 mm。`test.nii.gz` 是完整 MNI152 的 `(193,229,193)` uint8 最近邻检查图。`tmp/deform.mgz` 及两个 LTA 是中间文件。输入缺失、程序失败或网格不符时抛出异常，主流程在运行 JSON 中记录失败阶段。

## 3. 命令行

本组合函数以 Python API 接入 `fnit-recon-all`；完整流程的全部原始 T1 参数见[Python 配方](PYTHON_GENETIC_PROFILE.md)。只验证本阶段时运行独立脚本：

```bash
python tools/benchmark_recon_python_metrics_install.py --config /data/validation/config.json
```

`--config` 必填，为私有 JSON：`code_commit` 指 installed 源码提交；`benchmark_commit` 指验证脚本提交；`cases` 为列表，每项 `alias` 为公开别名、`subject` 为已完成 FNIT 目录、`hemi` 为 lh/rh；`assets`/`weights` 为经校验的资源目录；`reference_bin` 为独立参考程序目录；`baseline_cache` 为固定旧缓存模块；`output` 为不存在的新输出目录；`gpu_uuid`/`gpu_index` 分别为物理设备 UUID/编号。逻辑设备固定 cuda:0，线程固定4，通过 CUDA_VISIBLE_DEVICES 与 CPU affinity 选择实际资源。报告含数值误差、各阶段 wall 和源码/输入 SHA，不输出私有路径；大图和原日志留在本地。

## 4. 原软件调用

对应官方 8.2 链的主要命令依次是 `mri_synthmorph -m deform -i aff.lta -t deform.mgz invol.crop.nii.gz mni152.1.0mm.cropped.nii.gz`、`mri_warp_convert --inras deform.mgz --insrcgeom invol.crop.nii.gz --outfswarp warp.to.mni152.1.0mm.1.0mm.nii.gz --vg-thresh 1e-4 --lta1-inv reg.crop-to-invol.lta --lta2 reg.1.0mm.cropped.to.1.0mm.lta`、`mri_ca_register -invert-and-save 前向变换 逆向变换`、`mri_convert -rt nearest orig.mgz -at 前向变换 test.nii.gz`。FNIT 用 nibabel 几何生成两个等价的 LTA，避免让这一阶段依赖系统安装的模板目录。[固定版本的 mri_synthmorph](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_synthmorph/mri_synthmorph)、[mri_warp_convert](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_warp_convert/mri_warp_convert.cpp)可查阅。

## 5. 精度、运行时间与真实脑图

最新两例结果与检查图示例入口见[任务5](../../validation/recon_all/optimizations/20261011_metrics_install/README.md)。以下保留各旧版本的实际记录。

去标识真实 `sub-01` 的阶段原型在 CPU 上完成 PyTorch deform 推理用时 91.50 s；相同官方存档与 FNIT 输出的前向位移相关性超过 0.9999999999999、平均绝对差 0.0000084 mm。逆向位移相关性超过 0.9999999999、最大局部差 0.233 mm；检查图非零区域相关性 0.9999973，有 127 个不同体素。详细值见[机器记录](../../validation/recon_all/python_gpu_port/mni_nonlinear_real_20260929.json)。该阶段结果是同一真实 T1 的阶段比较，不能替代全流程 138 项验收。

2026-09-30 的两例新连续链输入对照显示：跳过未使用的重采样图后，前向 warp、逆向 warp 和检查图的 **SHA-256 均与原 CPU 阶段完全相同**。`sub-02` 同在 control-host 上用时 282.33→253.95 秒；`sub-01` 同在 accelerator-a-1 上用时 339.99→343.26 秒，共享负载使单次时间不能证明稳定提速。配对分解显示耗时还集中在 deform 模型和原生 warp 求逆。[逐例机器报告](../../validation/recon_all/python_gpu_port/performance_20260930/README.md)保留模型、转换、求逆及检查图秒数。

TF32 CUDA 的正向/逆向 warp 相对 CPU 的 P99 差为 0.075/1.108 mm，检查图相差 220,314 个体素，父子进程采样峰值 20,308,819,968 字节。本阶段采用 FP32 CUDA：两例正向/逆向 P99 差均为约 0.000015/0.000031 mm，最大逆向差分别为 0.000473/0.000183 mm；最近邻检查图分别相差 19/17 个体素。这些差异全部保留在严格复现诊断中，不能以文件不一致单独否定 GPU 路径。

CUDA FP32 的第一例独立链用时 161.73 秒，CPU 为 339.99 秒，进程采样峰值约 13.00 GB；这份试验包含原先的两张未使用重采样图。当前封装还会跳过它们，并在原生 warp 转换前释放模型引用；其阶段与整例实测另见[当前性能报告](../../validation/recon_all/python_gpu_port/performance_20260930/README.md)。

三个 MNI 非线性输出属于固定输出范围，后续 white/pial 依赖的是另一条 affine/辅助分割链；其实际调用关系见 `native_free.py`、`mni_aux_chain.py` 和 `aux_seg.py`。完整运行仍单列分区 Dice、双向表面距离和脑区指标，整体指标等效在尚无正式门槛时保持未判定。当前尚未定位全部与官方的历史局部位移差，不能将这些差异一概归因于随机性。

## 6. 近期版本和 benchmark

串行候选省去未消费的网络逆向积分/合成，反对称velocity两次前向及原生数值求逆保留。两例固定自产输入的forward、inverse、check与旧GPU路径SHA-256完全相同；阶段136.720→123.601秒、121.549→129.343秒，尚未证明稳定整阶段加速。实际前向与复现见[串行说明](SERIAL_OPTIMIZATION.md)。

2026-10-02 新增GPU后处理并接入CUDA recon-all；成熟散射最后半体素的native rint兼容修复、向量编码和scanner RAS校验详见 [专项说明](MNI_WARP_GPU.md)。本页上述历史性能与精度记录仍绑定各自原版本；当前同输入阶段见 [task_05 报告](../../validation/recon_all/optimizations/20261002_parallel/task_05/README.md)，当前整例见 [五任务结果](../../validation/recon_all/optimizations/20261002_parallel/FINAL_RESULTS.md)。

最新 GPU 后处理的两例、两轮完整自产 MNI 阶段实测为 36.173–38.675 秒，新运行 Conda 对照为 122.813–130.526 秒，四个配对加速 3.247×–3.608×，直接输出比较数值零差异；过程采样峰值 13.103 GB。完整分步、缓存范围及整例验收边界见上述 task_05 报告。

运行源码 `8d750e2` 的原始T1整例中，完整MNI阶段sub01为126.497→46.024秒，sub02为129.062→58.258秒。该时间包含模型加载、传输和读写，不能与冻结输入阶段的秒数互换。sub01与基线的三个正式输出体素、dtype和空间矩阵相同，NIfTI生产者描述、四元数负零和scalar图未使用的 `pixdim[4]` 仍在严格文件诊断中报告；没有伪造FreeSurfer描述以提高文件通过数。两例最终比较详情见五任务结果。


## 7. 文献与原实现

Hoffmann M, et al. SynthMorph: learning contrast-invariant registration without acquired images. *IEEE Trans Med Imaging*. 2022;41:543–558. [doi](https://doi.org/10.1109/TMI.2021.3116879)。原实现固定链接见第4节；GPU算子和子步骤说明见 [MNI_WARP_GPU](MNI_WARP_GPU.md)。


2026-10-11 两例真实验证已完成，完整阶段、分步骤、参考ELF迁移哈希和安装范围见[本轮记录](../../validation/recon_all/optimizations/20261011_metrics_install/README.md)。新旧GPU MNI数值零差异，12套完整统计文件逐字节一致；仅本次阶段范围，整例由主集成任务单列。
