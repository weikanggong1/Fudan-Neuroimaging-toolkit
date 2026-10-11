# Python 配方的 batch 预检修复

## 1. 功能简介

`run_recon_all_python_batch` 为每个被试启动独立 Python CLI，保持输出隔离、设备分配和总线程预算。本轮修复成熟 pipeline 入口的检查顺序：`backend="python"` 先调用单例使用的 `python_profile_options`，再校验解析后的选项和构造 CLI。N4 的显式 SimpleITK/Torch 选择保留，其余固定配方与单例一致。

旧 batch 对被覆盖前的 `wm_backend="native"`、`hemisphere_workers=1` 做检查，错误拒绝 Python 配方中合法的隔离 WM 或 Torch inflation。修复没有改变分割、网格、强度、线程数和生产默认；`backend="native"` 仍使用原参数检查。

```mermaid
flowchart LR
  J[原始 T1 路径和空输出目录列表] --> P[复用单例21键Python配方]
  P --> V[设备 / N4组合 / 线程 / 输出预检]
  V --> C[每设备独立Python CLI]
  C --> R[按输入顺序返回完整报告或汇总失败]
```

## 2. Python 调用、输入和输出

```python
from fnit.recon_all.batch import run_recon_all_python_batch

batch_reports = run_recon_all_python_batch(
    jobs=[{  # 列表保持被试顺序；每项仅含这两个具名字段
        "t1": "/data/sub01_T1w.nii.gz",  # 原始三维单幅T1w
        "subject_dir": "/data/subjects/sub01",  # 不存在或为空的目录
    }],
    weights_dir="/data/fnit-weights",  # 声明并校验的网络权重
    assets_dir="/data/fnit-assets",  # 声明并校验的模板、图谱和LUT
    devices=("cuda:0",),  # 各设备不同，Python配方须显式cuda:N
    threads=4,  # 每个被试总CPU预算，双侧各两线程
    backend="python",  # 先解析完整自有配方，不消费原版阶段程序
    n4_backend="simpleitk",  # 默认通用Python库完整N4，可选torch
    n4_execution="in-process",  # SimpleITK须in-process；Torch可isolated
    input_execution="serial",  # parallel仅SimpleITK/in-process和四线程
    wm_execution="isolated",  # 本键被固定配方解析为optimized隔离WM
    inflate_backend="torch",  # 固定配方同时解析双侧worker=2
    profile_stages=False,  # 不额外插入剖析同步
    cuda_allocator_cache="auto",  # 保留父缓存策略及既有低显存措施
)
```

`jobs`、资源目录、设备列表和线程含义如上；返回 `list[dict]`，与 `jobs` 同序，每项为单例完整报告。MRI 输出仍是 conform XYZ/mm 网格，表面仍是 surface RAS/mm；本修复没有新增影像。输入或组合错误在 dispatch 前抛异常；子进程失败汇总为 `RuntimeError`，保留该被试日志/报告，不回退原版程序。

完整参数见函数中文 docstring 和[主接口](README.md)。21键来自[唯一配方函数](../../src/fnit/recon_all/python_profile.py)：GCA、WM、编辑、defects、inflation、placement、topology、Gibbs、remesh、filled、归一化、sphere、MNI 和双侧调度按配方覆盖；显式 `n4_backend`/`n4_execution` 保留，`input_execution`、`profile_stages`、父 allocator 策略和资源输入保持。`sphere_normals_backend` 不属于这21键，继续原验证。空/重复设备、线程不足、Torch N4 与输入并行、SimpleITK 与隔离 N4 均拒绝。无新增依赖。

## 3. 命令行

batch 目前是 Python API，每个子任务复用同一 CLI：

```bash
fnit-recon-all /data/sub01_T1w.nii.gz /data/subjects/sub01 \
  --weights-dir /data/fnit-weights --assets-dir /data/fnit-assets \
  --device cuda:0 --threads 4 --backend python \
  --n4-backend simpleitk --n4-execution in-process --input-execution serial
```

命令的两个位置参数为原始 T1 和空输出目录；权重与资产必须提供，设备、线程、配方和 N4 设置与 Python 示例相同。非法配置或阶段失败退出非零。合同测试使用真实 CLI 参数解析和公开单例 wrapper，但替换耗时 MRI 执行体；不把 fixture 的 `complete` 当作真实完成。

## 4. 原软件调用

组合参考仍是 `recon-all -i T1w.nii.gz -s subject -all`，仅独立 benchmark 使用。单例/批量生产的 Python 配方不运行此命令；默认混合对照仍允许项目 Conda 内固定源码阶段程序，详细清单在[本轮调用审查](../../validation/recon_all/optimizations/20261010_batch_python_profile_prevalidation/reports/current_inventory.json)。

## 5. 当前代码验证、精度和耗时

本轮只修调度入口。两组旧配置在修复前实际被预检拒绝，在修复后由 CLI fixture 正确解析完整21键；显式 Torch/isolated N4 保留，非法 N4 输入并行和 native 原检查保持。完整结果、测试时间、Python/库版本、当前调用链和源文件 SHA 见[机器报告](../../validation/recon_all/optimizations/20261010_batch_python_profile_prevalidation/README.md)。

没有重新运行真实 T1、batch MRI、原软件或整例；影像误差、分区 Dice、表面距离、脑区偏差、显存和整例提速均未测。该接口修复不产生新的脑图；既有真实验证和脑图保持其原版本，见[Python候选](PYTHON_SURFACE_PIPELINE.md)、[SimpleITK N4](N4_SIMPLEITK.md)、[WM编辑](WM_ASEG_TORCH.md)。原程序脱离与数值/网格验收是分别检查的项目，整体指标等效仍未判定。

## 6. 更新和 benchmark 记录

- 修复前源码绑定 `fc36d6c1c761302ed5b7425b730b17ab4951090b`；修复后 batch 为工作树 overlay，实际 SHA 写入报告。
- 复用成熟 `python_profile_options`，没有第二套配方默认值，也没有删除 native 对照。
- 旧 N4/WM/数字拓扑与整例记录没有改标为本版结果。此处完成的是预检和 CLI 路由合同。

## 7. 参考文献和代码

- [FNIT batch 与源码](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/main/src/fnit/recon_all)
- [固定 FreeSurfer 参考源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)
- Dale AM, Fischl B, Sereno MI. Cortical surface-based analysis I. NeuroImage. 1999;9:179–194。
- Tustison NJ et al. N4ITK. IEEE Transactions on Medical Imaging. 2010;29:1310–1320。
