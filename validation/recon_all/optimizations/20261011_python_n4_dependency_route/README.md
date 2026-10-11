# Python GA：真实启动失败与 N4 依赖接线回归

## 1. 功能和范围

原始公开 T1 的首次 GA 整例启动在 N4 失败：运行解释器缺少主页已经声明的 SimpleITK。此前的 wheel 检查只验证导入与 CLI help，没有验证配方实际需要的 N4 依赖。现将依赖预检接入生产函数、整例 benchmark 和实际安装入口；缺少依赖时在模型、分配器、输出目录和子进程之前失败。不会改换 N4 算法，也不改变线程、精度或 CUDA 状态。

本页分别保留失败启动和修复后合同、安装结果。它们不是完成的整例，不替代真实阶段回归。通用 SimpleITK/ITK 库属于 Python 运行依赖；生产配方不调用原 FreeSurfer N4 程序。

## 2. 输入、输出、空间与版本

- 失败启动：公开 ds000114 `sub06` 原始 T1，SHA-256 `7e33afb28f631fac31d81e2428a6144f61aba4102859c694eeb49ee584de04f7`。从新输出目录启动，无手动补跑、检查点或官方结果输入。配方、包和包装脚本的不同版本见 [raw_t1_launch_v1.json](raw_t1_launch_v1.json)。
- 修复后包：Git 基底 `af1e782dd19a2e9473619f345dbf22c4fd8204a7`，仅三项实际源码 overlay，1462 项完整清单见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。清单 SHA `a403b973b4da16a8a403eeb96e92008935a20404bcd21093004d592a078d6d93`。未纳入他人的工作区 retessellation 修改。
- 预检输入：所选 `n4_backend` 字符串，无影像、坐标或单位。返回 JSON 兼容字典：后端、是否需要依赖、SimpleITK/ITK 版本及扩展库 SHA。完整算法的输入网格、体素类型和失败规则见 [N4 功能说明](../../../../docs/recon_all/N4_SIMPLEITK.md)。
- 安装输入：冻结源码目录、其清单、新输出目录和已装主页依赖的 Python 解释器，全部为显式 `Path`。输出包含 wheel、实际独立安装目标、入口检查及 `report.json`；无影像输出。
- 新 wheel：5,275,390 字节，SHA `832ae8411430cafb25b415c7cab57e9d7bf25bef4140564c899901bb60f181ac`。十个关键模块的源码与解码 wheel 相同，见 [installation.json](installation.json)。

## 3. Python 和命令行调用

```python
from fnit.recon_all.input_n4_chain import require_n4_runtime_dependencies

n4_dependencies = require_n4_runtime_dependencies(
    n4_backend="simpleitk",  # 只核对选中的完整 N4 后端，无影像计算
)
# n4_dependencies 记录版本和实际扩展库 SHA；缺失或损坏依赖抛异常。
```

调用生产整例和 benchmark 时自动预检，用户不需要手动调用。安装复核使用同一生产预检：

```bash
installation_arguments=(
  --source-directory /data/frozen-source         # 与本次清单一致的源码
  --source-manifest /data/SOURCE_MANIFEST.json   # 相对路径、大小和 SHA
  --output-directory /data/new-install-check    # 须不存在、位于源码之外
  --python-executable /conda/envs/fnit/bin/python # 已安装主页依赖的解释器
)
python tools/verify_recon_genetic_profile_install.py "${installation_arguments[@]}"
```

四个安装参数均必填，无隐式目录。不下载或升级依赖，不改变冻结源码。已有输出、越界清单、大小或 SHA 不一致、缺少所选依赖、构建/安装/入口失败均抛异常，私有日志保留在本次输出。生产 API 的成功返回结构和 MRI/surface RAS 约定见 [GA 配方说明](../../../../docs/recon_all/PYTHON_GENETIC_PROFILE.md)。

## 4. 原软件对应关系

依赖预检、wheel 验证和错误提前属于 FNIT 工程步骤，没有等价原软件 CLI。底层 N4 对照是固定版本 ITK N4 完整偏置场校正，完整参数及参考命令见 N4 说明；拓扑对照是隔离运行的 `mris_fix_topology -ga`。生产测试只读取公开 T1、声明的资源和自产中间结果。

## 5. 本次实测

| 范围 | 结果 | 边界 |
| --- | --- | --- |
| 原始 T1 首次 GA 启动 | EXIT 1，失败阶段 N4，缺少 SimpleITK | 保留失败运行，无完整表面或脑区指标 |
| 已完成导入阶段 | 51.414 秒；N4 失败调用 0.264 秒 | 不是完整 N4 时间 |
| 首次 CLI / API / 包装墙钟 | 66.778 / 61.827 / 81.065 秒 | 包装另含 13.481 秒资源哈希校验；不是整例耗时 |
| 失败前程序启动审计 | 7 个事件，0 次原软件阶段启动尝试 | 仅失败前段的 Python 审计；不是完整 OS 隔离证明 |
| 缺依赖提前失败及已有 GA 入口合同 | 9 测试、21 子测试通过；13.64 秒 | 实际私有调度函数和真实 benchmark main；无 MRI/GPU 前向 |
| 修复后离线构建 / 安装 / 入口组 | 2.883 / 0.970 / 6.893 秒 | 本地已有依赖、实际独立安装，不是干净 Conda 整例 |
| 修复后 fresh CLI help | 1.668 / 2.165 秒 | 实际从独立安装目标导入，不是影像性能 |
| CFFF 新目标安装 / 入口组 / 合计 | 2.488 / 14.097 / 16.588 秒；fresh CLI 2.457 / 2.959 秒 | 源码、wheel、安装目标十模块 SHA 相同；实际所选 N4 库预检通过 |
| N4 实际运行依赖 | SimpleITK 2.5.6 / ITK 5.4，扩展库 SHA `c25cd8cf7053c8224c246744e1f1bc9af12db0c95a8221b27e9546cc4511b62c` | 预检不初始化 CUDA、不改变调用者 TF32 标志 |
| CFFF 完整 N4 同输入回归 | API 187.914 秒；decoded 体素差异、最大/P99误差为0；nu0文件字节相同 | 使用公开 sub06 冻结 orig；不是从原始 T1 的整例，也不消除旧 N4 对官方差异 |
| 当前整例严格复现 / 整体等效 / 600 秒 | 未评估 / 未判定 / 未达到 | 本页不能据导入成功宣称完整重建成功 |

完整安全字段和原始私有收据 SHA 见 [raw_t1_failure_v1.json](raw_t1_failure_v1.json)。修复后的合同命令、代码 SHA 和 JUnit 见 [contracts.json](contracts.json)、[contracts.xml](contracts.xml)。影像阶段精度和脑图引用对应真实运行报告，不由接口测试生成。

服务器的实际安装、安全字段和原始收据 SHA 见 [cfff_installation.json](cfff_installation.json)。冻结控制器 [verify_wheel.py](verify_wheel.py) 接受必填 `--workspace`、`--run-directory`，从新 workspace 的 wheel、`expected.json` 及 [verify_entry.private.py](verify_entry.private.py) 安装到新 run；该入口脚本无私有内容，验证实际模块、21项配方、所选 N4 依赖与 fresh CLI。控制器只离线安装，不构建或下载依赖；失败不写成功 COMPLETE，日志留在本次私有目录。服务器已有固定分发包恢复在另一私有 target，旧环境 prefix 未改写；1核、CUDA不可见，不能证明干净 Conda 完整运行。完整 N4 输入、输出哈希和运行状态见[阶段依赖回归](../20261011_n4_dependency_preflight/README.md)。

失败运行的显存采样请求间隔 0.5 秒，实际最大间隔 1.281 秒，共 96 样本。目标卡计算进程合计观察峰值 8,212,447,232 字节、设备占用峰值 8,227,127,296 字节；容器与宿主进程归属未解析，父子 FNIT 同时占用峰值为 `null`，不是零，也不是整例 20 GB 通过证明。PyTorch allocated/reserved 及原始采样边界在机器报告中保留。

## 6. 更新记录

- 2026-10-11：原始公开 T1 启动暴露配方依赖缺失；保留其失败报告和原包/包装代码版本。
- 增加所选 N4 依赖的提前失败，实际安装验证由八模块扩至十模块，并检查真实 SimpleITK 扩展库。
- 恢复服务器已有固定版本分发包到独立私有依赖目标，不修改运行中环境 prefix；阶段和新整例分别验收。
- 历史导入检查、数字拓扑整例与 GA 同输入阶段结果继续按各自哈希保留，不改标为本次整例通过。

## 7. 参考文献与原代码

- [FNIT recon-all 源码](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/main/src/fnit/recon_all)。
- [SimpleITK N4 接口](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html)。
- Tustison NJ et al. N4ITK: Improved N3 Bias Correction. IEEE TMI 2010, 29:1310–1320。[DOI](https://doi.org/10.1109/TMI.2010.2046908)。
- [固定 FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
