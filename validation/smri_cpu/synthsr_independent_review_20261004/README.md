# SynthSR 集成独立审阅（2026-10-04）

## 1. 范围与结论

只读审阅 root 当前已接受的 SR 生产源码、真实 CPU/GPU 报告、整合测试绑定及打包规则。当前六个 SR Python 文件逐一等于冻结 `source_sr_v2`；未发现需要修改生产的新增风险。具体身份和检查结果见 [review.public.json](review.public.json)。本轮没有重跑真实推理或 2,092 项组件回归。

## 2. CPU 入口与回退

主 forward 只在 CPU FP32/no-grad 时延迟加载分派；helper 进一步要求 Linux x86-64/SSE2/FMA、MKLDNN、单 volume、eval/no-grad、原 Conv3d/BatchNorm3d 结构和 FP32 参数/BN buffers。训练、梯度、CPU autocast、其他 dtype/设备、替换叶层和不支持的平台沿原调用。

局部/全局 forward-pre、forward、backward-pre、backward hooks 均检查，包含模型与所有子模块。Conv 的 instance forward override、padding_mode、stride/dilation/groups 和 BN affine/running-stat 状态均有回退。原有测试已核对 hooks、训练梯度与 buffers、autocast dtype/数值、参数身份/内容/stride/version/RNG；这些测试文件 SHA 包含在当前整合记录中。

## 3. CUDA 与状态

独立 AST 核对 `626cab69` 原 forward 与现版删除唯一 CPU 分派前缀后的正文完全相同。非 CPU 输入不导入 CPU 数学 helper。临时 CL3D 权重在每次 CPU 卷积时创建，没有替换或缓存公开参数，避免 CPU→CUDA 时带入新 stride。原 checkpoint、参数加载和 CUDA TF32 政策的文件 SHA 未变。

真实 GPU 原/new 完整 CNN、浮点、uint8、headers/文件 SHA 以及 allocated/reserved 相同的报告绑定这六个现版文件。该门是保持旧 GPU 行为；没有把它改称与官方 CPU 逐值等价。

## 4. 线程、依赖与打包

NumBa mask 取 `min(既有NumBa mask, Torch线程数)`，在 `finally` 恢复，不修改 Torch、OpenMP 或 CUDA 设置。已有成功/异常恢复测试的实际源码与整合记录匹配。

主页 Conda 和项目依赖已有 `numba>=0.59`，它声明 llvmlite 依赖；没有新增 TF、oneDNN/Eigen headers 或原软件运行依赖。本地审阅 NumBa/LLVM 为 `0.67.0/0.49.0`，真实 node7 ISA 记录为 NumBa `0.61.2`。最低声明版本 `0.59` 本轮未重新安装测试，不把当前版本的实测推广为所有后端/版本的逐值保证。

setuptools `build_py.find_package_modules` 现场枚举包含 `_cpu_inference`、`_cpu_math` 及其余四个 SR 模块，私有文件名前缀不会将其排除。`license-files` 声明 licenses/*.txt 和 THIRD_PARTY_NOTICES；MANIFEST 包含 license 文件，排除外置权重。这里只核对模块发现和现行元数据/manifest，没有重新构建完整 wheel/sdist；科学 validation 主要在 repository 提供。

## 5. 实测 source 与整合记录绑定

整合记录的 **278 个 source/test SHA 全部匹配当前文件**，六个 SR 文件也全部匹配真实冻结 `source_sr_v2`。因此“2,092 passed、5 skipped”可绑定其当时测试源码，并与当前 SR 真实 CPU/GPU门连接；它本身是组件回归，不是整例科学 benchmark。

原始官方 whole-graph oracle、CPU 完整默认输出、四参数和六真实域的原门没有变化。EPI NPZ 保留已报告的浮点微差；GPU 共享负载和官方冷 CLI 的 I/O 波动仍不支持稳定加速断言。测试总耗时不替代正常 CLI 时间。

## 6. 发现与处理

发现 THIRD_PARTY_NOTICES 曾写“路径不调用 oneDNN library”，范围过宽，因为保留的 Torch Conv3d 使用其 MKLDNN 后端。已通知 root，root 将文字收窄为：自有 ELU/BN 调用 NumBa/LLVM，卷积继续使用 PyTorch 既有 oneDNN 后端。未修改生产数学源码，source hash 保持冻结。

实际 notice 已标明 oneDNN 2.7.3 的 Intel 2019–2022/Apache-2.0、上游 LICENSE 附加版权，以及 Eigen SSE 的 Gael Guennebaud 2008–2009/MPL-2.0；命名的修改源码和两份 license 均存在，SHA 留在 JSON。此处记录源码归属与打包事实，没有复制上游 headers 或改变资源再分发边界。

## 7. 审阅边界

本轮没有新增运行时风险，因此不重复已通过的 22 项 SR contract tests、真实 CPU/GPU 推理或整合套件。没有测试任意第三方 class monkeypatch、TorchScript/torch.compile、所有最低依赖版本或新部署环境；现有功能页没有把这些环境标为本轮逐值验收范围。生产源码只读，root 的并行未提交文件保持原样。
