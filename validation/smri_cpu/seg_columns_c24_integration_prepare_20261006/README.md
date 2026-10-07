# C24 窄生产接入与有限验证准备（2026-10-06）

## 1. 功能与当前状态

草稿仅为 `SegmentUNet.down[0].conv1` 的实际权重与完整 CPU shape 接入 C24 列缓冲复用。成熟 C72 数学、资格、符号、builder 和 cache key 保留原字节；C24 单独导出符号和缓存。

[真实层结果](../seg_columns_c24_real_results_20261006/README.md)已通过：六次全输出 bit0，10.889→5.003 秒。当前新增的是生产代码、software 合同、两个 metadata compile/reuse worker 和一次 CPU ABBA/GPU AB 计划。**本阶段未上传、未编译生产 DSO、未运行 MRI/完整 CNN/GPU**；根任务审查后分阶段一次执行。

流程为独立生产 cache 冷编译及暖复用 → 同一真实 T1 的 CPU A1/B1cold/B2warm/A2 → GPU A/B。编译或任一完整臂首次失败保存原回执并停止后续，不重试、不放宽阈值。

## 2. Python、输入与输出

公共接口、全部参数、输入与输出、精度和 fallback 见[七节功能文档](../../../docs/synthseg/CPU_COLUMNS_C24.md)。只查看计划的例子如下，不运行模型：

```python
import json
from pathlib import Path

c24_prepare_directory = Path("validation/smri_cpu/seg_columns_c24_integration_prepare_20261006")
c24_execution_plan = json.loads((c24_prepare_directory / "PLAN.json").read_text())
c24_cpu_arm_order = c24_execution_plan["whole_CPU"]["arm_order"]  # baseline/cold/warm/baseline
c24_gpu_arm_order = c24_execution_plan["whole_GPU"]["arm_order"]  # baseline/candidate
c24_new_dependencies = c24_execution_plan["packaging"]["new_dependencies"]  # 0
```

公开 PLAN 只含代码与指标/资源约束；实际 MRI/权重身份、环境 prefix、GPU UUID 和原 cache 文件身份在私有 bindings。私有 bindings 必须绑定最终公开 PLAN 的 SHA，由根现场核对。已有 C72 contract cache 的文件大小/SHA已只读核验，保持原实体目录；不删除或复制以制造 cold。

`root` 是现场统一入口；`source/baseline/candidate` 是各冻结源码的 `src` 目录；`workspace` 是本叶 worker；`plan` 是本公开 PLAN；`bindings` 为对应私有身份文件；`run` 必须全新且不存在。`output` 是单臂全新目录。各 approved 开关仅在根新批准对应阶段后传入。编译模式 `cold/warm` 只在新 C24 contract cache 使用两个新进程，第二进程不再编译，不进行张量数学。

完整臂输入相同原 T1，输出完整 `segmentation.nii.gz`、`volumes.csv`、WHOLE/退出回执与日志；与参考逐文件同名 bytes/SHA 相同才派下一臂。原始影像、分割文件、CSV 和 runtime binary 留私有记录。

## 3. 有限 CLI 与资源

当前仅复核计划：

```bash
python -m json.tool validation/smri_cpu/seg_columns_c24_integration_prepare_20261006/PLAN.json
```

待根审查后的冻结队列为 `compile_queue.py --approved-compile-contract`、`whole_queue.py --device cpu --approved-whole` 和 `whole_queue.py --device cuda:0 --approved-whole`，完整必填参数见上一节。运行目录存在即拒绝，保留旧 source/cache/prepare/run。

| 阶段 | worker | worker 秒 | outer 秒 | CPU/RSS | 新编译 |
| --- | ---: | ---: | ---: | --- | ---: |
| C24 metadata 冷/暖 | 2 | 180 | 720 | 8 GB AS；worker/child 各 4 GB RSS，线程状态只读保持 | 1/0 |
| 普通 33 类 CPU ABBA | 4 | 600 | 3300 | 同 8 物理核，32 GB AS/RSS | C24 1/0；C72 0 |
| 同默认精度 GPU AB | 2 | 300 | 1020 | 原 8 核亲和性；allocated/reserved/本人树采样各 20 GB | 0 |

各组 lock 等待最多 120 秒，CPU 与 GPU 使用既有共同锁。outer 包含 lock 等待与 child 收尾。编译 contract cache 与完整 CPU 的新 C24 cold cache 分离；后者 B1 编译开销计入完整时钟，并单列真实 compiler 秒。C72 使用已验证 warm 身份，任何新 C72 compile 不通过门槛。

新增依赖 0。新 CPP 为原验证源码相同 4,396 B；复用成熟 provider/GCC11/ABI0/header/flags/lock/atomic 帮手。Conda compiler flags/linker 参数保持成熟版本，旧 C72 三源未修改。wheel 已有通配 source；根授权本分支在 `MANIFEST.in` 明确包含 `_columns_c24.cpp`，setuptools FileList 静态 source inclusion 已通过。根整合后核对实际 sdist/wheel 文件清单；新 Conda 安装仍待验证。

## 4. 原软件对应

内部 conv1 没有独立官方 CLI。完整参考是 `mri_synthseg --i input_T1w.nii.gz --o labels.nii.gz --vol volumes.csv --cpu --threads 8`。本计划不启动官方程序；同节点同线程的已冻结官方结果供后续保存输出评分复用。生产不调用 FSL/FreeSurfer 或封装包。

## 5. 固定精度、时钟、输出和内存门

新代码只接受已验证权重、`[1,24,192,224,256]`、CPU FP32、8 intra 线程、eval/no-grad、oneDNNFalse，原 padding/stride/dilation/groups，无 autocast/hooks/subclass/forward AD/懒视图。数学 AST 与已验真实原型相同。fallback 资格失败与准备失败发生在数学前；候选已开始的错误不重复卷积。

CPU 每臂实际 2 个完整网络 forward；C72 每臂 2 层命中、28 copy/SGEMM，C24 仅 candidate 每臂 2 命中、12 copy/SGEMM。CPU 总 8 forward、C72 8 命中/112 copy/SGEMM、C24 4 命中/24 copy/SGEMM。GPU 总 4 forward，4 个 optional CPU 模块均未 import，build/compile/probe/copy/SGEMM=0。完整阶段解码 MRI 共 6 次；编译 metadata 解码/张量数学=0。

CPU clock 分列 preflight/construct/API/save/full worker、C24 compile 秒及冷进程墙钟；C24 单层旧倍率不外推整例。GPU保留原 cuDNNTrue/前向 matmulTrue、FP32/noautocast 和恢复；同 UUID、allocated/reserved 字节相等，driver采样有效非零、全部本人树≤20 GB，保留 interval/maxgap/failure。共享 GPU 单对计时只列观察。

每次先保存完整图/CSV和数值回执，再检查退出及 full gzip/CSV bytes/SHA，固定全部 NIfTI header/extension、整数标签和CSV字节；源码、6个资源和旧 C72 cache 前后相同。新脑图尚无；实际完整输出通过后才另获根后验评分/绘图批准，使用既有 scorer，不再跑 CNN。

## 6. 软件检查与版本记录

原 C72 加新 C24 guard/cache 合同 **56/56**；新 C24 队列与资源控制 **8/8**。它们仅用小 Tensor metadata/mock 与文件 fixtures，0 compiler/SGEMM/MRI/full CNN/GPU。队列测试首次误指向旧 harness，被 argparse 在 child 前拒绝；只修 fixture 引用后 4 项通过，没有科学重跑或阈值更改。独立审查后补齐实际前向 device/dtype/TF32/autocast 门、同锁 FD 继承、有限 KILL/wait 与 child_reaped 门和整个 controller 的 alarm；4 个本地资源合同同时通过，其中真实本地子进程证明父 FD 关闭后锁仍持有。外层 supervisor 由根有限启动。

[STATIC_CHECKS.json](STATIC_CHECKS.json)与[freeze.public.json](freeze.public.json)绑定候选 source、worker、文档、公开 PLAN 和本地软件检查。冻结真实层 prepare/run 不修改，根真实审查已通过其原回执。

打包 include、实际生产 cache 编译与全部完整结果当前 pending；不能把 software 合同写成真实生产验收。完成后由根统一修改功能页当前状态、总表与 main。

## 7. 来源与参考

自有 C24 CPP 和数学原型来源：[C24 准备](../seg_columns_c24_prepare_20261006/README.md)、[短合同](../seg_columns_c24_contracts_20261006/README.md)、[真实层](../seg_columns_c24_real_results_20261006/README.md)。新的 cache 复用项目成熟自有 builder，不复制分发原软件实现体或动态库。

- [SynthSeg](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis* (2023)。
- [PyTorch 2.5.1 ConvolutionMM3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。
- [胶水归属](../../../src/fnit/synthseg_parc/CPU_COLUMNS_NOTICE.md)。模型许可沿用外置资源，未增加再分发权利。
