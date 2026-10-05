# 下一步 CPU 接线候选：静态方案，尚未执行

## 现有接入口

`src/fnit/fnirt/optimizer.py:28` 的 `preconditioned_conjugate_gradient` 接受任意 matvec 回调。当前 CPU 与 CUDA 共用 torch.dot、torch.linalg.vector_norm；对角预条件先取 reciprocal 再相乘，beta 分母带 tiny clamp。CUDA optimized 有单独批量回传两个标量的分支。本轮自有代码尚未接入这里。

`registration.py:1696–1722` 每次 LM 线性化构造 damped 回调和预条件对角线，再调用 PCG，传入负 gradient。`registration.py:1082` 的 matvec 仍由 spline expand/adjoint、数据 normal 和 bending 组合；`_normal_cpu.py` 优化的是体素 FP64 normal 乘法，没有返回 CSC Hessian。当前 production 没有现成的严格 CSC 全系统 matvec，不能把本叶保存 A 的 CSC 回放当作该回调已经通过。

## 单一最小候选

先在独立 harness 写一个自有 FP64 CPU PCG 函数：复用本叶的通用 Numba dot/norm，预条件用除法；向量更新保持分开的乘法/加法；保留现有 tolerance、max_iterations、零 RHS 与失败返回语义。只接受无梯度的 FP64 CPU 张量/数组和 CPU matvec。生产接线待后续门通过后再讨论；若接线，CPU optimized 分支 lazy import，CUDA、reference、可微调用与其他 dtype 继续现有分支。

运算顺序由长度的 32/16/4 分块定义，LLVM fma 保证融合语义；没有形状 1177、迭代 69、参考 trace、地址对齐或安装 FSL 检查。现有证据只约束所见 COOPERLAKE unit-stride 分支。更大向量可能进入原生 OpenBLAS 线程分块，其他 selector 也可能改变归约；未证明前不能把这一顺序称为跨 BLAS/CPU 的原生位等价策略。

## 下一次唯一真实状态回放

既有 `fnirt-shared-followup-v2/assembly` 的 H、gradient、diagonal 三个 FP64 文件仍可读取，SHA 见 [next_probe_prerequisites.public.json](next_probe_prerequisites.public.json)。它们由 `caab8ffc…` 的当前组装代码在第二 accepted 参数、有效 lambda `9049.463427795125` 上保存；不需重新预处理、线性化、组装或调用原软件。

建议只回放一个 current LM 系统、两个求解臂：

1. 由保存的 full H 和独立保存的 full diagonal 构造 `A = H + 0.001 * diag(diagonal)`，预条件为 `1.001 * diagonal`，RHS 使用保存的 full current gradient。两臂共享同一严格 CSC matvec、同一输入、原 `1e-3/500` 门；旧臂调用现 production PCG 算术，新臂用自有函数。原生存档解仅在终点评分。
2. 报告自然轮数、递推/真实残差、旧新解差和各自对 native 解的误差；这里比较正 g 的内部求解，生产更新使用负号。不要直接用 materialized H 的对角线替代独立 diagonal：旧报告已发现 840 个末位差。不要把保存 H 的列物化结果称作当前 matrix-free 回调在任意方向上的逐位证据。

该配对隔离当前已组装系统的 PCG 算术作用，不解决 RHS/H 本身的差异。即使终点改善，也只进入 CPU 接线候选阶段，不能宣布完整非线性估计或最终 warp 等价。

## 接线前的小合同

在获批下一轮计算后，用独立准确舍入定义核对小长度和 32/16/4 边界、零 RHS、非正分母、无梯度/设备/dtype 守卫；这类合同是算法检查，不算 MRI benchmark。当前 93 行没有覆盖 n<=32、大向量线程分块或其他 CPU selector。GPU 源码与已有数学分支需要字节/调用图保护，不能因 CPU 候选修改默认 TF32 或停止容差。

本文件仅为静态方案；没有启动这些合同、current 系统求解、MRI 或原生进程。
