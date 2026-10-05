# FNIRT CPU level 重建：准备阶段1

## 1. 目的

只在第二 accepted 点（solve3）重建当前 CPU 线性化状态。现有组装仅保存 H、g、diagonal，未保存当前影像状态或 matrix-free 闭包。旧预处理数组可作为候选输入；通过逐位桥接前，不能称为已恢复缓存。

## 2. 输入及来源

使用已绑定的 solve3 F64 参数、旧 firstdiff 保存的 F32 smoothed_moving/initial_fixed、四个实际资源的头信息/affine、完整 GM config。旧 firstdiff 第一 accepted 参数与 solve2 相等，与 solve3 不同；此次仅复用不依赖系数的预处理数组。当前 optimized CPU blur 与旧源码不同，必须保留这个前提差异。输入 SHA、源 SHA、目标状态见 expected.public.json。

## 3. 调用边界

单次 system.linearize，内部一次 evaluate；仅该实例覆写返回状态的 effective_lambda。不重新平滑或归一化影像，不执行 direct_gradient、H 物化、matvec、PCG、原生进程或完整配准。当前生产和 GPU 均不修改。F32 影像及 F64 系数/基函数/梯度/权重遵从当前实现。

## 4. 接受门

count、SSD、bending energy、覆写前 lambda/cost 准确匹配，再检查 2g 和独立 2diag 的全部 F64 位（包括 signed zero）。第一个不匹配保存 scalar 诊断并非零退出，不尝试其它预处理或点。源码/输入/精度状态前后必须一致。

## 5. 私有状态格式

门全部通过才保存 level_state.private.npz：本次实际 state、basis、空间/cross weights、bending Gram/diagonal 和几何。moving 数组用原保存文件的路径/SHA 引用。保存的是数组及 metadata，不是原 Python cache 身份；CPU normal 的 packed layout 和 scratch 尚未初始化，因为没有调用 callback。未来 loader 和 callback 数值桥另行审查。

## 6. 调度与计时

nodecw7 八个 physical 核，共用 outer CPU lock 后取自身 inner lock；排既有 GEMS 与 Seg 任务后，child180s、20GB address-space cap。尚未启动。等待、外层 preflight、导入/哈希、恢复/几何、evaluate、linearize、桥接和 private checkpoint 写出分别记录；evaluate 包含于 linearize，不重复相加。

## 7. 当前状态

这是待 root 审查的源码与计划，不是科学验证结果。唯一科学阶段须在审查后授权。run_stage1.sh 的授权参数只防误启动，不代表获得批准。公开报告不包含影像、cache 数组、原软件源码或二进制。
