# FNIRT stage2：从 accepted level checkpoint 恢复 callback（待审）

## 1. 当前状态

仅冻结计划和本地源码，未上传、未登记 INDEX、未派发 worker，未执行数学。stage1 已经一次通过，报告提交 `0b691f32bdc47dc68f30516daf7d4e9f369e3235`。本计划不接入生产 CPU 或 GPU。

## 2. 输入与版本

只读取本次 accepted NPZ（SHA `78abee85…`、2647508B、68数组）及其 producer summary（SHA `eb31cfd1…`），已保存当前 full H/g/独立 diagonal，以及此前 current 系统 old/new 两解。所有路径和完整 SHA 在 `expected.public.json`。不读 native solution/A/RHS，不重跑官方。

绑定17 FNIRT源码、现有 coordinates，以及已发布 strict CSC body 所在 `replay_current.py`，前后27项来源/输入哈希保持一致。Git HEAD 只记录现场版本，不作为 cache 等价依据。原 coefficient point 是 solve3 第二 accepted 点；当前 H/g/diag 已由 stage1 完整 scalar/full-vector 位门桥接。

## 3. 状态恢复与 closure

NPZ加载只保证逻辑值。本 loader 按 producer summary 的 byte strides 显式 `torch.empty_strided`，copy后对68数组逐一核 shape/dtype/stride 和逻辑值SHA。完整 `2*gradient_half`、`2*diagonal_half` 再与已存 current 文件逐字节比较；失败即停。

原 Python alias graph 未持久化。两个臂分别恢复独立数组，互不共享 storage；每个数组独立可读，所有 checkpoint 操数在 callback 前后值SHA必须一致。mature callback只读取这些 basis、weights、Gram 等操作数，optimized臂自己的可变 scratch仍保留其成熟内部语义。原始packed layout/scratch为空，第一实际callback才初始化。

从绑定的当前 `registration.py::_LevelSystem.linearize` 抽取 `bend_normal`、`data_normal`、`matvec` 三个原 AST，顺序和标准化AST哈希固定。用明确的 context 工厂构造真实 closure，逐一核 `co_freevars` 的名称与顺序。self只提供保存的 bases/bending/fixed/estimate_scale，BendingOperator用`__new__`加已存Gram/diagonal/derivative-basis初始化，不调用构造函数重新计算。原body中的 `expand_coefficients`、`adjoint_field`、pack/unpack和bending.normal仍来自当前 FNIT。

本轮两臂只隔离 pointwise 路径：

- optimized：新建现有 `SpatialNormalCPU`，使用原判定、布局打包与scratch。
- reference控制：强制`cpu_normal=None`，执行同一`data_normal`内成熟的torch pointwise分支；bending及scale reduction与optimized相同。

原body不修改数值公式。注册 evaluate/linearize/gradient、Bending构造和PCG拒绝钩子，任意调用即失败；无 raw image、几何、blur、权重组装或梯度重新计算。

## 4. 方向、第三比较和接受边界

维度从已保存 current g 读取。首先仅三个单位列：0、n//2、n−1（去重）。optimized callback的full结果必须与既存dense H对应列逐位相等；首次不匹配保留scalar诊断并停止。三个原列通过只是有限恢复桥，不证明全部列，更不证明任意方向的浮点执行相同。

随后仅四个真实保存方向：current full g、old解、new解、new−old。原臂/控制臂各7个callback、第三CSC7次，最多14个callback与7个已有矩阵action。不做迭代求解，不产生新H列。

第三比较是已存full current H转CSC；抽取既有 `column_matvec` 原body AST，用已有Numba编译cache=False/fastmath=False，保持列顺序mul/add。它不替代当前callback，也不是新的原生oracle。每个方向报告三套值SHA、bits、maxabs、relativeL2及finite；任意方向对CSC或reference的差异如实记录，不以舍入差自动判为算法错误，不设置新的配准容差/接受门。

最终有效诊断还要求68操作数值不变、来源/输入/flags前后相同、CUDA未初始化、所有禁止调用计数0。该有效状态不代表候选CPU runtime通过或全链精度改善。

## 5. 调度、权限和计时

未来独立 canonical workspace/run：`smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-restored-v1`。审查批准后才能上传新文件和在六INDEX锁下登记自己的新叶。

nodecw7物理核32,36,40,44,48,52,56,60，OMP/OpenBLAS/MKL/Numba8、Torch8/interop1；CUDA_VISIBLE_DEVICES为空，禁 loader overrides。共用outer CPU lock后取独立inner lock，outerwait19000s、controller19600s、sciencechild180s，20GB address-space cap；不抢占其它任务。umask077，上传代码600、目录700，新output必须不存在。

时钟分别记导入/绑定、两套restore/closure工厂、每个callback/CSC及worker总界。第一callback可能包含Numba JIT与打包；顺序固定是数值控制，不是ABBA性能比较。前后FSL DSO身份快照只作为映射记录，不扩大为全程动态审计。

## 6. 预检与停止

本准备阶段只运行 stdlib AST/JSON/compile、bash-n、源码SHA静态检查，不运行callback/Numba kernel或模型。审查之后的唯一worker遇到首个输入/布局/full-g/diag/unit列门错误立即非零退出，不改blur/λ/方向或重试，不启动其它数学worker。异常summary保存已完成门和计数。

## 7. 尚未完成及后续

本计划没有matrix-free实测结果，没有新的PCG、native、完整配准或性能结论。即使有限callback比较完成，也不能把materialized H回放的自然53/49轮结果转写成生产matrix-free求解。后续CPU接线必须由这次证据另行决定。
