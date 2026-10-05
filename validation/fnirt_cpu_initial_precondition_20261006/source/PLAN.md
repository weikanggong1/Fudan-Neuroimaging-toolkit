# FNIRT 初始 z / rho / norm 最小诊断：仅准备

## 1. 目标和已有事实

Stage3 报告提交 `4ee61ed34d28e3c48da144c72ac84b3b8f7a56ad` 的两同系统自产 step 相对差6.97%，旧49/new53自然停止，true relative residual 均低于1e-3。保存 ledger 中两 unit0 action 相同，但首次 PCG direction 已不同。此次只准备一个 bounded initial-state 控制；未上传、登记、派发或执行数学，协调者审查 freeze 后才可另行授权。

首 direction 生成依赖 old inner-floor→reciprocal×rhs 和 candidate direct division，rho dot 结果尚未参与 direction 更新。本轮查明该初始路径及同一 z 上的 scalar 归约；不能把首叉写成最终6.97%的唯一原因。当前 RHS 对 official 的7.4e-7 来源尚未解决，本控制不代替该主要精度问题。

## 2. 输入和公式

仅读 accepted solve3 NPZ 中 `gradient_half` / `diagonal_half` 两个数组；只读取 archive index，不 materialize 其余66数组。另读取 producer summary 和 Stage3 原始 summary 的 schema / SHA / scalar/hash字段。输入以 `expected.public.json` 的大小和 SHA-256 绑定；current17生产源及已发布 candidate845a/own3faf 原封绑定。

恢复这两数组原逻辑值和记录 stride，桥 Stage3 的 operand hash。与 Stage3 同式：

```python
# CPU FP64/no-grad，g/diag由accepted checkpoint读取。
tau = 0.001  # 原Stage3受控LM值；不是已证明的历史solve3 damping
floor = torch.finfo(diagonal.dtype).eps * diagonal.abs().mean().clamp_min(1)
damping_diagonal = diagonal.clamp_min(floor)
rhs = -gradient
preconditioner = (1 + tau) * damping_diagonal
```

三个派生输入的逻辑值 SHA 必须等于两臂 Stage3 保存的 `system_values_before`。不读 H、native 解/A/RHS、原MRI、旧paired解或轨迹。

## 3. 源码抽取与数量

`prefixes.py` 对当前 `optimizer.py` 和 candidate `.py` 抽取第一且唯一 top-level `for` 之前的原始 statements，保留参数检查、precondition、direction、initial rho/norm 的表达式和顺序。函数改名，仅在最后追加一个 slots return；没有 loop/post-loop body，prefixAST另绑定。candidate 的 `cpu_vector` 原函数同 AST 编译。current17源/候选源 SHA 仍是最终实际证据，AST不是跨源码精度证明。

旧 prefix 的 norm/dot 经局部 namespace proxy 转发实际 `torch.linalg.vector_norm` / `torch.dot`；候选转发原自有 norm/dot。proxy 只计数、记录 scalar hex/输入hash，不 monkeypatch全局Torch。零RHS返回 PCGReport 明确拒绝，真实 bound RHS须非零。

最多两 prefix（各一次）生成：

- old prefix：实际 inner floor / reciprocal×residual / direction，norm1、initial rho1。
- candidate prefix：实际 NumPy division / direction，norm1、initial rho1。
- 两 prefix direction SHA 必须等于 Stage3 call2 保存值；首不匹配停止后续交叉。
- 两门通过后，只再对 old z 用自有 dot、对 new z 用 Torch dot：总 **4 dot / 2 norm**；两组各自比较的 residual/z 值hash完全相同。

**0callback、0PCG、0evaluate、0linearize、0new H、0native、0image、0GPU**。`matvec` 参数传入拒绝函数，prefixAST不含任何 loop；没有 extra solve或更紧tol。

## 4. 要记录的控制

记录 old 内部 `preconditioner.clamp_min(inner_floor)` 是否实际改变数值或bits、active计数和 supplied/effective weights hash。记录：

1. actual old reciprocal×rhs 对 actual candidate NumPy divide 的bits/max/orderedULP；
2. actual old 对同一 old effective weights 的 direct divide（只追加一个向量表达式，不prefix/reduction/callback/solve）；
3. actual candidate divide 对上述同 effective weights divide，用来区分 inner floor active 与表达式舍入；
4. 同一 old z / new z 上的 Torch、自有 initial rho，实际各个 RHS norm的64bit hex。

orderedULP按有限FP64的单调整数编码处理，signed zero保留区别；这是计算舍入量，不是 voxel 误差。不写数组或新checkpoint，不下载输入/向量。

## 5. 调度和失败

将来新的 canonical leaf 为 `smri_cpu_20261004/remaining_20261006/fnirt-initial-precondition-v1`。当前仅本地准备，暂无INDEX项。nodecw7 physicalCPU8 `32,36,40,44,48,52,56,60`、OMP/OpenBLAS/MKL/Numba8、Torch8/interop1；同commonCPU锁再module锁、outer wait19000/controller19600/child180s，20e9B AS上限（20GB，约18.63GiB）。保留环境prefix，CUDA invisible，不改TF32/默认dtype/grad外部状态。

umask077、目录700/文件600；已有output一律拒绝。任何源/input、schema、值/stride、初始系统hash、call2direction gate失败均停止，不自动变floor/线程/dtype或retry。成功或异常finally均记录已有operand afterhash / source/input / flags / FSL DSO前后身份快照，异常保留原science RC；runner仍执行after preflight。

## 6. 输出和结论

将来只输出 `initial/summary.public.json`、preflight/exit/queue原记录。`valid_bounded_diagnostic` 仅指有限控制完整；不批准runtime、更不判最终解准确性或解释RHS组装7.4e-7问题。没有速度门；JIT/导入时钟只记录scope。不根据参考长度、69轮、地址或被试特例选择算法；维数由数组读取。

## 7. 当前检查与来源

这里只做 AST/compile、finite JSON、bash-n、源码SHA和原prefix AST结构检查，不导入 NumPy/Torch/Numba，不调用prefix，不运行数学。引用当前FNIT自有PCG及已发布 private source；NumPy/Numba/llvmlite均来自主页Conda现有链，无新依赖、官方代码、对象或安装FSL链接。
