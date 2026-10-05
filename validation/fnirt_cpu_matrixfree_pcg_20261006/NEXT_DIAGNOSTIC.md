# 首差、原因假说与唯一最小后续建议（未执行）

## 已有事实

同一个 restored half matrix-free controlled-LM 系统，旧49 / 新53自然停止，两个 true relative residual 均低于 `1e-3`，但两自产 step 相对差6.97%。没有 native 同系统解，没有 outer LM 接受或完整配准，不能判哪条 step 更准确。

两臂 rhs/preconditioner 的值 SHA 相同；两个 unit direction/action SHA 相同。保存的首个 PCG callback direction 已不同：

| 首个 PCG callback | 当前 Torch | 自有候选 |
| --- | --- | --- |
| direction SHA | `e63284f6f4198c09827ee9815aae3d9b1720585a88d658c6eee129c382749cf4` | `b3fea96e550d2a7c35369a62e942775b6345524a028a7529af57d9075aaff7ea` |
| damped action SHA | `a06bff8f872e0c3b00bfe765d61767d4925cde41081427440f48678d6694dac5` | `dc65dbd9b994430d47cd5473ae2bc7a52606c28586faeebd03482195e2c0fbfa` |

这只是保存 SHA 的只读比较，没有读取或重算 direction 数组。旧源码先生成 `inverse_diagonal * residual`，候选先生成 `residual / weights`；rho dot 不会改变已经生成的 direction，归约源码也不写入输入。因此首次 vector 分叉已定位到预条件表达式路径，而不是依靠轮数变化猜测 dot/norm。

## 假说及证据范围

1. **预条件舍入是首叉来源。** 旧 inner floor→reciprocal×rhs 与候选 division 的表达式不同，首 direction SHA 的时间依赖和 source 一致。尚未分别测量 inner floor 的 active 元素数，以及 reciprocal×multiply 与 direct divide 的 ULP 差；不能区分这两个子原因，也不能说它独自造成最终6.97%。
2. **后续 rho/den/norm/update 可以传播首叉。** 两候选保留各自 dot/norm 和 beta floor/表达式，已有 source binding 证明算法范围不同。本轮未保存逐步 rho/alpha 标量，不能用最终轮数倒转给某个归约分配唯一责任。
3. **残差门可能允许相距较大的 step。** 两解的实际残差都小，但小 residual 本身不保证小 forward error；需要算子方向或谱性质。此处未测全局条件数、最小特征值或这两 step 的方向增益，不能断言严重病态、全局原因或算法 bug。
4. **callback 舍入路径与 materialized H 不同。** Stage2 已实际记录七方向 optimized/reference bits相同，但真实组合方向对 strict CSC 有1e−15左右绝对尾差；本轮 half/negative-RHS 系统与旧 full/positive-RHS materialized 回放不是同一个浮点系统。该背景允许 PCG轨迹变化，但不是这两 step 首叉的独立隔离证据。

## 唯一建议：初始预条件和标量交叉控制

只建议一个新的有限 worker，另经审查授权后执行；目前未准备新的科学 harness，也未派发。

- 仅读同一 accepted checkpoint 的 `gradient_half`/`diagonal_half`、Stage3 源/summary，使用原 `.001` controlled tau、原 floor 和相同 rhs/preconditioner；绑定原17生产源及候选845a/3faf，输入前后 SHA 不变。维数来自数组，不固定1177、地址或被试。
- 用实际旧 Torch 表达式及实际候选 NumPy division 分别重建首 `z`；各自 value SHA **必须**逐位桥接上表 direction SHA。首不匹配即保存标量停止，不能自动改线程、dtype、floor或重试。
- 只记录旧内部 clamp 对 supplied preconditioner 实际 changed-count；在同一 effective weights 上分别对比 reciprocal×rhs 和 direct divide 的 bits / max / ULP。这隔离 floor 与表达式舍入，不是新 PCG。
- 同一固定 `rhs/z` 上交叉 Torch / 自有 rho dot，以及相同 rhs 的 Torch / 自有 norm；数量固定为最多4 dot +2 norm，记录 scalar bits。没有 callback、PCG、new H、evaluate、linearize、MRI、native、eigen/direct solve 或更紧 tolerance。
- 同 CPU8/common lock、私密700/600、child180s/20GB/无CUDA，source/flags/operand 前后检查。元数据返回，不下载向量。该诊断只能定位初始预条件/标量差，不能证明最终方向准确性或批准运行时接入。

若初始桥失败，停止并报告恢复缺口；若桥通过，给出 active-floor/ULP 和相同输入标量结果后再决定下一步。不自动扩大到完整求解或矩阵重组装。
