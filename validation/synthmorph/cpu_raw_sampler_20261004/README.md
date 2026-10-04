# SynthMorph CPUjoint raw 八角采样：独立 NumBa 原型

## 范围与源码

仅优化 v29 CPUjoint 网络预处理的 FP32 raw-coordinate 三线性采样。生产文件、CUDA、共享 world resampler、最终影像采样和 CNN 未改。候选在 [raw_sampler_numba.py](raw_sampler_numba.py)，有限 worker 在 [benchmark_raw_sampler.py](benchmark_raw_sampler.py)。

既有 v29 `_cpu_preprocessing.py` SHA `9f531efabf8c992e2f1ee0d785ad6655bb09570d75a6e071f7c644e18a1b236a`；`spatial.py` SHA `0dd342cf0a9c94cd09a0396097adc40afbb838cfbce2f0963a28f417249d93a5`。本原型 SHA `b50a0d399da235a468bcee712296537e18e2b57c2e830229f63e4fb059a33caf`，worker SHA `ca88fad7a986eed0be3d59122806cf5ff5dafe0dd137bdb21b546875977c8127`。

原 Torch `grid`、`_dense_from_grid` 的 einsum 与 `coords + shift` 原样调用。采样在 NumBa voxel 并行循环中融合，`fastmath=False`，每个角依次执行 `wi*wj`、`*wk`、`*value`，再按 `itertools.product((0,1),repeat=3)` 的次序左累计；每步为 FP32。floor 后 clamp、基于 clamp 后 lower 的 upper、闭合中心域、border 索引及自定义 fill 保留。显式 fill 的域外中心直接写 fill，`fill=None` 则仍计算完整 border 八角。线程数不超过当前 Torch 与 NumBa 池上限，退出时恢复原 NumBa mask；这是验证原型，正式 wrapper 的安全门另行绑定。

## 实际精度门

复用 Morph 任务固定的真实完整 MRI、已保存原输入/矩阵和原版网络输入。两个 native 源 shape 均为 `(224,288,288)`，先按原 CPU 规则由 F64 解码数组转 FP32。192 与 256 各两幅网格，合计 **47,710,208 值**：

| 对照 | 数值/位模式差异 | 最大误差/RMSE |
| --- | ---: | ---: |
| 全部 raw 采样 vs v29 成熟八角 Torch | 0 / 0 | 0 / 0 |
| 全部归一化网络输入 vs 保存原版输入 | 0 / 0 | 0 / 0 |
| 192 两幅 raw vs 独立原版 raw（14,155,776 值） | 0 / 0 | 0 / 0 |

256 未保存独立官方 raw 数组，所以其官方直接 raw 对照未测；已通过同 extent 完整官方 normalized 输入及 v29 raw 对照。实际各输入文件、矩阵、预期/候选数组 SHA 见 [replay.public.json](replay.public.json)。raw 与 normalized 门分别评分，没有用仅归一化相同替代 raw 检查。

27 项合同样例覆盖 B=2/C=3、非连续输入、半整数、clipping、极远域外 `±1e30`、0/自定义 3.25/None fill，均与 v29 逐位相同且输入不变。它们是局部合同测试，不代替真实 benchmark。实际 nodecw7 NumBa 0.61.2 生成的两个数组布局签名汇编未发现融合乘加；LLVM/汇编 SHA 已保存。本地 NumBa 0.67 同样通过有限边界门。

## 八核 microbenchmark

nodecw7 同一 `0,4,8,12,16,20,24,28` 核组，公共 `nodecw7.synth.cpu8.lock`；OMP/MKL/OpenBLAS/NumBa/Torch 均为 8，interop=1，GPU hidden。各图暖调用 v29→候选→候选→v29；计时包括原坐标生成、candidate finite-coordinate guard、分配和全部采样。读取、哈希、评分在计时外，没有额外保存影像。

| 每图暖调用中位数 | v29 Torch | NumBa 原型 | 比值 |
| --- | ---: | ---: | ---: |
| 192 | 0.934859 s | 0.156391 s | 5.98 |
| 256 | 2.679885 s | 0.516106 s | 5.19 |

空 NumBa cache 的 192 第一次函数调用包含 JIT：原型 **2.045279 s**，v29 **1.341854 s**；首次导入 NumBa 不在这两个函数时钟中。

256 单独新进程、另一空 cache 的有限 probe：v29 首调用 **2.736543 s**，原型首调用含 JIT **2.699238 s**，新增懒导入 **0.211535 s**，导入加首调用为 **2.910774 s**。原型输出 SHA 与已通过的同输入 full raw SHA 相同，实际坐标 SHA 为 `9063d673b789b4748d75383e9990580b7cf72dc4fdd36120d79be12d9efee08c`。完整记录见 [cold256.public.json](cold256.public.json)。

共享节点 load 为 88–101，时钟仍会波动。完整 v1 验证命令 GNU wall **39.96 s**，包含全部文件读取、比较/哈希、冷 JIT 和边界测试；peak RSS **3,978,309,632 B** 包含多臂同时保存的参考/候选数组，不能当作单候选峰值。没有新跑整段 joint/CNN/GPU，不宣称完整 SynthMorph 提速。真实 cold/warm 收益只支持继续评估 CPU 采样接入。

## 复现与后续接入

[run_replay.sh](run_replay.sh) 记录固定索引定位的 v29 freeze、实际参考目录、线程、核组和锁。源码在 canonical `workspaces/.../remaining_20261004/synth/morph-raw-sampler`；私密产物在对应 `runs/.../synth/morph-raw-sampler`。未改变既有环境 prefix 或 Morph 活跃 freeze。

root 已批准推进 CPU 限定安全 wrapper。Morph 任务负责复制新 helper 并作最小接入：保留现有 CPU/F32/no-grad/eval/hook/autocast 路径，原 einsum 坐标生成后懒调用；不可用、禁用、非有限或编译失败时继续原 Torch 八角正文。最终 helper 改名/安全包装后的 SHA 与本原型不同，须重做同 47,710,208 值门和 cold/warm 绑定；本页原型记录保持原样，不改标为正式生产源码验收。

## 安全 wrapper 交接

[_cpu_raw_sampler_ready.py](_cpu_raw_sampler_ready.py) SHA `041ac65195eac90e8c4d36bccf9764f69b0ea28c6b17fb4a696966b3ebf7696d` 交由 Morph 任务复制为正式新模块，此工作树仍只改 validation。它的 `try_sample(volume,locations,fill_value=0)` 返回新 Tensor，或返回 `None` 由调用者继续原 Torch 正文。保留上层模型 eval/hook guard；helper 检查 CPU/Linux x86/F32/no-grad/autocast-off、非有限值和采样网格大小，不接收 CUDA。

小于 32,768 个位置、显式 `FNIT_SYNTHMORPH_CPU_RAW_NUMBA=0`、NumBa 缺失/JIT 禁用或编译/运行失败时保留 Torch；NumBa 错误本进程记录一次，后续不重复失败。新 wrapper 预算取进入前 NumBa mask 与当前 Torch 线程数的较小值，并在 `finally` 恢复 mask。私有 `backend_info()` 可验证实际后端与原因。

[test_ready_helper.py](test_ready_helper.py) 本地 **21 passed，6.98 s**，覆盖真实采用的 NumBa kernel 路径、多 batch/channel、非连续数据、singleton 源轴、fill=None/custom、非有限/禁用/NumBa 不可用、grad/autocast、失败一次缓存和 mask 恢复。源数组实际 F64/F32 copy 都是 F-contiguous；硬件/ISA 与布局见 [platform.public.json](platform.public.json)。这些安全测试不替代最终生产 helper 的实际 full 输入门，下一份记录将绑定 Morph 复制后的 source SHA。
