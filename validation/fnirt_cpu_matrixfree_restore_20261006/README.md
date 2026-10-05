# FNIRT CPU：恢复 callback 的有限真实方向比较

## 1. 功能与结论

从 [stage1 accepted level checkpoint](../fnirt_cpu_level_rehydrate_20261006/README.md) 恢复两套独立 CPU 状态，比较当前 optimized callback、强制现有 pointwise reference 分支和既存 current H 的 strict CSC action。唯一 worker exit0：三个原单位列逐位恢复，七个方向的 optimized/reference 结果全部逐位相等；四个真实保存方向对 CSC 有很小的浮点尾差。

本轮没有 evaluate、linearize、gradient、新 H 组装、PCG、原生进程或完整配准。生产源码未改，也未接受默认 CPU 接线。流程：来源绑定 → 显式恢复值与 stride → 两独立 cache → 三列精确恢复门 → 四个实际保存方向 → 操作数/flags/来源后置检查。结果见 [ACCEPTANCE](ACCEPTANCE.public.json)、[原始 summary](run/stage2/summary.public.json) 和 [解释边界](INTERPRETATION.public.json)。

## 2. Python 调用、输入和输出

这是私有诊断 harness，不是生产 API。参数见 [expected](source/expected.public.json)。输入均来自已保存的当前 FNIT 状态，未读 native 解/A/RHS：

| 输入 | 格式与意义 |
| --- | --- |
| accepted checkpoint | 私有 NPZ，68数组、2647508B，SHA `78abee85…`；F32影像状态、bool mask、F64系数/basis/weights/Gram/梯度等，只在服务器读取 |
| producer summary | stage1 metadata，SHA `eb31cfd1…`；当前 solve3 第二 accepted 点、有效λ、count、原 byte-stride 和 checkpoint 身份 |
| current full H/g/diagonal | 小端F64；H SHA `426e0d4a…`、g `79e5d87f…`、独立 diagonal `833b3d6c…`；H仅加载为第三比较，不重新组装 |
| old/new解 | 同 current A/g 的既存两解，SHA `58579e0f…` / `a798a3cc…`；仅作有限方向，不求解或读原生参考解 |
| 当前源码 | 17 FNIRT +coordinates +既有 strict CSC源，共19项；输入8项，合计27项前后绑定 |

维度1177由已存g长度确定；3×7×8×7位移系数加全局强度比例为混合坐标单位，不能统一写成毫米或voxel。callback full结果为 `2 * matvec(direction)`，与full H单位一致；本轮不加damping，也不计算A。

`restore_callback.py` 从当前 `linearize` 抽取原 `bend_normal`、`data_normal`、`matvec` AST及其closure变量顺序，未调用外层方法。Bending通过`__new__`安装已存Gram等字段，构造函数计数0；原展开、adjoint、bending.normal、pack/unpack及scale求和不改。

```python
# callback由已审的restore工厂返回；这里只展示当次完整变量意义。
# direction：CPU F64、无grad的一维既存方向，长度由saved_gradient决定。
# full_action：当前H坐标中的完整曲率action；没有PCG或更新参数。
full_action = 2 * restored_matrixfree_callback(direction)
```

optimized臂使用新`SpatialNormalCPU`；reference臂只强制`cpu_normal=None`，其余原body相同。两个臂分别恢复68数组，互不共享storage，所有操作数前后值SHA不变。NPZ加载后有4项stride与写前记录不同（coefficients、bending_diagonal、state_field、state_gradient_fsl）；显式`empty_strided`恢复后，全部shape/dtype/stride及逻辑值逐位核对通过。原地址、storage offset、alias图未持久化，不能称原Python cache复原。

输出仅scalar、schema、源/值SHA、差异统计和时钟。私有输入数组未下载、未公开。optimized最终packed layout为`[2,0,1]`，copy1548288B，scratch shape`[3,24,28,24]`、stride`[672,28,1,2016]`；reference臂无此cache。

## 3. 命令行与复核

只读公开报告复核，不导入科学依赖或启动worker：

```bash
python validation/fnirt_cpu_matrixfree_restore_20261006/verify_report.py
```

当次 [runner](source/run_stage2.sh) 在nodecw7共用CPU锁后取自身inner lock，实际worker命令为：

```bash
# fnit_root：现场核过README/INDEX的统一FNIT根目录。
# callback_workspace：本次冻结源码、expected和PLAN所在新叶。
# callback_output：必须不存在的新私有输出目录。
python stage2_callback.py --root "$fnit_root" \
  --workspace "$callback_workspace" --output "$callback_output" --approved-stage2
```

参数`--root`解析canonical来源，`--workspace`读取冻结声明，`--output`写scalar报告，`--approved-stage2`防误启动，不能替代授权。原任务已完成一次，无需重跑。

实际controller160006，共用CPU锁、八个physical核32,36,…,60，线程8/interop1；child180s、controller19600s、outerwait19000s、20GB address-space cap。代码600、目录700、umask077；GPU不可见，禁loader overrides。科学失败仍执行after preflight，科学非零RC优先保留；科学成功而postcheck失败也非零退出。真实science/post/overall RC均0，见 [launch](run/launch.public.json)、[exitcode](run/stage2.exitcode)、[最终INDEX](run/index_completed.public.json)。

## 4. 对应原软件

这是FNIRT内部callback诊断，没有独立官方CLI。本轮没有执行原生软件；第三比较是当前FNIT物化H，不能称官方同系统求解oracle。完整FNIRT调用与既有真实原软件benchmark见 [功能说明](../../docs/fnirt/README.md)。

strict CSC使用既有自有`column_matvec`原body AST，Numba cache=False、fastmath=False，按列顺序mul/add。它与复合spline callback的浮点执行路径不同。开始/结束FSL DSO映射快照为空，仍只代表两个记录时刻，不是全程系统调用审计。

## 5. 精度、状态门及计时

| 方向 | optimized/reference bits差 | optimized/CSC bits差 | 对CSC maxabs | 对CSC relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 原单位列0 | 0 | 0 | 0 | 0 |
| 原单位列588 | 0 | 0 | 0 | 0 |
| 原单位列1176（强度） | 0 | 0 | 0 | 0 |
| saved current full g | 0 | 877 | 3.552714e−15 | 2.839670e−19 |
| saved old解 | 0 | 1128 | 1.776357e−15 | 1.804498e−16 |
| saved new解 | 0 | 1143 | 1.776357e−15 | 1.787853e−16 |
| saved new−old | 0 | 1107 | 1.075529e−16 | 1.354571e−14 |

三列原callback与对应既存dense H列逐位相等，包含signed zero。它只证明三个选中列，不能推广至全部列或任意方向。七个方向optimized/reference全等支持该冻结状态中pointwise控制无观测差异。对CSC的小差如实记录，不自动称算法bug、全链精度修复或新的配准接受门。

实际optimized7/reference7、CSC7。evaluate/linearize/gradient/Bending构造/solver/native均0；完整H未重新物化。27项输入/来源、每臂68操作数值SHA、精度flags前后相同；CUDA未初始化，cuDNN TF32=True、matmul TF32=False、默认F32、grad_enabled=True。完整grad/diagonal仅从已存half值恢复到full单位核字节，没有重新求导。

导入及绑定1.913258s、两套checkpoint恢复/factory0.089968s、worker至后置绑定/summary写前3.514521s；maxRSS439464KiB。首次optimized callback1.203449s、首次CSC0.219729s可能含冷JIT/打包，其后时钟完整列在summary。本轮按数值控制顺序运行，不能用这些数计算速度比；address-space cap也不是物理内存峰值。

这是有限内部状态诊断，没有新的配准脑图输出；不替代真实整配准脑图或准确性验收。已有current系统PCG的53/49轮结果仍属于H回放，不能转写成生产matrix-free求解结论。

## 6. 版本、失败与绑定记录

1. 初始准备freeze `92fbbfdc…`未上传或执行；root要求失败收尾后冻结`782927b4…`。旧worker/runner及freeze完整保留在 [pre_bookkeeping](source/pre_bookkeeping/freeze.public.json)，[history](source/PREPARATION_HISTORY.public.json) 记录只改bookkeeping、全部内部数值AST不变。
2. 远端统一目录现场HEAD为`f75e724653cac80e04d29b71df8eaf0f66f95bfb`，前后源码SHA一致；不以Git头同一证明cache。唯一controller自然exit0，无数值失败或重试。
3. 整包只读metadata采集一次返回relay125；原stderr未留，分为三个小响应后成功。relay编码大小上限与因果证据边界见 [FAILURES_AND_STATUS](FAILURES_AND_STATUS.public.json)。科学worker未重复，源/结果未改。
4. [COLLECTION_BINDINGS](COLLECTION_BINDINGS.public.json) 核27个远端as-run文件，原summary SHA`5a9ae71f…`。INDEX六锁登记/完成保留其它任务字段及权限；[MANIFEST](MANIFEST.public.json) 绑定公开叶，自身除外。

冻结PLAN的“未运行”字样是准备时状态，原文保留；最终summary、exit和completed INDEX为执行完成证据。上传与收集脚本是stdio调度/metadata操作，不进入科学worker。

## 7. 参考与后续边界

当前FNIT：[registration](../../src/fnit/fnirt/registration.py)、[spline](../../src/fnit/fnirt/spline.py)、[CPU normal](../../src/fnit/fnirt/_normal_cpu.py)。原方法：[FSL FNIRT](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt.html)；Andersson、Jenkinson、Smith，*Non-linear registration, aka spatial normalisation*，FMRIB Technical Report TR07JA2（2007）。

本轮完成有限callback恢复；尚未验证matrix-free PCG、原生轨迹、非线性估计或默认CPU精度修复。后续只准备两臂实际matrix-free PCG计划/冻结源码，必须另行审查后才能运行；当前没有新求解或生产接线。
