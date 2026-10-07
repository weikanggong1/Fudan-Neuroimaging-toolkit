# Robust target-preparation A：同例数据和几何门通过

本阶段只比较robust注册前的目标掩膜和右侧atlas头信息。**两份保存图像均0个不同体素，shape/dtype/13个MGH字段/存储affine全exact。** 目标39×45×56、前景5835；反射atlas131×241×99、前景512815。原四保存文件在两次只读阶段前后SHA一致。没有注册、GEMS拟合、CUDA计算或生产pipeline接线。

- 真实as-run源：`91e83165381ae4dfc4be1c2c71bee208b20c2d1d`，worker `8e637550…`、原frozen500 Python tree `4422320d…`，原计划 `4c83935e…`。最初 `BINDINGS.public.json` 的 `prepared_not_dispatched` 是当时冻结状态；最终完成状态由 `RESULTS.public.json` 表明。
- 原官方 arm exit0。原FNIT产生并保存两MGZ后因report裁剪坐标 `np.int64` 序列化失败exit1；随后首次只读score因MGH shape `np.int32` 同类错误exit1。两个原controller返回码不改写，partial按原字节在私密目录保存为 `.partial.txt`，不是完整JSON。
- 仅从原4MGZ补录缺失scalar/header，没有重新准备图像。修复后的metadata writer与两处shape整数只改变报告；原500冻源不动。补录exit0/正式门PASS；原/修复SHA及AST边界见 `RESULTS.public.json`。
- 补录总54.505秒包含54.038秒等共用CPU锁，实际scalar/header读取比较0.202秒；它不是prep时间。原完整/partial中真正持久化的时钟及字节偏移见 `CLOCK_PROOFS.public.json`；缺失阶段时钟和FNIT RSS标NA，未复原完整旧report。
- 模块合同共35个不同case（原34批次、新增1JSON合同分别执行），另1个评分/原子JSON合同；5个stub runner边界合同原样保留。没有一次35项重跑/新MRI计时。见两份 `CONTRACTS*.public.json`。
- 两个API的pixel/geometry计算AST unchanged；新模块report的bbox坐标显式int、scorer两个MGH shape维度显式int。报告先序列化再原子no-clobber发布。默认CPU/GPU source和环境保持。

![CC0保存目标掩膜三视图](preparation_targets.png)

## 运行边界与来源

- 冻结源为原 clean main `db61cebc` 的498个现有 Python 文件加2个新模块，500文件树 SHA `4422320dcae48c251c849c4f9736ac5bd5ae304983e4e9ab6e3436032a52e5a6`。只复制本仓库 Python 代码，没有影像、权重或原软件可执行程序。
- 官方 arm 从固定安装源 AST 截取 `preprocess_images`43–66行和 `align_atlas_to_seg`224–261行，到目标保存即止；不导入 SAMSEG/GEMS，不执行注册。该 arm 的 Surfa/SciPy 仅存在于独立官方对照环境。
- FNIT arm 仅调用独立 PyTorch/nibabel 前处理模块，随后独立 score arm 比较保存后的两份 MGH。三臂各是新的 CPU 进程，共用8物理核的同一锁。
- 工作空间和运行目录实际权限0700、文件0600。子进程 AS 上限20,000,000,000字节、单臂180秒，等待锁600秒，controller总上限1200秒。清空 CUDA 可见设备；移除外部动态库/Python路径和 OpenBLAS 固定核心类型，8线程固定；启用不写 `.pyc`。失败/超时/取消停止其自有子进程，并保留原返回码。
- Worker 在 `finally` 保存输入、官方源及500个冻结源的前后SHA校验、实际 flags/affinity/RSS/时间、是否误导入原生GEMS及0注册/0拟合范围。科学返回码与后校验返回码分别保存，不把设置失败计作数值差异。
- 正式门固定为0不同体素、shape/dtype/MGH几何扫描字段/存储affineexact与非空；补录已通过原门，未放宽。不比较gzip字节或可选MGH标签。


当前例原ASEG为1mm，没有触发resize；不能据此声称高分辨率编译分支验收。单次prep观察的官方clock含编译/保存后重读，FNIT clock不含该重读；没有稳定speedup结论，也没有affine/最终ROI结论。

Python输入、输出、全部参数、示例、对应原软件两次注册命令以及许可和文献见[功能说明](../../../docs/robust_register/PREPARATION.md)。公开图仅CC0粗分割派生的二值目标，不发布影像或模板数组。下一B仅草拟源码/许可/参数和组件验收计划，真实计算另审。
