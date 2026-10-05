# Robust target-preparation A：已冻结，真实对照待授权

这里只实现并准备比较官方 robust 注册之前的目标掩膜及右侧 atlas 头信息。没有注册、mesh 拟合或 pipeline 接线，默认 CPU/GPU 路径保持。

- 34 项局部预处理合同、5 项仅 stub 的进程边界合同通过；见 `CONTRACTS.public.json`。
- 冻结源为原 clean main `db61cebc` 的498个现有 Python 文件加2个新模块，500文件树 SHA `4422320dcae48c251c849c4f9736ac5bd5ae304983e4e9ab6e3436032a52e5a6`。只复制本仓库 Python 代码，没有影像、权重或原软件可执行程序。
- 官方 arm 从固定安装源 AST 截取 `preprocess_images`43–66行和 `align_atlas_to_seg`224–261行，到目标保存即止；不导入 SAMSEG/GEMS，不执行注册。该 arm 的 Surfa/SciPy 仅存在于独立官方对照环境。
- FNIT arm 仅调用独立 PyTorch/nibabel 前处理模块，随后独立 score arm 比较保存后的两份 MGH。三臂各是新的 CPU 进程，共用8物理核的同一锁。
- 工作空间和运行目录实际权限0700、文件0600。子进程 AS 上限20,000,000,000字节、单臂180秒，等待锁600秒，controller总上限1200秒。清空 CUDA 可见设备；移除外部动态库/Python路径和 OpenBLAS 固定核心类型，8线程固定；启用不写 `.pyc`。失败/超时/取消停止其自有子进程，并保留原返回码。
- Worker 在 `finally` 保存输入、官方源及500个冻结源的前后SHA校验、实际 flags/affinity/RSS/时间、是否误导入原生GEMS及0注册/0拟合范围。科学返回码与后校验返回码分别保存，不把设置失败计作数值差异。
- 正式门：目标及反射 atlas **0个不同体素**，shape、dtype、MGH 几何/扫描字段和存储 affine 全相同，官方前景非空。不比较 gzip 字节或可选MGH标签。当前**尚未执行真实 worker**，没有真实精度或速度结论。

一次官方/FNIT顺序观察不是ABBA或稳定速度 benchmark。既有真实case为1mm，不能用它声称高分辨率resize编译分支已验收。34项小合同用于边界与接口，不替代真实MRI对照。

Python输入、输出、全部参数、示例、对应原软件两次注册命令以及许可和文献见[功能说明](../../../docs/robust_register/PREPARATION.md)。前一完整右HA候选已经完成但最终28区验收未过；这里不会读取其细分输出反拟合或启动第二整例。
