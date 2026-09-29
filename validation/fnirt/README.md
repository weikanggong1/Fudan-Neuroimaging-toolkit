# TorchFNIRT 真实数据验证

[返回功能文档](../../docs/fnirt/README.md)

本目录按预设和源码哈希记录真实数据对照。

## FSL 无配置默认值：2026-09-29

同一真实去脑 T1、MNI152 2 mm 模板和 FSL FLIRT 初始矩阵分别输入无 `--config`
的 FSL FNIRT 与 FNIT `default`。联合优化版本的 warped T1 脑内 Pearson r 为 0.99584，
支持区 Dice 为 0.99875，coefficient 全数组 Pearson r 为 0.99480。两份系数图的
shape、intent-2007、输出网格、gzip CRC 和有限值检查通过。FNIT 配准与写出
44.80 s，GPU 峰值分配 1.089 GB；FSL CPU 进程 253.51 s，但写出有效文件后
返回 255，因此仅作条件性对照，也不发布加速比。输入与源码哈希、指标定义见
[默认预设报告](default_preset_20260929.public.json)，执行和比较脚本见
[`validate_default_real.py`](validate_default_real.py)。

## Oxford TBSS/FA：2026-09-28 的候选

1 例去标识化真实 UKB 格式 FA
同时输入当时的 TorchFNIRT 候选和 FSL 6.0.7.4。两侧固定相同的 preprocessed FA、
FMRIB58_FA_1mm、FSL scaled-mm affine、implicit zero mask 和 Oxford s1/s2/s3 配置。
FLIRT registration weight 不传给任一 FNIRT 实现。
该 FA 报告对应当时的源码快照，不作为本次预设改动后的新测量；旧版包入口的
[源码等价分析](../runtime_dependencies/package_entry_source_equivalence.public.json)
只覆盖报告注明的调用路径与版本。

- [`report.real.current.json`](report.real.current.json)：输入、配置、源码和脚本 SHA-256，四类输出合同与完整 3D 指标，计时、显存、负载及验收结论。
- [`run_current_matched.py`](run_current_matched.py)：当时 TorchFNIRT 候选的单例执行脚本。
- [`run_official_matched.sh`](run_official_matched.sh)：FSL 三阶段参考命令及 full-affine Jacobian 生成命令。
- [`validate_real_current.py`](validate_real_current.py)：比较、报告和图生成脚本。
- [`SHA256SUMS`](SHA256SUMS)：上述文件与公开图的校验值。
- [真实 FA 与 Jacobian 对照图](../../docs/fnirt/figures/fnirt_real_current.png)。

四类输出的 shape、affine、float32 和 coefficient intent-2007 合同通过。Pearson r 为：
coefficient `0.999893`，warped FA `0.999203`，nonlinear Jacobian `0.999064`，含 affine
Jacobian `0.994737`。MAE 分别为 `0.018804`、`0.003795`、`0.010260` 和
`0.037611`。这些误差超过浮点舍入，所以 `numerical_equivalence_passed=false`。

TorchFNIRT 的同步核心、外部 wall 和 peak CUDA allocation 分别为 `16.933 s`、
`25.16 s` 和 `3.598 GB`。FSL 三阶段外部 wall 合计 `1265.14 s`。GPU 上有空闲但常驻
的其他进程，FSL 启动时主机 load average 为 `94.86/89.93/88.62`；两侧进程边界也不同，
因此报告保留原始时间，不给出加速比。该 FA 对照只有 1 例。
