# EDDY 2111 overlay 源码对照

用户提供的 `eddy_fsl2111_strict_overlay.zip` 是待验证候选。其校验清单通过，但附带状态文件明确说真实数据验收尚未完成。本次将它与 [FSL EDDY 2111.0](https://git.fmrib.ox.ac.uk/fsl/eddy) 及 [miscmaths Simplex](https://git.fmrib.ox.ac.uk/fsl/miscmaths) 的实现逐段对照，再用 gpucw1 的 `eddy_cuda10.2` 做同输入输出验证。源码目标提交为 `ecfef26151c2613d0f4e1b45dcbbe100b58db50c`；没有把此提交误写成服务器二进制的已核实构建号。

| 组件 | ZIP 候选的差异 | 本版处理 |
|---|---|---|
| Gaussian process | 使用旧的逐 shell 球面协方差；选点、平滑及残差预测和 CUDA 路径不同 | 改为默认 NewSpherical 协方差、相同 GP 随机选点及 shell 均值；按 CUDA predictor 的零值处理方式预测 |
| GP 超参数 | SciPy Nelder–Mead 的步骤和终止判据与 FSL 不同 | 按 FSL `Simplex` 的 unit 起始步长、反射/扩张/收缩及相对代价停止条件实现 |
| 运动与涡流更新 | 原候选的目标掩膜、平滑边界和参数更新与 FSL 注册循环不同 | 分开处理 GP 脑掩膜与扫描共同有效视野；使用 FSL 掩膜平滑、有限差分导数及法方程；缓存不随参数变化的网格和样条系数 |
| TOPUP 与重采样 | 场插值、位移反演、Jacobian 和最终输出有效掩膜使图像误差累积 | 区分 DerivativeCalculator 与最终 FieldGpuUtils 的反演路径；使用 cubic B-spline Jacobian；修正最终重采样的 PE 边缘外推及输出掩膜 |
| 离群复查 | 原候选最终复查沿用 `--ff=10` | 对照 `eddy.cpp`，最终 `FinalOLCheck` 暂时改用 `ff=1`，随后恢复 |
| restricted movement RMS | 原候选把位移向量的 PE 分量清零 | 对照源码，只把 PE 平移参数清零，再计算完整 3D 位移 |
| TF32 | 全局禁用 TF32 与 FNIT 约定不符；直接全开会降低场系数和刚体有限差分精度 | 默认允许 TF32，仅在 TOPUP 系数展开和小步长刚体矩阵乘法中临时使用完整 float32 |

FSL 最终输出的 PE 边缘有效掩膜与单独读取的 `--epvalid=False` 配置不能仅凭参数名推断一致。本版最终重采样按实际二进制输出边界处理；GP 拟合和注册阶段仍使用各自的有效掩膜。把官方 `.eddy_parameters` 放入重采样器时，选取的真实 AP 帧达到接近逐值一致；完整运行后的剩余差异主要来自联合参数估计、离群决定和 FSL GPU 的数值实现。

[最终真实数据指标](README.md)在预设阈值内，但仍不是逐体素相同；14/15 的离群切片重合及运动 RMS 误差清楚限定了这一结论。`src/fnit/eddy/core.py` 的旧近似实现已经移除，公开 `TorchEDDY` 指向这一个数值路径。
