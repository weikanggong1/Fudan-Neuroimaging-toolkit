# FLIRT 移除 Surfa：真实 T1 同输入核对（2026-09-28）

## 验证对象

本次只更换 PyTorch FLIRT 的图像载入和返回体类型：路径输入从 Surfa 转为仓库内 NiBabel `load_volume`，内存输入按 `.data`、`.geom`、`.new` 协议接收；算法、搜索参数和输出写盘逻辑未改。以旧版包为迁移基准，另将官方 FSL 6.0.7.4 作为算法精度的外部参照。完整参数和结果结构见[功能文档](../../docs/flirt/README.md)。

输入是一例真实 T1 的原始体素每轴隔一取一所得 2 mm 图像，形状 128³，SHA256 `bd58c5197c080f27529bd6dd8c40147991494f391f9bfa3aaca48035cd6cd864`；没有生成模拟影像。固定图像为 FSL 6.0.7.4 的 `MNI152_T1_2mm.nii.gz`，形状 91×109×91，SHA256 `0585cd056bf5ccfb8bf97a5f6a66082d4e7caad525718fc11e40d80a827fcb92`。输入图像和模板均不随仓库发布。

## 命令和输入输出

下面的 `-in` 是待移动三维图像，`-ref` 是固定模板；`-out` 生成参考网格上的图像，`-omat` 生成输入到参考的 FSL scaled-mm 4×4 矩阵；`-dof` 和 `-cost` 指定验证配置，`--device cpu` 固定运行设备。旧版和新版使用相同参数，仅改变包版本。

```bash
fnit flirt -in real_t1_2mm.nii.gz -ref MNI152_T1_2mm.nii.gz \
  -out registered.nii.gz -omat affine.mat -dof 6 -cost normmi --device cpu
fnit flirt -in real_t1_2mm.nii.gz -ref MNI152_T1_2mm.nii.gz \
  -out registered.nii.gz -omat affine.mat -dof 12 -cost corratio --device cpu
```

官方 FSL 同输入命令，仅将 `fnit flirt` 换成 `flirt` 并去掉 `--device cpu`。API 使用 `run_flirt(input=..., reference=..., output=..., omat=..., init=None, inweight=None, refweight=None, dof=6/12, cost="normmi"/"corratio", device="cpu", overwrite=False)`；各命名参数与命令行相同，`init` 是可选初始矩阵，两个 `weight` 是相应图像网格上的权重，`overwrite` 控制已有输出的覆盖。返回的 `moved` 是参考网格图像，`matrix` 是 FSL 坐标矩阵，`moving_to_fixed_world` / `fixed_to_moving_world` 是 world-RAS 仿射，`qc` 记录代价和运行信息。

## 新旧包配对结果

同一 Conda 环境、headcw CPU、8 个 PyTorch 线程，每个配置的旧版和新版各执行一次。墙钟时间含进程启动、载入、配准与写盘。输出的 NumPy 数组逐值比对；`.mat` 与 `.nii.gz` 按 SHA256 比对。

| 配置 | 旧版 API | 新版 API | 矩阵 / 体素差异 | 保存文件 |
|---|---:|---:|---:|---|
| 6 DOF / normmi | 33.82 s | 28.30 s | 0 个不同值；仿射差 0 | `.mat` 和 `.nii.gz` 字节相同 |
| 12 DOF / corratio | 52.61 s | 61.04 s | 0 个不同值；仿射差 0 | `.mat` 和 `.nii.gz` 字节相同 |

两版的最终代价与评估次数也相同：6 DOF 为 `-1.0511478185653687`、7,550 次；12 DOF 为 `0.4517233073711395`、12,685 次。禁用 Python 的 Surfa 导入后，`fnit flirt` 的 6 DOF 命令仍完成，输出字节与旧版相同；Surfa 旧内存体能够直接作为输入。禁用导入的针对性测试结果为 **20 passed, 1 skipped**。两种模式各只有一次计时，不能据此判定移除 Surfa 的速度收益。

## 官方 FSL 同输入对照

官方 FSL 6.0.7.4 的完整命令墙钟时间分别为 24.75 s（6 DOF）和 12.17 s（12 DOF）。矩阵差用移动图像的 5×5×5 体素网格点，换算到 world-RAS 后计算两种配准结果的位移差；图像指标对完整参考网格体素计算。

| 配置 | 矩阵位移 RMS / 最大 | 图像 Pearson | 图像 MAE / RMSE | 输出类型 |
|---|---:|---:|---:|---|
| 6 DOF / normmi | 0.4065 / 0.4419 mm | 0.99608 | 0.9934 / 2.7882 | FSL uint8，FNIT float32 |
| 12 DOF / corratio | 1.0658 / 1.8799 mm | 0.99886 | 0.8510 / 1.7530 | FSL uint8，FNIT float32 |

两者图像尺寸和仿射一致，矩阵及图像数值没有达到逐值一致。旧、新 FNIT 文件完全相同，说明此差异来自原有 FLIRT 实现及输出类型约定，而非本次 Surfa 迁移。旧版[10 例 GM 基准](../flirt/report.public.json)在另一输入集合取得 12 DOF 矩阵 RMS 最大 0.0290 mm；它尚未以新版重新执行，不能代替当前 T1 个例结果。

## 范围与限制

本次没有测试新路径的 GPU 配对：在 gpucw1 GPU1 初始化时 CUDA 报 OOM，随后按同一 Conda 环境转为 CPU；没有占用其他 GPU 或重复尝试。没有重新测试权重图像、自定义 `-init`、异常头文件、其他参数组合或旧 10 例全套基准。机器可读结果见[`report.json`](report.json)。
