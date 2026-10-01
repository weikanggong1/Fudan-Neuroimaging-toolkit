# MCFLIRT 真实帧控制

`first8_cpu.public.json`、`first8_cuda.public.json` 及 `first8_warm.public.json` 记录同一例真实 UKB BOLD/SBRef 的前 8 帧估计，参照为同数据的原 MCFLIRT 完整 490 帧运行。脑内距离只比较前 7 帧：原代码 8 mm 阶段没有给最后一帧传递初值，截取片段会改变最后一帧的实验条件。

`first8_sampling.public.json` 固定原矩阵文本，单独核对最终样条和 int32 截断：脑内 RMSE 0.2816、逐体素时间 r 均值 0.9999987，中位数 1。矩阵文本量化也可能影响整数边界，未将所有余差归为采样差异。

`full490_cpu.public.json` 使用完整 490 帧 BOLD 和同一 SBRef，4 线程 CPU 估计 387.60 秒、45,794 次 cost。比较覆盖全部 490 帧：逐帧脑内 pull RMS 均值 0.00906 mm、中位数 0.00864 mm、最大 0.02603 mm。旋转参数 RMSE 为 0.0000787–0.0001182 rad，平移参数 RMSE 为 0.003109–0.003576 mm。相对旋转角在求角度前将 3×3 部分通过 SVD 投影到 SO(3)，避免文本精度与 float32 的非正交误差影响 `acos(trace)`。计时包括路径输入解压和估计，不含重采样、写盘、哈希及对照计算；原独立 MCFLIRT 命令的 326.10 秒包含最终采样和写盘，两者不构成相同计时边界的加速比。

`full490_gpu.public.json` 从冻结的 `1eb9c417` 完整 GPU 流程读取已捕获的运动输出、490 个矩阵和参数，不另跑估计。脑内时间 r 均值 0.99957825、中位数 0.99983382，SD 图 r 为 0.99999784。原和 FNIT 的运动输出都是 int32，FNIT 脑内非整数比例为 0；矩阵 pull RMS 均值 0.00881 mm、最大 0.03008 mm。统计使用 CPU，先计算 EPI 指标再变换三维指标图；CPU 指标计算与展示映射的计时不是 GPU 运动模型耗时。完整 FEAT 阶段为 973.119 秒，其中还包括脑提取、缩放和高通；本次没有单独计时 MCFLIRT 估计和采样，因此不将 FEAT 总时间当作 MCFLIRT 函数耗时。

报告只有匿名标量、源文件 SHA-256 和输入 SHA-256。每帧矩阵保存在服务器的私有文件中，未上传原始影像或矩阵数组。JSON 中 source hash 指向当时实际测量的文件；随后增加的输出接口及检查没有用于该次测量。`first8_warm.public.json` 的同进程第二次估计观察为 CPU 5.41 秒、CUDA 11.13 秒；共享服务器负载未控制，两个 device 内各自重复的矩阵完全相同。完整链的最终源码和结果以[全流程记录](../fmri/matched_native.md)为准。

复测时先在服务器完成原 FSL MCFLIRT 运行，再执行 FNIT 的独立控制：

```bash
# 原数据仍在服务器；8帧control只输出匿名报告和私有矩阵，不生成图像。
python validation/mcflirt/compare_motion.py \
  --bold "$RAW_BOLD" --reference "$SBREF" --brain-mask "$EPI_MASK" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --output-dir "$FRESH_CONTROL_OUTPUT" --device cuda:1 --frames 8
```

完整运动估计使用同一四维输入和完整原矩阵目录，比较包括最后一帧；逐帧数组只留在私有目录：

```bash
# 四线程 CPU，计时包含输入解压、准备和估计；另行核对原程序六列 .par。
OMP_NUM_THREADS=4 python validation/mcflirt/compare_motion.py \
  --bold "$RAW_BOLD" --reference "$SBREF" --brain-mask "$EPI_MASK" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --original-parameters "$ORIGINAL_MCFLIRT_PARAMETERS" \
  --output-dir "$FRESH_FULL_CONTROL_OUTPUT" --device cpu --threads 4 --all-frames
```

不要把小样本冷运行当成完整运动校正的速度。CUDA 第一次执行会编译固定尺寸 NCC 归约；原 MCFLIRT 的命令耗时还包含最终重采样及写盘，不能直接与只估计矩阵的计时相除。

`render_comparison.py` 在同一 EPI 网格计算完整时序的均值、population SD 和逐体素时间 r，再使用同一原 BBR 矩阵与 FNIRT pull 将三维指标图变换到 MNI 展示。它不会将两份四维 BOLD 再次重采样。mean/SD 使用三线性展示，r/有效体素图使用最近邻；统计在 EPI 中计算，MNI 图只用于展示。示例参数如下，矩阵与参数目录必须包含全部帧：

```bash
# 输入均来自已经完成的原 MCFLIRT 与 FNIT TorchMCFLIRT，原始时序不公开。
# MATRIX_T1_TO_EPI_WORLD 是原 inverse BBR，PULL_MNI_TO_T1_RAS 是 MNI 网格的 RAS-mm pull displacement。
# PRIVATE_MAP_OUTPUT 保存指标图和逐帧数值；FIGURE_OUTPUT 仅为去标识化的 MNI PNG。
OMP_NUM_THREADS=8 python validation/mcflirt/render_comparison.py \
  --original "$ORIGINAL_MCFLIRT_BOLD" --candidate "$FNIT_MCFLIRT_BOLD" \
  --epi-mask "$ORIGINAL_EPI_MASK" --template "$MNI152_2MM_TEMPLATE" \
  --mni-mask "$ORIGINAL_MNI_MASK" \
  --reference-to-source-world "$MATRIX_T1_TO_EPI_WORLD" \
  --mni-to-t1-pull "$PULL_MNI_TO_T1_RAS" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --candidate-matrices "$FNIT_MCFLIRT_MATRICES" \
  --original-parameters "$ORIGINAL_MCFLIRT_PARAMETERS" \
  --candidate-parameters "$FNIT_MCFLIRT_PARAMETERS" \
  --private-output "$PRIVATE_MAP_OUTPUT" --figure-out "$FIGURE_OUTPUT" \
  --report-out "$ANONYMOUS_REPORT_OUTPUT" --source-revision "$FNIT_GIT_REVISION" --threads 8
```

`native_cost_probe.cc` 是仅用于验证的独立小程序：链接安装好的 NEWIMAGE，读取真实帧，按 MCFLIRT 的 8/4 mm 参考重采样计算四个固定平移矩阵的 NCC。它没有复制原软件代码，也不被 FNIT 运行时调用。构建需要原 FSL 开发头文件与库，使用原软件做 benchmark 时才需要。对应输入和重采样输出均应留在私有验证目录。
